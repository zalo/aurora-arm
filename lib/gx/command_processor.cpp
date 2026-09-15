#include "command_processor.hpp"

#include "../gfx/depth_peek.hpp"
#include "../gfx/hash.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../internal.hpp"
#include "dolphin/gd/GDGeometry.h"
#include "dolphin/gx/GXAurora.h"
#include "fifo.hpp"
#include "gx.hpp"
#include "pipeline.hpp"
#include "regs.hpp"
#include "resident_geometry.hpp"
#include "shader_info.hpp"
#include "texture.hpp"
#include "vertex_loader.hpp"

#include <absl/container/flat_hash_map.h>
#include <tracy/Tracy.hpp>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::gx::fifo {
namespace {
constexpr Module Log{"aurora::gx::fifo"};

u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) noexcept {
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    buf.reserve_extra((vtxCount / 4) * 6 * sizeof(u16));

    for (u16 v = 0; v < vtxCount; v += 4) {
      u16 idx0 = vtxStart + v;
      u16 idx1 = vtxStart + v + 1;
      u16 idx2 = vtxStart + v + 2;
      u16 idx3 = vtxStart + v + 3;

      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;

      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLES) {
    buf.reserve_extra(vtxCount * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == GX_TRIANGLEFAN) {
    buf.reserve_extra(((u32(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(std::array{vtxStart, static_cast<u16>(idx - 1), idx});
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    buf.reserve_extra(((static_cast<u32>(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(std::array{static_cast<u16>(idx - 2), static_cast<u16>(idx - 1), idx});
      } else {
        buf.append(std::array{static_cast<u16>(idx - 1), static_cast<u16>(idx - 2), idx});
      }
      numIndices += 3;
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    if (g_config.cpuVertexDecode && prim != GX_POINTS) {
      // Quads expanded by the CPU vertex decoder: four vertices per segment
      buf.reserve_extra((vtxCount / 4) * 6 * sizeof(u16));
      for (u16 v = 0; v < vtxCount; v += 4) {
        const u16 idx0 = vtxStart + v;
        buf.append(std::array{idx0, static_cast<u16>(idx0 + 1), static_cast<u16>(idx0 + 3)});
        buf.append(std::array{static_cast<u16>(idx0 + 3), static_cast<u16>(idx0 + 2), idx0});
        numIndices += 6;
      }
    } else {
      // One quad, drawn as an instance per segment or point
      buf.reserve_extra(6 * sizeof(u16));
      buf.append<u16>(0);
      buf.append<u16>(1);
      buf.append<u16>(3);
      buf.append<u16>(3);
      buf.append<u16>(2);
      buf.append<u16>(0);
      numIndices = 6;
    }
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
constexpr u8 CP_CMD_NOP = GX_NOP;
constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

struct FogRangeLutKey {
  std::array<u16, 10> rangeK;
  f32 rangeCenter;
  f32 renderWidth;
  u32 targetWidth;

  bool operator==(const FogRangeLutKey&) const = default;
};

struct FogRangeLutEntry {
  FogRangeLutKey key;
  std::vector<f32> factors;
};

constexpr size_t MaxFogRangeLuts = 32;
std::vector<FogRangeLutEntry> sFogRangeLuts;

struct DrawCache {
  PipelineConfig config{};
  ShaderInfo shaderInfo{};
  gfx::PipelineRef pipelineRef{};
  GXBindGroups bindGroups{};
  uint64_t bindGeneration = 0;
  GXVtxFmt fmt = GX_MAX_VTXFMT;
  u8 lineMode = 0;
  bool hasPipeline = false;
  gfx::Range uniformRange{};
  gfx::Range fogRange{};
  FogRangeLutKey fogRangeKey{};
  bool hasFogRange = false;
  GXVtxFmt lastDrawFmt = GX_MAX_VTXFMT;
};
DrawCache sDrawCache;

// Pipeline state resolved for one hash of the raw pipeline-relevant GX state.
struct PipelineMemoEntry {
  PipelineConfig config;
  ShaderInfo shaderInfo;
  gfx::PipelineRef pipelineRef;
};
// Most dirty-pipeline events re-send a material that was already resolved
// earlier; the memo lets them skip populate_pipeline_config(),
// build_shader_info() and the canonical config hash. Measured on a Mali-G52
// handheld (Miyoo Flip) on Melee's Onett stage: ~460 such events per frame,
// FIFO translation 11.7 -> 9.9 ms/frame, bit-exact output.
constexpr size_t MaxPipelineMemoEntries = 1024;
absl::flat_hash_map<uint64_t, PipelineMemoEntry> sPipelineMemo;
uint32_t sPipelineMemoMisses = 0;

// Hashes every input populate_pipeline_config() reads, which through the
// resulting ShaderConfig also determines build_shader_info() and the pipeline
// reference: primitive, vertex descriptor/format and indexed array layout, fog
// type and range flag, TEV swap table and stages, indirect stages, color
// channels, texgens, alpha compare, cull/depth/blend state, destination alpha,
// polygon offsets, write masks and the render pass sample count. Anything not
// listed here (matrices, colors, textures, lights, viewport) only feeds
// uniforms or bind groups, which are resolved separately per draw.
uint64_t pipeline_state_hash(GXPrimitive prim, GXVtxFmt fmt) noexcept {
  const auto& state = g_gxState;
  Hasher hasher;
  const auto put = [&](const auto& value) { hasher.update(&value, sizeof(value)); };
  put(prim);
  put(fmt);
  const auto& vtxFmt = state.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto type = state.vtxDesc[i];
    put(type);
    if (type == GX_NONE) {
      continue;
    }
    put(vtxFmt.attrs[i]);
    if (type == GX_INDEX8 || type == GX_INDEX16) {
      put(state.arrays[i].stride);
      put(state.arrays[i].le);
    }
  }
  put(state.fog.type);
  put(state.fog.rangeEnabled);
  put(state.tevSwapTable);
  put(state.numTevStages);
  hasher.update(state.tevStages.data(), state.numTevStages * sizeof(TevStage));
  put(state.numIndStages);
  hasher.update(state.indStages.data(), state.numIndStages * sizeof(IndStage));
  put(state.colorChannelConfig);
  put(state.numTexGens);
  hasher.update(state.tcgs.data(), state.numTexGens * sizeof(TcgConfig));
  put(state.alphaCompare);
  put(state.cullMode);
  put(state.depthFunc);
  put(state.blendMode);
  put(state.blendFacSrc);
  put(state.blendFacDst);
  put(state.blendOp);
  put(state.dstAlpha);
  put(state.frontOffset);
  put(state.frontScale);
  put(state.backOffset);
  put(state.backScale);
  put(state.clamp);
  put(state.depthCompare);
  put(state.depthUpdate);
  put(state.alphaUpdate);
  put(state.colorUpdate);
  const uint32_t sampleCount = gfx::get_sample_count();
  put(sampleCount);
  return hasher.digest();
}

// Resolves the draw cache's pipeline config, shader info and pipeline reference
// for the current GX state, reusing the memoized result when this state was
// resolved before.
void resolve_pipeline(GXPrimitive prim, GXVtxFmt fmt) noexcept {
  auto& cache = sDrawCache;
  const uint64_t key = pipeline_state_hash(prim, fmt);
  if (const auto it = sPipelineMemo.find(key); it != sPipelineMemo.end()) {
    cache.config = it->second.config;
    cache.shaderInfo = it->second.shaderInfo;
    cache.pipelineRef = it->second.pipelineRef;
    return;
  }
  ++sPipelineMemoMisses;
  populate_pipeline_config(cache.config, prim, fmt);
  cache.shaderInfo = build_shader_info(cache.config.shaderConfig);
  cache.pipelineRef = gfx::pipeline_ref(cache.config);
  if (sPipelineMemo.size() >= MaxPipelineMemoEntries) {
    // A frame's working set is a few hundred materials; start over rather than
    // track recency.
    sPipelineMemo.clear();
  }
  sPipelineMemo.emplace(key, PipelineMemoEntry{cache.config, cache.shaderInfo, cache.pipelineRef});
}

FogRangeLutKey fog_range_lut_key() noexcept {
  const auto& state = g_gxState.fog;
  const f32 logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
  const f32 renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
  return {
      .rangeK = state.rangeK,
      .rangeCenter = ((static_cast<f32>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                     1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
      .renderWidth = renderWidth,
      .targetWidth = gfx::get_render_target_size().x,
  };
}

std::vector<f32> build_fog_range_lut(const FogRangeLutKey& key) {
  std::array<f32, 10> rangeK;
  for (u32 i = 0; i < rangeK.size(); ++i) {
    const u32 source = (i & ~1u) | (1u - (i & 1u));
    rangeK[i] = static_cast<f32>(key.rangeK[source]) / 64.f;
  }

  std::vector<f32> lut(key.targetWidth);
  for (u32 x = 0; x < key.targetWidth; ++x) {
    const f32 screenX = ((static_cast<f32>(x) + 0.5f) / key.renderWidth) * 2.f - 1.f;
    const f32 offset = screenX - key.rangeCenter;
    const f32 rangeIndex = std::clamp(9.f - std::abs(offset) * 9.f, 0.f, 9.f);
    const u32 lower = static_cast<u32>(rangeIndex);
    const u32 upper = std::min(lower + 1, 9u);
    const f32 fraction = rangeIndex - static_cast<f32>(lower);
    const f32 k = std::max(rangeK[lower] * (1.f - fraction) + rangeK[upper] * fraction, 0.000001f);
    lut[x] = std::sqrt(offset * offset + k * k) / k;
  }
  return lut;
}

const std::vector<f32>& resolve_fog_range_lut(const FogRangeLutKey& key) {
  for (const auto& entry : sFogRangeLuts) {
    if (entry.key == key) {
      return entry.factors;
    }
  }
  if (sFogRangeLuts.size() == MaxFogRangeLuts) {
    sFogRangeLuts.erase(sFogRangeLuts.begin());
  }
  sFogRangeLuts.emplace_back(FogRangeLutEntry{key, build_fog_range_lut(key)});
  return sFogRangeLuts.back().factors;
}

gfx::Range push_fog_range_lut(const FogRangeLutKey& key) {
  const auto& lut = resolve_fog_range_lut(key);
  return gfx::push_storage(reinterpret_cast<const u8*>(lut.data()), lut.size() * sizeof(f32));
}

u8 line_mode_for_prim(GXPrimitive prim) noexcept {
  switch (prim) {
  case GX_LINES:
    return 1;
  case GX_LINESTRIP:
    return 2;
  case GX_POINTS:
    return 3;
  default:
    return 0;
  }
}
} // namespace

static void handle_draw(u8 cmd, ByteReader& reader) noexcept;
static void handle_aurora(ByteReader& reader) noexcept;

ProcessResult process(const u8* data, u32 size) noexcept {
  ZoneScoped;
  ByteReader reader{{data, size}};

  while (!reader.empty()) {
    const u8 cmd = reader.read<u8>();
    u8 opcode = cmd & CP_OPCODE_MASK;

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      const u32 value = reader.read<u32>();
      handle_bp(value);
      if (reg_get(value, 8, 24) == GX_BP_REG_DRAWDONE) {
        return {static_cast<u32>(reader.offset()), true};
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      const u8 addr = reader.read<u8>();
      handle_cp(addr, reader.read<u32>());
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const u32 header = reader.read<u32>();
      const u32 count = ((header >> 16) & 0xFFFF) + 1;
      const u16 addr = header & 0xFFFF;
      handle_xf(addr, reader.take(count * sizeof(u32)));
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const u32 arrayType = GX_POS_MTX_ARRAY + (opcode - CP_CMD_LOAD_INDX_A) / 0x08;
      const u16 srcArrayIdx = reader.read<u16>();
      const u16 addrLen = reader.read<u16>();

      const u16 len = (addrLen >> 12) + 1;
      const u16 dstAddr = addrLen & 0x0FFF;
      auto const& array = g_gxState.arrays[arrayType];
      const u32 srcOffset = static_cast<u32>(srcArrayIdx) * array.stride;
      const u32 srcSize = static_cast<u32>(len) * sizeof(u32);
      AURORA_ASSERT(array.data != nullptr, "indexed XF load from unmapped array {}", arrayType);
      AURORA_ASSERT(srcOffset <= array.size && srcSize <= array.size - srcOffset,
                    "indexed XF load outside array {}: offset={}, size={}, array size={}", arrayType, srcOffset,
                    srcSize, array.size);
      auto const* srcData = static_cast<const u8*>(array.data) + srcOffset;
      if (!copy_xf_data(dstAddr, srcData, len, array.le ? std::endian::little : std::endian::big)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      reader.skip(8);
      break;
    }

    case CP_CMD_INVAL_VTX: {
      for (auto& array : g_gxState.arrays) {
        array.cachedRange = {};
      }
      g_gxState.dirty |= DirtyImmediates;
      break;
    }

    case GX_AURORA: {
      handle_aurora(reader);
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      handle_draw(cmd, reader);
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        handle_draw(cmd, reader);
      } else {
        // Hex dump surrounding bytes for debugging
        {
          const size_t pos = reader.offset();
          size_t dumpStart = (pos > 17) ? pos - 17 : 0;
          size_t dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (size_t i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, reader.offset() - 1);
      }
      break;
    }
  }
  return {size, false};
}

[[noreturn]] static void handle_draw_overrun(size_t totalVtxBytes, const ByteReader& reader) noexcept {
  // Hex dump around the draw command for debugging
  const size_t pos = reader.offset();
  const size_t size = reader.size();
  const u8* data = reader.data();
  size_t cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  size_t dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  size_t dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (size_t i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, reader.remaining());
}

static u32 calc_vtx_size(GXVtxFmt fmt) noexcept {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto& attrFmt = vtxFmt.attrs[i];
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 6 : 2;
      break;
    }
  }
  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;
  return vtxSize;
}

// Resolves the pipeline for the current GX state, primitive and vertex format into sDrawCache.
static void prepare_pipeline(GXPrimitive prim, GXVtxFmt fmt) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;
  const u8 lineMode = line_mode_for_prim(prim);
  const bool pipelineValid = cache.hasPipeline && (state.dirty & DirtyPipeline) == 0 && cache.fmt == fmt &&
                             cache.lineMode == lineMode && cache.config.msaaSamples == gfx::get_sample_count();
  if (!pipelineValid) {
    const bool hadPipeline = cache.hasPipeline;
    const auto prevSampledTextures = cache.shaderInfo.sampledTextures;
    const auto prevSampledIndTextures = cache.shaderInfo.sampledIndTextures;
    resolve_pipeline(prim, fmt);
    cache.fmt = fmt;
    cache.lineMode = lineMode;
    cache.hasPipeline = true;
    state.dirty = (state.dirty & ~DirtyPipeline) | DirtyUniform;
    if (!hadPipeline || prevSampledTextures != cache.shaderInfo.sampledTextures ||
        prevSampledIndTextures != cache.shaderInfo.sampledIndTextures) {
      cache.bindGeneration = 0;
    }
  }
}

// CPU vertex decoding: converts the raw GX vertices of a draw into the frame's vertex stream with
// the loader for the draw's pipeline configuration. Lines are expanded into quads here (four
// records per segment) instead of by instancing in the vertex shader, so vtxCount becomes the
// expanded vertex count. Points keep one record each: the pipeline steps the vertex buffer per
// instance and the draw renders one instance of a shared quad per point (see prepare_idx_buffer),
// so a particle system emitting tens of thousands of GX_POINTS per frame writes each point once.
static gfx::Range push_decoded_verts(GXPrimitive prim, GXVtxFmt fmt, std::span<const u8> vertexData, u16& vtxCount,
                                     size_t alignment, u32 matrixWordZ = 0) noexcept {
  ZoneScoped;
  prepare_pipeline(prim, fmt);
  const auto& state = g_gxState;
  const auto& config = sDrawCache.config.shaderConfig;
  const auto& loader = vertex_loader(config);
  AURORA_ASSERT(loader.vtxStride != 0 && vertexData.size() == static_cast<size_t>(vtxCount) * loader.vtxStride,
                "vertex data of {} bytes does not hold {} vertices of {} bytes", vertexData.size(), vtxCount,
                loader.vtxStride);
  const size_t stride = loader.layout.stride;
  u8* out = nullptr;
  if (config.lineMode == 0 || config.lineMode == 3) {
    const gfx::Range range = gfx::map_verts(static_cast<size_t>(vtxCount) * stride, alignment, out);
    if (out != nullptr) {
      decode_vertices(loader, vertexData.data(), vtxCount, out, state.arrays, state.currentPnMtx, matrixWordZ);
    }
    return range;
  }
  const u32 instances = line_instance_count(config.lineMode, vtxCount);
  AURORA_ASSERT(instances * 4 <= 0xFFFF, "too many line/point primitives in one draw ({})", instances);
  // Decode into local memory, then stream the expanded quads out; the destination is never read.
  static std::vector<u8> decoded;
  decoded.resize(static_cast<size_t>(vtxCount) * stride);
  decode_vertices(loader, vertexData.data(), vtxCount, decoded.data(), state.arrays, state.currentPnMtx,
                  matrixWordZ);
  vtxCount = static_cast<u16>(instances * 4);
  const gfx::Range range = gfx::map_verts(static_cast<size_t>(vtxCount) * stride, alignment, out);
  if (out != nullptr) {
    expand_line_vertices(loader, decoded.data(), instances, out);
  }
  return range;
}

// Resolves pipeline, texture bind groups, uniform record and fog range table for the current GX state into
// sDrawCache and fills the parts of the immediates derived from them.
static void prepare_draw_state(GXPrimitive prim, GXVtxFmt fmt, DrawImmediateData& immediates) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;
  prepare_pipeline(prim, fmt);

  const bool bindGroupsValid =
      (state.dirty & DirtyTextures) == 0 && cache.bindGeneration == texture::current_bind_generation();
  if (!bindGroupsValid) {
    const auto prevBindGroup = cache.bindGroups.textureBindGroup;
    const bool rebound = resolve_sampled_textures(cache.shaderInfo);
    cache.bindGroups = build_bind_groups(cache.shaderInfo);
    cache.bindGeneration = texture::current_bind_generation();
    state.dirty &= ~DirtyTextures;
    // For the texture_size_bias uniform (size, LOD bias, array layer): a
    // different texture in the same array keeps the bind group.
    if (rebound || cache.bindGroups.textureBindGroup != prevBindGroup) {
      state.dirty |= DirtyUniform;
    }
  }

  const bool uniformValid = (state.dirty & DirtyUniform) == 0 && cache.uniformRange.size != 0;
  if (!uniformValid) {
    cache.uniformRange = build_uniform(cache.shaderInfo);
    state.dirty &= ~DirtyUniform;
  }
  if (cache.config.shaderConfig.fogRangeEnabled) {
    const auto key = fog_range_lut_key();
    if (!cache.hasFogRange || cache.fogRangeKey != key) {
      cache.fogRange = push_fog_range_lut(key);
      cache.fogRangeKey = key;
      cache.hasFogRange = true;
    }
  }
  immediates.fogRangeBase = cache.fogRange.offset / sizeof(u32);

  state.dirty &= ~DirtyImmediates;
}

// Adjacent draw batching (AuroraConfig::batchDraws). Consecutive GX draws usually differ only in their
// uniform record (matrices, colors, texture LOD); with the uniform table the record index rides in the
// decoded vertices, so draws that share a pipeline, texture bind group, destination alpha, fog range table
// and uniform window merge into one draw call regardless of the state between them. Streamed draws append
// vertices and rebased indices (points add instances); resident display lists carry no record in their
// vertices and merge only with the same record.
struct BatchStats {
  uint64_t attempts = 0;
  uint64_t merged = 0;
  uint64_t noPrevious = 0;
  uint64_t kind = 0; // previous draw is resident / streamed while this one is not
  uint64_t pipeline = 0;
  uint64_t texture = 0;
  uint64_t dstAlpha = 0;
  uint64_t fog = 0;
  uint64_t window = 0;
  uint64_t record = 0;
  uint64_t limit = 0;
  uint64_t gap = 0;
};
BatchStats sBatchStats;

static void batch_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, std::span<const u8> vertexData,
                       std::span<const u8> indexData) noexcept {
  ZoneScoped;
  auto& state = g_gxState;
  auto& cache = sDrawCache;
  DrawImmediateData immediates{.currentPnMtx = state.currentPnMtx};
  prepare_draw_state(prim, fmt, immediates);
  const u8 lineMode = line_mode_for_prim(prim);
  const bool points = lineMode == 3;
  const u32 record = uniform_record_index(cache.uniformRange.offset);
  const u32 expandedCount = lineMode == 1 || lineMode == 2 ? line_instance_count(lineMode, vtxCount) * 4 : vtxCount;

  auto& stats = sBatchStats;
  ++stats.attempts;
  auto* previous = gfx::get_last_draw_command<DrawData>();
  bool merge = false;
  // Same pipeline implies the same primitive class, so a point draw only ever meets a point draw here.
  if (previous == nullptr) {
    ++stats.noPrevious;
  } else if (previous->residentArena != 0) {
    ++stats.kind;
  } else if (previous->pipeline != cache.pipelineRef) {
    ++stats.pipeline;
  } else if (previous->bindGroups.textureBindGroup != cache.bindGroups.textureBindGroup) {
    ++stats.texture;
  } else if (previous->dstAlpha != state.dstAlpha) {
    ++stats.dstAlpha;
  } else if (previous->immediateData.fogRangeBase != immediates.fogRangeBase) {
    ++stats.fog;
  } else if (uniform_window_index(previous->uniformRange.offset) != uniform_window_index(cache.uniformRange.offset)) {
    ++stats.window;
  } else if (previous->vtxCount + expandedCount > 0xFFFF) {
    ++stats.limit;
  } else if (!gfx::vertices_follow(previous->vertRange) || (!points && !gfx::indices_follow(previous->idxRange))) {
    ++stats.gap;
  } else {
    merge = true;
    ++stats.merged;
  }

  const gfx::Range vertRange = push_decoded_verts(prim, fmt, vertexData, vtxCount, merge ? 0 : 4, record << 8);
  const u16 base = merge ? static_cast<u16>(previous->vtxCount) : 0;
  static ByteBuffer indices;
  indices.clear();
  u32 numIndices = 0;
  if (points) {
    // One instance per point of the shared quad; merges add instances.
    if (!merge) {
      numIndices = prepare_idx_buffer(indices, GX_POINTS, 0, vtxCount);
    }
  } else if (!indexData.empty()) {
    // GX_AURORA_DRAW_INDEXED: host-endian 16-bit indices relative to this draw's vertices.
    numIndices = static_cast<u32>(indexData.size() / sizeof(u16));
    indices.reserve_extra(indexData.size());
    for (size_t i = 0; i < indexData.size(); i += sizeof(u16)) {
      u16 index;
      std::memcpy(&index, indexData.data() + i, sizeof(index));
      AURORA_ASSERT(index < vtxCount, "GX index {} outside the draw's {} vertices", index, vtxCount);
      indices.append(static_cast<u16>(base + index));
    }
  } else {
    numIndices = prepare_idx_buffer(indices, prim, base, vtxCount);
  }
  gfx::Range idxRange;
  if (numIndices != 0) {
    idxRange = gfx::push_indices(indices.data(), indices.size(), merge ? 0 : 4);
  }
  cache.lastDrawFmt = fmt;
  if (merge) {
    previous->vertRange.size += vertRange.size;
    previous->vtxCount += vtxCount;
    if (points) {
      previous->instanceCount += vtxCount;
    } else {
      previous->idxRange.size += idxRange.size;
      previous->indexCount += numIndices;
    }
    gfx::detail::increment_merged_draw_count();
    return;
  }
  immediates.vtxStart = vertRange.offset;
  gfx::push_draw_command(DrawData{
      .pipeline = cache.pipelineRef,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = cache.uniformRange,
      .immediateData = immediates,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = points ? vtxCount : 1u,
      .bindGroups = cache.bindGroups,
      .dstAlpha = state.dstAlpha,
      .residentArena = 0,
  });
}

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u32 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices, u32 residentArena = 0) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;

  DrawImmediateData immediates{.vtxStart = vertRange.offset, .currentPnMtx = state.currentPnMtx};
  if (!g_config.cpuVertexDecode) { // the CPU vertex decoder resolves indexed arrays itself
    for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
      if (state.vtxDesc[i] != GX_INDEX8 && state.vtxDesc[i] != GX_INDEX16) {
        continue;
      }
      auto& array = state.arrays[i];
      if (array.cachedRange.size == 0) {
        array.cachedRange = gfx::push_storage(static_cast<const uint8_t*>(array.data), array.size);
      }
      immediates.arrayStart[i - GX_VA_POS] = array.cachedRange.offset;
    }
  }

  prepare_draw_state(prim, fmt, immediates);

  // Points draw one instance per point; lines one per segment, unless the CPU vertex decoder
  // already expanded them into quads.
  uint32_t instanceCount = 1;
  if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  } else if (!g_config.cpuVertexDecode) {
    if (prim == GX_LINES) {
      instanceCount = vtxCount / 2;
    } else if (prim == GX_LINESTRIP) {
      instanceCount = vtxCount - 1;
    }
  }
  cache.lastDrawFmt = fmt;
  gfx::push_draw_command(DrawData{
      .pipeline = cache.pipelineRef,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = cache.uniformRange,
      .immediateData = immediates,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .bindGroups = cache.bindGroups,
      .dstAlpha = state.dstAlpha,
      .residentArena = residentArena,
  });
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange) noexcept {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    static ByteBuffer idxBuf;
    numIndices = prepare_idx_buffer(idxBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), 4);
    idxBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices);
}

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, ByteReader& reader) noexcept {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calc_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (totalVtxBytes > reader.remaining())
    UNLIKELY { handle_draw_overrun(totalVtxBytes, reader); }

  if (batch_draws_enabled()) {
    batch_draw(prim, fmt, vtxCount, reader.take(totalVtxBytes), {});
    return;
  }

  // Consecutive draws with unchanged state merge: triangle primitives by appending indices, and
  // CPU-decoded points (one record per point, instanced over a shared quad) by adding instances.
  const bool cleanState = g_gxState.dirty == 0 && fmt == sDrawCache.lastDrawFmt;
  const bool mergePoints = g_config.cpuVertexDecode && prim == GX_POINTS && sDrawCache.lineMode == 3;
  const bool mergeTriangles = sDrawCache.lineMode == 0 && prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS;
  auto* lastDraw = cleanState && (mergeTriangles || mergePoints) ? gfx::get_last_draw_command<DrawData>() : nullptr;
  const bool canMerge = lastDraw != nullptr && lastDraw->residentArena == 0 &&
                        lastDraw->instanceCount == (mergePoints ? lastDraw->vtxCount : 1u);

  // Push vertex data to the buffer, raw or decoded on the CPU. Merged draws must remain contiguous
  // with the previous range.
  const auto vertexData = reader.take(totalVtxBytes);
  const gfx::Range vertRange = g_config.cpuVertexDecode
                                   ? push_decoded_verts(prim, fmt, vertexData, vtxCount, canMerge ? 0 : 4)
                                   : gfx::push_verts(vertexData.data(), vertexData.size(), canMerge ? 0 : 4);

  // Try to merge with previous draw call
  if (canMerge) {
    if (mergePoints) {
      CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
            "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
            vertRange.offset);
      lastDraw->vertRange.size += vertRange.size;
      lastDraw->vtxCount += vtxCount;
      lastDraw->instanceCount += vtxCount;
      gfx::detail::increment_merged_draw_count();
      return;
    }
    u32 numIndices = 0;
    gfx::Range idxRange;
    static ByteBuffer idxBuf;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(idxBuf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(idxBuf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), hadIndexRange ? 0 : 4);
      idxBuf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    gfx::detail::increment_merged_draw_count();
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
}

static void handle_draw(u8 cmd, ByteReader& reader) noexcept {
  const auto fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  const auto prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  draw_prim(prim, fmt, reader.read<u16>(), reader);
}

// Draws a resident display list (resident_geometry.hpp): its absolute 32-bit indices are appended
// to the frame's index stream and the whole arena is bound. Consecutive calls under unchanged state
// merge into one draw, as streamed geometry does.
static void push_resident_draw(const resident::Entry& entry, GXVtxFmt fmt) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;
  const u32 arena = entry.arena + 1;
  const u32 stride = resident::arena_stride(entry.arena);
  const gfx::Range vertRange{entry.firstVertex * stride, entry.vertexCount * stride};
  const auto* indexData = reinterpret_cast<const u8*>(entry.indices.data());
  const size_t indexBytes = entry.indices.size() * sizeof(u32);
  const auto indexCount = static_cast<u32>(entry.indices.size());

  if (batch_draws_enabled()) {
    // Resident vertices carry no record index: merge with the previous resident draw of the same arena
    // whenever pipeline, textures, record, destination alpha and fog table match, whatever changed between.
    DrawImmediateData immediates{.vtxStart = 0, .currentPnMtx = state.currentPnMtx};
    prepare_draw_state(GX_TRIANGLES, fmt, immediates);
    auto& stats = sBatchStats;
    ++stats.attempts;
    auto* lastDraw = gfx::get_last_draw_command<DrawData>();
    bool merge = false;
    if (lastDraw == nullptr) {
      ++stats.noPrevious;
    } else if (lastDraw->residentArena != arena || lastDraw->instanceCount != 1) {
      ++stats.kind;
    } else if (lastDraw->pipeline != cache.pipelineRef) {
      ++stats.pipeline;
    } else if (lastDraw->bindGroups.textureBindGroup != cache.bindGroups.textureBindGroup) {
      ++stats.texture;
    } else if (lastDraw->uniformRange.offset != cache.uniformRange.offset) {
      ++stats.record;
    } else if (lastDraw->dstAlpha != state.dstAlpha) {
      ++stats.dstAlpha;
    } else if (lastDraw->immediateData.fogRangeBase != immediates.fogRangeBase) {
      ++stats.fog;
    } else if (!gfx::indices_follow(lastDraw->idxRange)) {
      ++stats.gap;
    } else {
      merge = true;
      ++stats.merged;
    }
    const gfx::Range idxRange = gfx::push_indices(indexData, indexBytes, merge ? 0 : 4);
    cache.lastDrawFmt = fmt;
    if (merge) {
      lastDraw->idxRange.size += idxRange.size;
      lastDraw->indexCount += indexCount;
      lastDraw->vtxCount += entry.vertexCount;
      const u32 begin = std::min(lastDraw->vertRange.offset, vertRange.offset);
      const u32 end =
          std::max(lastDraw->vertRange.offset + lastDraw->vertRange.size, vertRange.offset + vertRange.size);
      lastDraw->vertRange = {begin, end - begin};
      gfx::detail::increment_merged_draw_count();
      return;
    }
    gfx::push_draw_command(DrawData{
        .pipeline = cache.pipelineRef,
        .vertRange = vertRange,
        .idxRange = idxRange,
        .uniformRange = cache.uniformRange,
        .immediateData = immediates,
        .vtxCount = entry.vertexCount,
        .indexCount = indexCount,
        .instanceCount = 1,
        .bindGroups = cache.bindGroups,
        .dstAlpha = state.dstAlpha,
        .residentArena = arena,
    });
    return;
  }

  const bool cleanState = state.dirty == 0 && fmt == cache.lastDrawFmt && cache.lineMode == 0;
  auto* lastDraw = cleanState ? gfx::get_last_draw_command<DrawData>() : nullptr;
  if (lastDraw != nullptr && lastDraw->residentArena == arena && lastDraw->instanceCount == 1) {
    const gfx::Range idxRange = gfx::push_indices(indexData, indexBytes, 0);
    CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
          "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
          idxRange.offset);
    lastDraw->idxRange.size += idxRange.size;
    lastDraw->indexCount += indexCount;
    lastDraw->vtxCount += entry.vertexCount;
    const u32 begin = std::min(lastDraw->vertRange.offset, vertRange.offset);
    const u32 end = std::max(lastDraw->vertRange.offset + lastDraw->vertRange.size, vertRange.offset + vertRange.size);
    lastDraw->vertRange = {begin, end - begin};
    gfx::detail::increment_merged_draw_count();
    return;
  }
  const gfx::Range idxRange = gfx::push_indices(indexData, indexBytes, 4);
  push_gx_draw(GX_TRIANGLES, fmt, entry.vertexCount, vertRange, idxRange, indexCount, arena);
}

// GX_AURORA_CALL_DL: decodes the list into resident geometry on first use and draws it by reference
// afterwards. Lists the cache does not take (state loads, lines, points, mixed vertex formats,
// over budget) are processed inline exactly as if they had been copied into the FIFO.
static void handle_call_display_list(const u8* list, u32 size) noexcept {
  auto& stats = resident::stats();
  auto& state = g_gxState;
  ++stats.calls;
  u32 pos = 0;
  while (pos < size && list[pos] == CP_CMD_NOP) {
    ++pos;
  }
  if (pos >= size) {
    return;
  }
  // The first draw selects the vertex format; decode_display_list validates the rest.
  u8 op = list[pos];
  if (op == GX_AURORA && pos + 4 <= size &&
      (static_cast<u16>(list[pos + 1]) << 8 | list[pos + 2]) == GX_AURORA_DRAW_INDEXED) {
    op = list[pos + 3];
  }
  const u8 prim = op & CP_OPCODE_MASK;
  const bool triangles = prim == GX_TRIANGLES || prim == GX_TRIANGLESTRIP || prim == GX_TRIANGLEFAN || prim == GX_QUADS;
  if (!g_config.cpuVertexDecode || !g_config.residentDisplayLists || !triangles) {
    ++stats.fallbacks;
    process(list, size);
    return;
  }
  const auto fmt = static_cast<GXVtxFmt>(op & CP_VAT_MASK);
  prepare_pipeline(GX_TRIANGLES, fmt);
  const auto& config = sDrawCache.config.shaderConfig;
  const u64 key = resident::entry_key(list, size, config, state.arrays, state.currentPnMtx);
  const u64 listHash = resident::list_hash(list, size);
  resident::Entry* entry = resident::find(key);
  if (entry != nullptr && entry->listHash == listHash) {
    ++stats.hits;
  } else {
    resident::Entry decoded{.list = list, .listBytes = size, .listHash = listHash};
    if (!resident::decode_display_list(list, size, fmt, vertex_loader(config), state.arrays, state.currentPnMtx,
                                       decoded)) {
      resident::erase(key);
      ++stats.fallbacks;
      process(list, size);
      return;
    }
    ++stats.misses;
    entry = &resident::insert(key, std::move(decoded));
  }
  push_resident_draw(*entry, fmt);
}

void handle_aurora(ByteReader& reader) noexcept {
  ZoneScoped;
  const u16 subCmd = reader.read<u16>();

  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    const f32 left = reader.read<f32>();
    const f32 top = reader.read<f32>();
    const f32 width = reader.read<f32>();
    const f32 height = reader.read<f32>();
    const f32 nearZ = reader.read<f32>();
    const f32 farZ = reader.read<f32>();
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    auto& proj = g_gxState.proj;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        proj[r][c] = reader.read<f32>();
      }
    }
    // Invalidate projection XF regs
    for (u32 reg = 0x20; reg <= 0x26; ++reg) {
      g_gxState.xfRegValid.reset(reg);
    }
    g_gxState.dirty |= DirtyUniform;
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    const u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;
    const u64 arrayAddr = reader.read<u64>();
    const u32 arraySize = reader.read<u32>();
    const bool le = reader.read<u8>() == 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      if (array.le != le) {
        // Endianness is baked into the shader
        g_gxState.dirty |= DirtyPipeline;
      }
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      array.cachedRange = {};
      g_gxState.dirty |= DirtyImmediates;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 newWidth = reader.read<u32>();
    const u32 newHeight = reader.read<u32>();
    const auto newFormat = static_cast<GXTexFmt>(reader.read<u32>());
    const auto newTlut = static_cast<GXTlut>(reader.read<u32>());
    u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (reader.read<u8>() != 0) {
      newFlags |= 1u;
    } else {
      newFlags &= ~1u;
    }
    const u32 newTexObjId = reader.read<u32>();
    const u32 newTexDataVersion = reader.read<u32>();
    if (slot.data != newData || slot.mWidth != newWidth || slot.mHeight != newHeight ||
        slot.mFormat != static_cast<u32>(newFormat) || slot.tlut != newTlut || slot.flags != newFlags ||
        slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion) {
      slot.data = newData;
      slot.mWidth = newWidth;
      slot.mHeight = newHeight;
      slot.mFormat = newFormat;
      slot.tlut = newTlut;
      slot.flags = newFlags;
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    const auto idx = reader.read<u8>();
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const auto newFormat = static_cast<GXTlutFmt>(reader.read<u32>());
    const u16 newNumEntries = reader.read<u16>();
    const u32 newTlutObjId = reader.read<u32>();
    const u32 newTlutDataVersion = reader.read<u32>();
    const u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (slot.data != newData || slot.format != newFormat || slot.numEntries != newNumEntries ||
        slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion || slot.flags != newFlags) {
      if (slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion) {
        texture::invalidate_bindings();
      }
      slot.data = newData;
      slot.format = newFormat;
      slot.numEntries = newNumEntries;
      slot.tlutObjId = newTlutObjId;
      slot.tlutDataVersion = newTlutDataVersion;
      slot.flags = newFlags;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    const f32 frontOffset = reader.read<f32>();
    const f32 frontScale = reader.read<f32>();
    const f32 backOffset = reader.read<f32>();
    const f32 backScale = reader.read<f32>();
    const f32 clamp = reader.read<f32>();
    if (g_gxState.frontOffset != frontOffset || g_gxState.frontScale != frontScale ||
        g_gxState.backOffset != backOffset || g_gxState.backScale != backScale || g_gxState.clamp != clamp) {
      g_gxState.frontOffset = frontOffset;
      g_gxState.frontScale = frontScale;
      g_gxState.backOffset = backOffset;
      g_gxState.backScale = backScale;
      g_gxState.clamp = clamp;
      g_gxState.dirty |= DirtyPipeline;
    }
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    g_gxState.texCopyDstWidth = reader.read<u32>();
    g_gxState.texCopyDstHeight = reader.read<u32>();
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(reader.read<u32>());
    reader.skip(1); // mipmap is not implemented, but remains part of the command payload
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    g_gxState.texCopyDest = reinterpret_cast<const void*>(reader.read<u64>());
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    gfx::depth_peek::request_snapshot();
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    const u32 width = reader.read<u32>();
    const u32 height = reader.read<u32>();
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    evict_texture_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    evict_tlut_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    evict_copy_texture(reinterpret_cast<const void*>(reader.read<u64>()));
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    const u8 cmd = reader.read<u8>();
    const u32 byteLen = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calc_vtx_size(fmt);
      }
      AURORA_ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
                    "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      AURORA_ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), reader);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    const u8 cmd = reader.read<u8>();
    u16 vtxCount = reader.read<u16>();
    const u32 indexCount = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    AURORA_ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
                  static_cast<u32>(prim));
    const size_t idxBytes = static_cast<size_t>(indexCount) * sizeof(u16);
    // Index data is always host-endian; push it to the GPU buffer as-is
    const auto indexData = reader.take(idxBytes);
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calc_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    if (batch_draws_enabled()) {
      const auto vertexData = reader.take(totalVtxBytes);
      if (indexCount != 0) {
        batch_draw(prim, fmt, vtxCount, vertexData, indexData);
      }
      return;
    }
    const gfx::Range idxRange = gfx::push_indices(indexData.data(), indexData.size(), 4);
    const auto vertexData = reader.take(totalVtxBytes);
    const gfx::Range vertRange = g_config.cpuVertexDecode ? push_decoded_verts(prim, fmt, vertexData, vtxCount, 4)
                                                          : gfx::push_verts(vertexData.data(), vertexData.size(), 4);
    if (indexCount != 0) {
      push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount);
    }
  } else if (subCmd == GX_AURORA_FRAME_BEGIN) {
    gfx::begin_reserved_frame(reader.read<u32>());
  } else if (subCmd == GX_AURORA_FRAME_END) {
    // Asynchronous frames: the producer does not join the processor, so the per-frame work
    // aurora::end_frame() performs after drain() runs here, on the thread that owns the state.
    clear_draw_cache();
    texture::end_frame();
    gfx::finish();
    gfx::end_deferred_frame();
    dispatch_after_frame();
  } else if (subCmd == GX_AURORA_CALL_DL) {
    ZoneScopedN("CALL_DL");
    const auto* list = reinterpret_cast<const u8*>(reader.read<u64>());
    const u32 size = reader.read<u32>();
    handle_call_display_list(list, size);
  } else if (subCmd == GX_AURORA_INVALIDATE_RESIDENT) {
    const auto* base = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 size = reader.read<u32>();
    resident::invalidate(base, size);
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = reader.read_string();
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = reader.read_string();
    gfx::insert_debug_marker(std::move(label));
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
}

void clear_draw_cache() noexcept {
  if (g_config.renderStats && batch_draws_enabled()) {
    static uint32_t frames = 0;
    if (++frames % 300 == 0) {
      const auto& s = sBatchStats;
      std::fprintf(stderr,
                   "[gx-batch] frames=300 attempts=%llu merged=%llu no-previous=%llu kind=%llu pipeline=%llu "
                   "texture=%llu dst-alpha=%llu fog=%llu window=%llu record=%llu limit=%llu gap=%llu\n",
                   static_cast<unsigned long long>(s.attempts), static_cast<unsigned long long>(s.merged),
                   static_cast<unsigned long long>(s.noPrevious), static_cast<unsigned long long>(s.kind),
                   static_cast<unsigned long long>(s.pipeline), static_cast<unsigned long long>(s.texture),
                   static_cast<unsigned long long>(s.dstAlpha), static_cast<unsigned long long>(s.fog),
                   static_cast<unsigned long long>(s.window), static_cast<unsigned long long>(s.record),
                   static_cast<unsigned long long>(s.limit), static_cast<unsigned long long>(s.gap));
      sBatchStats = {};
    }
  }
  sDrawCache.bindGeneration = 0;
  sDrawCache.uniformRange = {};
  sDrawCache.fogRange = {};
  sDrawCache.hasFogRange = false;
}

void reset_pipeline_memo() noexcept {
  sPipelineMemo.clear();
  sDrawCache.hasPipeline = false;
}

namespace testing {
uint32_t pipeline_memo_misses() noexcept { return sPipelineMemoMisses; }
const PipelineConfig& cached_pipeline_config() noexcept { return sDrawCache.config; }
} // namespace testing

} // namespace aurora::gx::fifo
