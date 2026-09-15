#include "vertex_loader.hpp"

#include <absl/container/flat_hash_map.h>
#include <xxhash.h>

#include <bit>
#include <cstring>
#include <memory>
#include <type_traits>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace aurora::gx {
namespace {
constexpr u32 MatrixLocation = 0;
constexpr u32 PosLocation = 1;
constexpr u32 NrmLocation = 2;
constexpr u32 ColorLocation = 3;
constexpr u32 TexLocation = 5;
constexpr u32 BinrmLocation = 13;
constexpr u32 TangentLocation = 14;
constexpr u32 LineEndLocation = 15;
constexpr u16 NoOffset = UINT16_MAX;
constexpr u8 Zero[16]{};

u32 attr_location(GXAttr attr) noexcept {
  if (attr == GX_VA_POS) {
    return PosLocation;
  }
  if (attr == GX_VA_NRM) {
    return NrmLocation;
  }
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    return ColorLocation + (attr - GX_VA_CLR0);
  }
  return TexLocation + (attr - GX_VA_TEX0);
}

u32 numeric_comp_size(u32 compType) noexcept {
  return compType == GX_F32 ? 4 : (compType == GX_U16 || compType == GX_S16) ? 2 : 1;
}

u32 color_size(u32 compType) noexcept {
  return (compType == GX_RGB565 || compType == GX_RGBA4) ? 2 : (compType == GX_RGB8 || compType == GX_RGBA6) ? 3 : 4;
}

// Address of the indexed array record for one vertex, or zeros when it lies outside the array.
template <int IndexBytes>
const u8* indexed_record(const u8* src, size_t stride, size_t within, size_t bytes, const u8* array,
                         size_t arraySize) noexcept {
  const size_t index = IndexBytes == 1 ? size_t(src[0]) : (size_t(src[0]) << 8) | src[1];
  const size_t offset = index * stride + within;
  return offset + bytes <= arraySize ? array + offset : Zero;
}

// Fixed-point and float attributes: Comps source components converted to floats, remaining
// destination components cleared. Conversion matches the vertex shader's f32(v) / f32(1u << frac):
// multiplying by the reciprocal power of two is exact.
template <int IndexBytes, typename T, bool Little, int Comps>
void step_numeric(const VertexLoaderStep& s, const u8* __restrict raw, size_t count, size_t vtxStride,
                  u8* __restrict out, size_t outStride, const u8* array, size_t arraySize) noexcept {
  constexpr size_t Bytes = Comps * sizeof(T);
  const size_t within = s.within;
  const size_t stride = s.stride;
  const u32 components = s.components;
  const float scale = s.scale;
  const u8* src = raw + s.srcOffset;
  u8* dst = out + s.dstOffset;
  for (size_t i = 0; i < count; ++i, src += vtxStride, dst += outStride) {
    const u8* p;
    if constexpr (IndexBytes == 0) {
      p = src + within;
    } else {
      p = indexed_record<IndexBytes>(src, stride, within, Bytes, array, arraySize);
    }
    float* d = reinterpret_cast<float*>(dst);
#if defined(__aarch64__)
    // Two or three 16-bit components (the common GX position and texture coordinate formats)
    // convert together; the loads never exceed the six-byte XYZ record.
    if constexpr (sizeof(T) == 2 && (Comps == 2 || Comps == 3)) {
      uint32_t xy;
      std::memcpy(&xy, p, 4);
      uint16x4_t h = vreinterpret_u16_u32(vdup_n_u32(xy));
      if constexpr (Comps == 3) {
        uint16_t z;
        std::memcpy(&z, p + 4, 2);
        h = vset_lane_u16(z, h, 2);
      }
      if constexpr (!Little) {
        h = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(h)));
      }
      float32x4_t v;
      if constexpr (std::is_signed_v<T>) {
        v = vcvtq_f32_s32(vmovl_s16(vreinterpret_s16_u16(h)));
      } else {
        v = vcvtq_f32_u32(vmovl_u16(h));
      }
      v = vmulq_n_f32(v, scale);
      vst1_f32(d, vget_low_f32(v));
      if constexpr (Comps == 3) {
        vst1q_lane_f32(d + 2, v, 2);
      } else if (components > 2) {
        d[2] = 0.f;
      }
      continue;
    }
#endif
    for (int j = 0; j < Comps; ++j) {
      if constexpr (sizeof(T) == 4) {
        uint32_t bits;
        std::memcpy(&bits, p + j * 4, 4);
        if constexpr (!Little) {
          bits = bswap(bits);
        }
        d[j] = std::bit_cast<float>(bits);
      } else if constexpr (sizeof(T) == 2) {
        uint16_t bits;
        std::memcpy(&bits, p + j * 2, 2);
        if constexpr (!Little) {
          bits = bswap(bits);
        }
        d[j] = float(std::bit_cast<T>(bits)) * scale;
      } else {
        d[j] = float(std::bit_cast<T>(p[j])) * scale;
      }
    }
    for (u32 j = Comps; j < components; ++j) {
      d[j] = 0.f;
    }
  }
}

template <int IndexBytes, int Fmt, bool Little>
void step_color(const VertexLoaderStep& s, const u8* __restrict raw, size_t count, size_t vtxStride, u8* __restrict out,
                size_t outStride, const u8* array, size_t arraySize) noexcept {
  constexpr size_t Bytes = (Fmt == GX_RGB565 || Fmt == GX_RGBA4) ? 2 : (Fmt == GX_RGB8 || Fmt == GX_RGBA6) ? 3 : 4;
  const size_t stride = s.stride;
  const u8* src = raw + s.srcOffset;
  u8* dst = out + s.dstOffset;
  for (size_t i = 0; i < count; ++i, src += vtxStride, dst += outStride) {
    const u8* p;
    if constexpr (IndexBytes == 0) {
      p = src;
    } else {
      p = indexed_record<IndexBytes>(src, stride, 0, Bytes, array, arraySize);
    }
    float* d = reinterpret_cast<float*>(dst);
    if constexpr (Fmt == GX_RGB565 || Fmt == GX_RGBA4 || Fmt == GX_RGBA6) {
      uint32_t v = 0;
      for (size_t k = 0; k < Bytes; ++k) {
        v |= uint32_t(p[k]) << ((Little ? k : Bytes - k - 1) * 8);
      }
      if constexpr (Fmt == GX_RGB565) {
        d[0] = float((v >> 11) & 31) / 31.f;
        d[1] = float((v >> 5) & 63) / 63.f;
        d[2] = float(v & 31) / 31.f;
        d[3] = 1.f;
      } else {
        constexpr u32 Bits = Fmt == GX_RGBA4 ? 4 : 6;
        constexpr u32 Mask = (1u << Bits) - 1;
        for (u32 j = 0; j < 4; ++j) {
          d[j] = float((v >> ((3 - j) * Bits)) & Mask) / float(Mask);
        }
      }
    } else {
      d[0] = float(p[0]) / 255.f;
      d[1] = float(p[1]) / 255.f;
      d[2] = float(p[2]) / 255.f;
      d[3] = Fmt == GX_RGBA8 ? float(p[3]) / 255.f : 1.f;
    }
  }
}

// One byte of the matrix index word; the PN matrix index is stored divided by three.
template <int IndexBytes>
void step_matrix_index(const VertexLoaderStep& s, const u8* __restrict raw, size_t count, size_t vtxStride,
                       u8* __restrict out, size_t outStride, const u8* array, size_t arraySize) noexcept {
  const size_t stride = s.stride;
  const bool pn = s.attr == GX_VA_PNMTXIDX;
  const u8* src = raw + s.srcOffset;
  u8* dst = out + s.dstOffset;
  for (size_t i = 0; i < count; ++i, src += vtxStride, dst += outStride) {
    const u8* p;
    if constexpr (IndexBytes == 0) {
      p = src;
    } else {
      p = indexed_record<IndexBytes>(src, stride, 0, 1, array, arraySize);
    }
    *dst = pn ? u8(*p / 3) : *p;
  }
}

// Locations present in the layout without a source attribute read as zero.
void step_zero(const VertexLoaderStep& s, const u8*, size_t count, size_t, u8* __restrict out, size_t outStride,
               const u8*, size_t) noexcept {
  u8* dst = out + s.dstOffset;
  const size_t bytes = s.within;
  for (size_t i = 0; i < count; ++i, dst += outStride) {
    std::memset(dst, 0, bytes);
  }
}

template <int IndexBytes, typename T, bool Little>
VertexLoaderStep::Fn numeric_fn(u32 comps) noexcept {
  switch (comps) {
  case 1:
    return &step_numeric<IndexBytes, T, Little, 1>;
  case 2:
    return &step_numeric<IndexBytes, T, Little, 2>;
  default:
    return &step_numeric<IndexBytes, T, Little, 3>;
  }
}
template <int IndexBytes, typename T>
VertexLoaderStep::Fn numeric_fn(bool le, u32 comps) noexcept {
  return le ? numeric_fn<IndexBytes, T, true>(comps) : numeric_fn<IndexBytes, T, false>(comps);
}
template <int IndexBytes>
VertexLoaderStep::Fn numeric_fn(u32 compType, bool le, u32 comps) noexcept {
  switch (compType) {
  case GX_F32:
    return numeric_fn<IndexBytes, float>(le, comps);
  case GX_U16:
    return numeric_fn<IndexBytes, uint16_t>(le, comps);
  case GX_S16:
    return numeric_fn<IndexBytes, int16_t>(le, comps);
  case GX_S8:
    return numeric_fn<IndexBytes, int8_t>(le, comps);
  default:
    return numeric_fn<IndexBytes, uint8_t>(le, comps);
  }
}
VertexLoaderStep::Fn numeric_fn(u32 indexBytes, u32 compType, bool le, u32 comps) noexcept {
  switch (indexBytes) {
  case 0:
    return numeric_fn<0>(compType, le, comps);
  case 1:
    return numeric_fn<1>(compType, le, comps);
  default:
    return numeric_fn<2>(compType, le, comps);
  }
}

template <int IndexBytes, bool Little>
VertexLoaderStep::Fn color_fn(u32 fmt) noexcept {
  switch (fmt) {
  case GX_RGB565:
    return &step_color<IndexBytes, GX_RGB565, Little>;
  case GX_RGB8:
    return &step_color<IndexBytes, GX_RGB8, Little>;
  case GX_RGBX8:
    return &step_color<IndexBytes, GX_RGBX8, Little>;
  case GX_RGBA4:
    return &step_color<IndexBytes, GX_RGBA4, Little>;
  case GX_RGBA6:
    return &step_color<IndexBytes, GX_RGBA6, Little>;
  default:
    return &step_color<IndexBytes, GX_RGBA8, Little>;
  }
}
template <int IndexBytes>
VertexLoaderStep::Fn color_fn(u32 fmt, bool le) noexcept {
  return le ? color_fn<IndexBytes, true>(fmt) : color_fn<IndexBytes, false>(fmt);
}
VertexLoaderStep::Fn color_fn(u32 indexBytes, u32 fmt, bool le) noexcept {
  switch (indexBytes) {
  case 0:
    return color_fn<0>(fmt, le);
  case 1:
    return color_fn<1>(fmt, le);
  default:
    return color_fn<2>(fmt, le);
  }
}

VertexLoaderStep::Fn matrix_index_fn(u32 indexBytes) noexcept {
  switch (indexBytes) {
  case 0:
    return &step_matrix_index<0>;
  case 1:
    return &step_matrix_index<1>;
  default:
    return &step_matrix_index<2>;
  }
}

u16 layout_offset(const DecodedVertexLayout& layout, u32 location) noexcept {
  for (u32 i = 0; i < layout.count; ++i) {
    if (layout.attributes[i].shaderLocation == location) {
      return static_cast<u16>(layout.attributes[i].offset);
    }
  }
  return NoOffset;
}

VertexLoader build_vertex_loader(const ShaderConfig& config) noexcept {
  VertexLoader loader;
  loader.layout = decoded_vertex_layout(config);
  loader.vtxStride = config.vtxStride;
  loader.lineMode = config.lineMode;
  const u16 matrixOffset = layout_offset(loader.layout, MatrixLocation);
  loader.hasMatrices = matrixOffset != NoOffset;
  loader.matrixOffset = loader.hasMatrices ? matrixOffset : 0;
  loader.posOffset = layout_offset(loader.layout, PosLocation);
  loader.lineEndOffset = layout_offset(loader.layout, LineEndLocation);
  for (u32 i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto& m = config.attrs[i];
    if (m.attrType == GX_NONE) {
      continue;
    }
    const u32 indexBytes = m.attrType == GX_DIRECT ? 0 : m.attrType == GX_INDEX8 ? 1 : 2;
    const bool le = m.attrType != GX_DIRECT && m.le;
    VertexLoaderStep step{};
    step.attr = static_cast<u8>(attr);
    step.indexBytes = static_cast<u8>(indexBytes);
    step.srcOffset = m.offset;
    step.stride = m.stride;
    if (attr <= GX_VA_TEX7MTXIDX) {
      if (!loader.hasMatrices) {
        continue;
      }
      step.fn = matrix_index_fn(indexBytes);
      step.dstOffset = static_cast<u16>(matrixOffset + (i / 4) * 4 + (i % 4));
      loader.steps.push_back(step);
      continue;
    }
    const u16 baseOffset = layout_offset(loader.layout, attr_location(attr));
    if (baseOffset == NoOffset) {
      continue;
    }
    if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
      step.fn = color_fn(indexBytes, m.compType, le);
      step.dstOffset = baseOffset;
      loader.steps.push_back(step);
      continue;
    }
    const u32 compSize = numeric_comp_size(m.compType);
    const u32 comps = attr == GX_VA_NRM ? 3 : m.cnt;
    const u32 slices = attr == GX_VA_NRM && m.cnt == 9 ? 3 : 1;
    for (u32 slice = 0; slice < slices; ++slice) {
      const u16 sliceOffset = slice == 0 ? baseOffset : layout_offset(loader.layout, BinrmLocation + slice - 1);
      if (sliceOffset == NoOffset) {
        continue;
      }
      step.fn = numeric_fn(indexBytes, m.compType, le, comps);
      step.components = attr >= GX_VA_TEX0 ? 2 : 3;
      // GX_NRM_NBT3 carries one index per slice; GX_NRM_NBT one index for the nine components.
      step.srcOffset = static_cast<u8>(m.offset + (m.nbt3 ? slice * indexBytes : 0));
      step.dstOffset = sliceOffset;
      step.within = slice * 3 * compSize;
      step.scale = 1.f / float(1u << m.frac);
      loader.steps.push_back(step);
    }
  }
  // Attributes the layout declares but no step writes (position without a source, the line end
  // before expansion) are cleared so that every output byte is written.
  for (u32 i = 0; i < loader.layout.count; ++i) {
    const auto& attr = loader.layout.attributes[i];
    if (attr.shaderLocation == MatrixLocation) {
      continue;
    }
    bool written = false;
    for (const auto& step : loader.steps) {
      written |= step.attr > GX_VA_TEX7MTXIDX && step.dstOffset == attr.offset;
    }
    if (written) {
      continue;
    }
    VertexLoaderStep step{};
    step.fn = &step_zero;
    step.attr = GX_VA_POS;
    step.dstOffset = static_cast<u16>(attr.offset);
    step.within = decoded_vertex_attr_size(attr.shaderLocation);
    loader.steps.push_back(step);
  }
  return loader;
}
} // namespace

bool decoded_vertex_has_location(const ShaderConfig& config, u32 location) noexcept {
  switch (location) {
  case MatrixLocation:
    if (config.lineMode != 0) {
      return true;
    }
    for (u32 i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7MTXIDX; ++i) {
      if (config.attrs[i].attrType != GX_NONE) {
        return true;
      }
    }
    return false;
  case PosLocation:
    return true;
  case NrmLocation:
    return config.attrs[GX_VA_NRM].attrType != GX_NONE;
  case BinrmLocation:
  case TangentLocation:
    return config.attrs[GX_VA_NRM].attrType != GX_NONE && config.attrs[GX_VA_NRM].cnt == 9;
  case LineEndLocation:
    return config.lineMode == 1 || config.lineMode == 2;
  default:
    if (location < TexLocation) {
      return config.attrs[GX_VA_CLR0 + location - ColorLocation].attrType != GX_NONE;
    }
    return config.attrs[GX_VA_TEX0 + location - TexLocation].attrType != GX_NONE;
  }
}

u32 decoded_vertex_attr_size(u32 location) noexcept {
  if (location == ColorLocation || location == ColorLocation + 1 || location == LineEndLocation) {
    return 16;
  }
  if (location >= TexLocation && location < BinrmLocation) {
    return 8;
  }
  return 12;
}

DecodedVertexLayout decoded_vertex_layout(const ShaderConfig& config) noexcept {
  DecodedVertexLayout layout{};
  const auto add = [&](u32 location, wgpu::VertexFormat format) {
    if (!decoded_vertex_has_location(config, location)) {
      return;
    }
    layout.attributes[layout.count++] = {
        .format = format,
        .offset = layout.stride,
        .shaderLocation = location,
    };
    layout.stride += decoded_vertex_attr_size(location);
  };
  add(MatrixLocation, wgpu::VertexFormat::Uint32x3);
  add(PosLocation, wgpu::VertexFormat::Float32x3);
  add(NrmLocation, wgpu::VertexFormat::Float32x3);
  add(ColorLocation, wgpu::VertexFormat::Float32x4);
  add(ColorLocation + 1, wgpu::VertexFormat::Float32x4);
  for (u32 i = 0; i < 8; ++i) {
    add(TexLocation + i, wgpu::VertexFormat::Float32x2);
  }
  add(BinrmLocation, wgpu::VertexFormat::Float32x3);
  add(TangentLocation, wgpu::VertexFormat::Float32x3);
  add(LineEndLocation, wgpu::VertexFormat::Float32x4);
  return layout;
}

const VertexLoader& vertex_loader(const ShaderConfig& config) noexcept {
  struct Key {
    std::array<AttrConfig, MaxVtxAttr> attrs;
    u8 vtxStride;
    u8 lineMode;
  };
  static_assert(std::has_unique_object_representations_v<AttrConfig>);
  const Key key{config.attrs, config.vtxStride, config.lineMode};
  const uint64_t hash = XXH3_64bits(&key, sizeof(key));
  static uint64_t lastHash = 0;
  static const VertexLoader* last = nullptr;
  if (last != nullptr && hash == lastHash) {
    return *last;
  }
  static absl::flat_hash_map<uint64_t, std::unique_ptr<VertexLoader>> loaders;
  auto it = loaders.find(hash);
  if (it == loaders.end()) {
    it = loaders.emplace(hash, std::make_unique<VertexLoader>(build_vertex_loader(config))).first;
  }
  lastHash = hash;
  last = it->second.get();
  return *last;
}

void decode_vertices(const VertexLoader& loader, const u8* raw, size_t count, u8* out,
                     const std::array<AttrArray, MaxVtxAttr>& arrays, u32 currentPnMtx) noexcept {
  const size_t stride = loader.layout.stride;
  if (loader.hasMatrices) {
    u8* dst = out + loader.matrixOffset;
    const u32 words[3] = {currentPnMtx, 0u, 0u};
    for (size_t i = 0; i < count; ++i, dst += stride) {
      std::memcpy(dst, words, sizeof(words));
    }
  }
  for (const auto& step : loader.steps) {
    const u8* array = nullptr;
    size_t arraySize = 0;
    if (step.indexBytes != 0) {
      const auto& source = arrays[step.attr];
      if (source.data != nullptr) {
        array = static_cast<const u8*>(source.data);
        arraySize = source.size;
      }
    }
    step.fn(step, raw, count, loader.vtxStride, out, stride, array, arraySize);
  }
}

u32 line_instance_count(u8 lineMode, u32 vtxCount) noexcept {
  switch (lineMode) {
  case 1:
    return vtxCount / 2;
  case 2:
    return vtxCount == 0 ? 0 : vtxCount - 1;
  default:
    return vtxCount;
  }
}

void expand_line_vertices(const VertexLoader& loader, const u8* decoded, size_t instances, u8* out) noexcept {
  const size_t stride = loader.layout.stride;
  const u8 lineMode = loader.lineMode;
  const size_t mo = loader.matrixOffset;
  const size_t po = loader.posOffset;
  const size_t lo = loader.lineEndOffset;
  alignas(16) u8 recA[MaxDecodedVertexStride];
  alignas(16) u8 recB[MaxDecodedVertexStride];
  for (size_t i = 0; i < instances; ++i) {
    const u8* a = decoded + (lineMode == 1 ? i * 2 : i) * stride;
    const u8* b = decoded + (lineMode == 3 ? i : lineMode == 1 ? i * 2 + 1 : i + 1) * stride;
    std::memcpy(recA, a, stride);
    u32 aM0;
    u32 bM0;
    std::memcpy(&aM0, a + mo, 4);
    std::memcpy(&bM0, b + mo, 4);
    if (lo != NoOffset) {
      float lineEnd[4];
      std::memcpy(lineEnd, b + po, 12);
      lineEnd[3] = float(bM0 & 255u);
      std::memcpy(recA + lo, lineEnd, 16);
    }
    if (lineMode == 3) {
      std::memcpy(recB, recA, stride);
    } else {
      std::memcpy(recB, b, stride);
      std::memcpy(recB + po, a + po, 12);
      const u32 m0 = (bM0 & ~255u) | (aM0 & 255u);
      std::memcpy(recB + mo, &m0, 4);
      if (lo != NoOffset) {
        std::memcpy(recB + lo, recA + lo, 16);
      }
    }
    u32 m2A;
    u32 m2B;
    std::memcpy(&m2A, recA + mo + 8, 4);
    std::memcpy(&m2B, recB + mo + 8, 4);
    for (u32 corner = 0; corner < 4; ++corner, out += stride) {
      u8* rec = corner >= 2 ? recB : recA;
      const u32 m2 = (corner >= 2 ? m2B : m2A) | (corner << 24);
      std::memcpy(rec + mo + 8, &m2, 4);
      std::memcpy(out, rec, stride);
    }
  }
}

} // namespace aurora::gx
