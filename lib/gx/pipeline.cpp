#include "pipeline.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/resource_cache.hpp"

#include "gx_fmt.hpp"
#include "resident_geometry.hpp"
#include "shader_info.hpp"
#include "vertex_loader.hpp"
#include "../webgpu/gpu.hpp"

#include <tracy/Tracy.hpp>

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

namespace aurora::gx {
namespace {
std::mutex sPipelineConfigMutex;
absl::flat_hash_map<gfx::PipelineRef, PipelineConfig> sPipelineConfigs;
} // namespace

bool find_pipeline_config(gfx::PipelineRef ref, PipelineConfig& config) {
  std::lock_guard lock{sPipelineConfigMutex};
  const auto it = sPipelineConfigs.find(ref);
  if (it == sPipelineConfigs.end()) {
    return false;
  }
  config = it->second;
  return true;
}

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const auto shader = build_shader(config.shaderConfig);
  const auto hash = xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX));
  const auto label = fmt::format("GX Pipeline {:x} shader {:x}", hash, xxh3_hash(config.shaderConfig));
  {
    std::lock_guard lock{sPipelineConfigMutex};
    sPipelineConfigs[hash] = config;
  }
  if (config.shaderConfig.cpuVertexDecode) {
    const auto layout = decoded_vertex_layout(config.shaderConfig);
    // Points keep one record per point and draw it as one instance of a shared quad
    const bool perInstance = config.shaderConfig.lineMode == 3;
    const wgpu::VertexBufferLayout vertexBuffer{
        .stepMode = perInstance ? wgpu::VertexStepMode::Instance : wgpu::VertexStepMode::Vertex,
        .arrayStride = layout.stride,
        .attributeCount = layout.count,
        .attributes = layout.attributes.data(),
    };
    if (config.shaderConfig.residentRecords) {
      // Second vertex binding: the per-vertex resident record index (u32), fetched with the same index as
      // binding 0 (the arena) so a merged resident draw can mix records within one uniform window.
      static const wgpu::VertexAttribute recordAttr{
          .format = wgpu::VertexFormat::Uint32,
          .offset = 0,
          .shaderLocation = RecordLocation,
      };
      const wgpu::VertexBufferLayout recordBuffer{
          .stepMode = wgpu::VertexStepMode::Vertex,
          .arrayStride = sizeof(uint32_t),
          .attributeCount = 1,
          .attributes = &recordAttr,
      };
      const std::array buffers{vertexBuffer, recordBuffer};
      return build_pipeline(config, {buffers.data(), buffers.size()}, shader, label.c_str());
    }
    return build_pipeline(config, {&vertexBuffer, 1}, shader, label.c_str());
  }
  return build_pipeline(config, {}, shader, label.c_str());
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
    return;
  }

  const auto& resources = gfx::detail::resources();
  wgpu::IndexFormat indexFormat = wgpu::IndexFormat::Uint16;
  // The residentRecords variant reads its record per-vertex from binding 1 rather than from the immediates.
  bool residentRecordsVariant = false;
  if (data.residentArena != 0) {
    PipelineConfig config;
    residentRecordsVariant = find_pipeline_config(data.pipeline, config) && config.shaderConfig.residentRecords;
    uint64_t arenaSize = 0;
    const auto& arena = resident::arena_buffer(data.residentArena - 1, arenaSize);
    if (!arena) {
      return;
    }
    pass.SetVertexBuffer(0, arena, 0, arenaSize);
    if (residentRecordsVariant) {
      uint64_t recordSize = 0;
      const auto& record = resident::record_buffer(data.residentArena - 1, recordSize);
      if (!record) {
        return;
      }
      pass.SetVertexBuffer(1, record, 0, recordSize);
    }
    indexFormat = wgpu::IndexFormat::Uint32;
  } else if (g_config.cpuVertexDecode) {
    pass.SetVertexBuffer(0, resources.vertexBuffer, data.vertRange.offset, data.vertRange.size);
  }
  auto immediates = data.immediateData;
  uint32_t uniformOffset = data.uniformRange.offset;
  const wgpu::BindGroup* uniformGroup = &resources.uniformBindGroup;
  if (uniform_table_enabled()) {
    // Bind the record's 64 KiB window and let the shader index the record: streamed batched draws carry
    // it in their vertices, resident geometry and unbatched draws take it from the immediates. The
    // residentRecords variant carries it per-vertex (binding 1), so its _pad is 0.
    const uint32_t record = uniform_record_index(uniformOffset);
    immediates._pad = (!batch_draws_enabled() || data.residentArena != 0) && !residentRecordsVariant ? record : 0;
    uniformOffset = uniform_window_index(uniformOffset) * UniformWindowSize;
    uniformGroup = &resources.uniformWindowBindGroup;
  }
  pass.SetImmediates(0, &immediates, sizeof(immediates));
  const std::array offsets{uniformOffset};
  pass.SetBindGroup(1, *uniformGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
  }
  pass.SetIndexBuffer(resources.indexBuffer, indexFormat, data.idxRange.offset, data.idxRange.size);
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  if (data.indexCount == 0) {
    pass.Draw(data.vtxCount, data.instanceCount);
  } else {
    pass.DrawIndexed(data.indexCount, data.instanceCount);
  }
}

// GPU side of the resident display-list arenas (resident_geometry.hpp); render thread only.
namespace resident {
namespace {
constexpr Module Log{"aurora::gx::resident"};

struct ArenaBuffer {
  wgpu::Buffer buffer;
  uint64_t size = 0;
};
std::vector<ArenaBuffer> sArenaBuffers;
// Per-arena per-vertex uniform record indices (AuroraConfig::residentRecords), indexed by absolute arena
// vertex index like the arena itself. Fully rewritten each frame, so growth need not preserve old contents.
std::vector<ArenaBuffer> sRecordBuffers;
constexpr uint64_t MinArenaBufferSize = 1024 * 1024;
} // namespace

void encode_uploads(wgpu::CommandEncoder& encoder, const std::vector<gfx::ArenaUpload>& uploads) {
  for (const auto& upload : uploads) {
    if (upload.data.empty()) {
      continue;
    }
    AURORA_ASSERT(upload.offset % 4 == 0 && upload.data.size() % 4 == 0,
                  "resident arena upload of {} bytes at {} is not 4-byte aligned", upload.data.size(), upload.offset);
    if (upload.arena >= sArenaBuffers.size()) {
      sArenaBuffers.resize(upload.arena + 1);
    }
    auto& arena = sArenaBuffers[upload.arena];
    const uint64_t end = upload.offset + upload.data.size();
    if (arena.size < end) {
      // Grow by doubling; the old contents move with a GPU copy so entries keep their offsets.
      uint64_t size = std::max(arena.size * 2, MinArenaBufferSize);
      while (size < end) {
        size *= 2;
      }
      const auto label = fmt::format("Resident vertex arena {}", upload.arena);
      const wgpu::BufferDescriptor descriptor{
          .label = label.c_str(),
          .usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
          .size = size,
      };
      auto grown = webgpu::g_device.CreateBuffer(&descriptor);
      if (arena.buffer) {
        encoder.CopyBufferToBuffer(arena.buffer, 0, grown, 0, arena.size);
      }
      arena.buffer = std::move(grown);
      arena.size = size;
    }
    // Staged copy, ordered in the command stream after every earlier pass that reads the arena. A
    // queue WriteBuffer into a buffer the GPU may still be reading stalls the CPU for a frame on GL
    // drivers (measured on Mali-G52), and the arena is read by every frame.
    const wgpu::BufferDescriptor stagingDescriptor{
        .label = "Resident arena upload",
        .usage = wgpu::BufferUsage::CopySrc,
        .size = upload.data.size(),
        .mappedAtCreation = true,
    };
    auto staging = webgpu::g_device.CreateBuffer(&stagingDescriptor);
    std::memcpy(staging.GetMappedRange(0, upload.data.size()), upload.data.data(), upload.data.size());
    staging.Unmap();
    encoder.CopyBufferToBuffer(staging, 0, arena.buffer, upload.offset, upload.data.size());
  }
}

const wgpu::Buffer& arena_buffer(u32 arena, uint64_t& size) noexcept {
  static const wgpu::Buffer none;
  if (arena >= sArenaBuffers.size()) {
    size = 0;
    return none;
  }
  size = sArenaBuffers[arena].size;
  return sArenaBuffers[arena].buffer;
}

// Per-frame upload of resident record indices; mirrors encode_uploads but the record buffer is fully
// rewritten each frame, so a grown buffer keeps no old contents (no CopyBufferToBuffer on grow).
void encode_record_uploads(wgpu::CommandEncoder& encoder, const std::vector<gfx::ArenaUpload>& uploads) {
  for (const auto& upload : uploads) {
    if (upload.data.empty()) {
      continue;
    }
    AURORA_ASSERT(upload.offset % 4 == 0 && upload.data.size() % 4 == 0,
                  "resident record upload of {} bytes at {} is not 4-byte aligned", upload.data.size(), upload.offset);
    if (upload.arena >= sRecordBuffers.size()) {
      sRecordBuffers.resize(upload.arena + 1);
    }
    auto& record = sRecordBuffers[upload.arena];
    const uint64_t end = upload.offset + upload.data.size();
    if (record.size < end) {
      uint64_t size = std::max(record.size * 2, MinArenaBufferSize);
      while (size < end) {
        size *= 2;
      }
      const auto label = fmt::format("Resident record arena {}", upload.arena);
      const wgpu::BufferDescriptor descriptor{
          .label = label.c_str(),
          .usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst,
          .size = size,
      };
      record.buffer = webgpu::g_device.CreateBuffer(&descriptor);
      record.size = size;
    }
    // Staged copy, like the arena upload: a queue WriteBuffer into a buffer the GPU may still read stalls
    // a frame on GL drivers, and the record buffer is read by every resident draw of the frame.
    const wgpu::BufferDescriptor stagingDescriptor{
        .label = "Resident record upload",
        .usage = wgpu::BufferUsage::CopySrc,
        .size = upload.data.size(),
        .mappedAtCreation = true,
    };
    auto staging = webgpu::g_device.CreateBuffer(&stagingDescriptor);
    std::memcpy(staging.GetMappedRange(0, upload.data.size()), upload.data.data(), upload.data.size());
    staging.Unmap();
    encoder.CopyBufferToBuffer(staging, 0, record.buffer, upload.offset, upload.data.size());
  }
}

const wgpu::Buffer& record_buffer(u32 arena, uint64_t& size) noexcept {
  static const wgpu::Buffer none;
  if (arena >= sRecordBuffers.size()) {
    size = 0;
    return none;
  }
  size = sRecordBuffers[arena].size;
  return sRecordBuffers[arena].buffer;
}

void release_buffers() noexcept {
  sArenaBuffers.clear();
  sRecordBuffers.clear();
}
} // namespace resident

} // namespace aurora::gx
