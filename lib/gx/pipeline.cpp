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

#include <algorithm>
#include <cstring>
#include <vector>

namespace aurora::gx {

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const auto shader = build_shader(config.shaderConfig);
  const auto label =
      fmt::format("GX Pipeline {:x} shader {:x}", xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)),
                  xxh3_hash(config.shaderConfig));
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
  if (data.residentArena != 0) {
    uint64_t arenaSize = 0;
    const auto& arena = resident::arena_buffer(data.residentArena - 1, arenaSize);
    if (!arena) {
      return;
    }
    pass.SetVertexBuffer(0, arena, 0, arenaSize);
    indexFormat = wgpu::IndexFormat::Uint32;
  } else if (g_config.cpuVertexDecode) {
    pass.SetVertexBuffer(0, resources.vertexBuffer, data.vertRange.offset, data.vertRange.size);
  }
  pass.SetImmediates(0, &data.immediateData, sizeof(data.immediateData));
  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, resources.uniformBindGroup, offsets.size(), offsets.data());
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

void release_buffers() noexcept { sArenaBuffers.clear(); }
} // namespace resident

} // namespace aurora::gx
