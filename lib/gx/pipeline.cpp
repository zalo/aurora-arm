#include "pipeline.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/resource_cache.hpp"

#include "gx_fmt.hpp"
#include "shader_info.hpp"
#include "vertex_loader.hpp"

#include <tracy/Tracy.hpp>

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
  if (g_config.cpuVertexDecode) {
    pass.SetVertexBuffer(0, resources.vertexBuffer, data.vertRange.offset, data.vertRange.size);
  }
  pass.SetImmediates(0, &data.immediateData, sizeof(data.immediateData));
  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, resources.uniformBindGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
  }
  pass.SetIndexBuffer(resources.indexBuffer, wgpu::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
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

} // namespace aurora::gx
