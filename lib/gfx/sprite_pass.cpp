#include "sprite_pass.hpp"

#include "frame.hpp"
#include "gles_direct.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <array>
#include <mutex>

namespace aurora::gfx::sprite_pass {
using webgpu::g_device;

bool enabled() noexcept { return threshold() != 0; }
uint32_t threshold() noexcept { return g_config.halfResolutionSpritePoints; }

namespace {
std::mutex g_mutex;
webgpu::TextureWithSampler g_color;
webgpu::TextureWithSampler g_depth;
wgpu::TextureView g_sceneView;
wgpu::RenderPipeline g_prepPipeline, g_compositePipeline;
wgpu::BindGroupLayout g_prepLayout, g_compositeLayout;
wgpu::Sampler g_linearSampler;
EncoderTaskId g_prepTask = InvalidEncoderTask;
EncoderTaskId g_compositeTask = InvalidEncoderTask;

constexpr const char* PrepShader = R"(
@group(0) @binding(0) var scene_depth: texture_depth_2d;
var<private> positions: array<vec2f, 3> = array(vec2f(-1.0, 1.0), vec2f(-1.0, -3.0), vec2f(3.0, 1.0));
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    return vec4f(positions[vi], 0.0, 1.0);
}
struct Out { @builtin(frag_depth) depth: f32, @location(0) color: vec4f };
@fragment fn fs_main(@builtin(position) pos: vec4f) -> Out {
    let size = vec2i(textureDimensions(scene_depth));
    let coord = clamp(vec2i(pos.xy) * 2, vec2i(0), size - vec2i(1));
    var out: Out;
    out.depth = textureLoad(scene_depth, coord, 0);
    out.color = vec4f(0.0);
    return out;
}
)";
constexpr const char* CompositeShader = R"(
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var sprites: texture_2d<f32>;
struct VertexOutput { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
var<private> positions: array<vec2f, 3> = array(vec2f(-1.0, 1.0), vec2f(-1.0, -3.0), vec2f(3.0, 1.0));
var<private> uvs: array<vec2f, 3> = array(vec2f(0.0, 0.0), vec2f(0.0, 2.0), vec2f(2.0, 0.0));
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(sprites, samp, in.uv);
}
)";

wgpu::ShaderModule make_module(const char* source, const char* label) {
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = source}};
  const wgpu::ShaderModuleDescriptor desc{.nextInChain = &wgsl, .label = label};
  return g_device.CreateShaderModule(&desc);
}

void ensure_pipelines() {
  if (g_prepPipeline) {
    return;
  }
  const auto colorFormat = webgpu::g_graphicsConfig.surfaceConfiguration.format;
  {
    const std::array entries{wgpu::BindGroupLayoutEntry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture = {.sampleType = wgpu::TextureSampleType::Depth, .viewDimension = wgpu::TextureViewDimension::e2D},
    }};
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Sprite prep layout", .entryCount = entries.size(), .entries = entries.data()};
    g_prepLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const auto module = make_module(PrepShader, "Sprite prep");
    const std::array targets{wgpu::ColorTargetState{.format = colorFormat, .writeMask = wgpu::ColorWriteMask::None}};
    const wgpu::FragmentState fragment{
        .module = module, .entryPoint = "fs_main", .targetCount = targets.size(), .targets = targets.data()};
    const wgpu::DepthStencilState depth{
        .format = webgpu::g_graphicsConfig.depthFormat,
        .depthWriteEnabled = true,
        .depthCompare = wgpu::CompareFunction::Always,
    };
    const wgpu::PipelineLayoutDescriptor plDesc{.bindGroupLayoutCount = 1, .bindGroupLayouts = &g_prepLayout};
    const auto pipelineLayout = g_device.CreatePipelineLayout(&plDesc);
    const wgpu::RenderPipelineDescriptor desc{
        .label = "Sprite prep",
        .layout = pipelineLayout,
        .vertex = {.module = module, .entryPoint = "vs_main"},
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
        .depthStencil = &depth,
        .fragment = &fragment,
    };
    g_prepPipeline = g_device.CreateRenderPipeline(&desc);
  }
  {
    const std::array entries{
        wgpu::BindGroupLayoutEntry{.binding = 0,
                                   .visibility = wgpu::ShaderStage::Fragment,
                                   .sampler = {.type = wgpu::SamplerBindingType::Filtering}},
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Fragment,
            .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D}},
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Sprite composite layout", .entryCount = entries.size(), .entries = entries.data()};
    g_compositeLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const auto module = make_module(CompositeShader, "Sprite composite");
    // scene' = sprites.rgb + scene * (1 - coverage); the scene alpha is untouched.
    static const wgpu::BlendState blend{
        .color = {.operation = wgpu::BlendOperation::Add,
                  .srcFactor = wgpu::BlendFactor::One,
                  .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
        .alpha = {.operation = wgpu::BlendOperation::Add,
                  .srcFactor = wgpu::BlendFactor::Zero,
                  .dstFactor = wgpu::BlendFactor::One},
    };
    const std::array targets{wgpu::ColorTargetState{
        .format = colorFormat,
        .blend = &blend,
        .writeMask = wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue,
    }};
    const wgpu::FragmentState fragment{
        .module = module, .entryPoint = "fs_main", .targetCount = targets.size(), .targets = targets.data()};
    const wgpu::PipelineLayoutDescriptor plDesc{.bindGroupLayoutCount = 1, .bindGroupLayouts = &g_compositeLayout};
    const auto pipelineLayout = g_device.CreatePipelineLayout(&plDesc);
    const wgpu::RenderPipelineDescriptor desc{
        .label = "Sprite composite",
        .layout = pipelineLayout,
        .vertex = {.module = module, .entryPoint = "vs_main"},
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
        .fragment = &fragment,
    };
    g_compositePipeline = g_device.CreateRenderPipeline(&desc);
    const wgpu::SamplerDescriptor samplerDesc{
        .label = "Sprite composite sampler",
        .addressModeU = wgpu::AddressMode::ClampToEdge,
        .addressModeV = wgpu::AddressMode::ClampToEdge,
        .magFilter = wgpu::FilterMode::Linear,
        .minFilter = wgpu::FilterMode::Linear,
    };
    g_linearSampler = g_device.CreateSampler(&samplerDesc);
  }
}

void prep(const EncoderTaskContext&, const wgpu::CommandEncoder& cmd, const void*, size_t, void*) {
  ensure_pipelines();
  webgpu::TextureWithSampler color, depth;
  {
    std::lock_guard lock{g_mutex};
    color = g_color;
    depth = g_depth;
  }
  if (!color.view || !depth.view || !webgpu::g_depthBuffer.view) {
    return;
  }
  const std::array entries{wgpu::BindGroupEntry{.binding = 0, .textureView = webgpu::g_depthBuffer.view}};
  const wgpu::BindGroupDescriptor bgDesc{.layout = g_prepLayout, .entryCount = entries.size(), .entries = entries.data()};
  const auto bindGroup = g_device.CreateBindGroup(&bgDesc);
  const std::array attachments{wgpu::RenderPassColorAttachment{
      .view = color.view, .loadOp = wgpu::LoadOp::Clear, .storeOp = wgpu::StoreOp::Store, .clearValue = {0, 0, 0, 0}}};
  const wgpu::RenderPassDepthStencilAttachment depthAttachment{
      .view = depth.view,
      .depthLoadOp = wgpu::LoadOp::Clear,
      .depthStoreOp = wgpu::StoreOp::Store,
      .depthClearValue = 1.f,
  };
  const wgpu::RenderPassDescriptor passDesc{
      .label = "Sprite prep pass",
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .depthStencilAttachment = &depthAttachment,
      .timestampWrites = webgpu::gpu_prof::pass_writes("Sprite prep"),
  };
  const auto pass = cmd.BeginRenderPass(&passDesc);
  pass.SetPipeline(g_prepPipeline);
  pass.SetBindGroup(0, bindGroup);
  pass.Draw(3);
  pass.End();
}

void composite(const EncoderTaskContext&, const wgpu::CommandEncoder& cmd, const void*, size_t, void*) {
  ensure_pipelines();
  webgpu::TextureWithSampler color;
  {
    std::lock_guard lock{g_mutex};
    color = g_color;
  }
  const wgpu::TextureView scene = g_sceneView ? g_sceneView : webgpu::g_frameBuffer.view;
  if (!color.view || !scene) {
    return;
  }
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_linearSampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = color.view},
  };
  const wgpu::BindGroupDescriptor bgDesc{
      .layout = g_compositeLayout, .entryCount = entries.size(), .entries = entries.data()};
  const auto bindGroup = g_device.CreateBindGroup(&bgDesc);
  const std::array attachments{
      wgpu::RenderPassColorAttachment{.view = scene, .loadOp = wgpu::LoadOp::Load, .storeOp = wgpu::StoreOp::Store}};
  const wgpu::RenderPassDescriptor passDesc{
      .label = "Sprite composite pass",
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes("Sprite composite"),
  };
  const auto pass = cmd.BeginRenderPass(&passDesc);
  pass.SetPipeline(g_compositePipeline);
  pass.SetBindGroup(0, bindGroup);
  pass.Draw(3);
  pass.End();
}

void ensure_targets() {
  const auto& efb = webgpu::g_frameBuffer;
  const uint32_t width = std::max(efb.size.width / 2, 1u);
  const uint32_t height = std::max(efb.size.height / 2, 1u);
  std::lock_guard lock{g_mutex};
  if (g_color.view && g_color.size.width == width && g_color.size.height == height) {
    return;
  }
  g_color = webgpu::create_render_texture(width, height, false);
  const wgpu::TextureDescriptor depthDesc{
      .label = "Sprite depth",
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {width, height, 1},
      .format = webgpu::g_graphicsConfig.depthFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&depthDesc);
  auto view = texture.CreateView();
  g_depth = {.texture = std::move(texture),
             .view = std::move(view),
             .size = {width, height, 1},
             .format = webgpu::g_graphicsConfig.depthFormat};
}
} // namespace

const webgpu::TextureWithSampler& color_target() {
  ensure_targets();
  return g_color;
}

const webgpu::TextureWithSampler& depth_target() {
  ensure_targets();
  return g_depth;
}

EncoderTaskId prep_task() {
  if (g_prepTask == InvalidEncoderTask) {
    g_prepTask = register_encoder_task_type({.label = "Sprite prep", .callback = prep});
    // Only its own targets are touched: mapped frame streams need no upload for it.
    gles_direct::register_stream_independent_task(g_prepTask);
  }
  return g_prepTask;
}

EncoderTaskId composite_task() {
  if (g_compositeTask == InvalidEncoderTask) {
    g_compositeTask = register_encoder_task_type({.label = "Sprite composite", .callback = composite});
    gles_direct::register_stream_independent_task(g_compositeTask);
  }
  return g_compositeTask;
}

void set_scene_view(wgpu::TextureView view) { g_sceneView = std::move(view); }

void shutdown() {
  std::lock_guard lock{g_mutex};
  g_color = {};
  g_depth = {};
  g_sceneView = {};
  g_prepPipeline = {};
  g_compositePipeline = {};
  g_prepLayout = {};
  g_compositeLayout = {};
  g_linearSampler = {};
  if (g_prepTask != InvalidEncoderTask) {
    unregister_encoder_task_type(g_prepTask);
    g_prepTask = InvalidEncoderTask;
  }
  if (g_compositeTask != InvalidEncoderTask) {
    unregister_encoder_task_type(g_compositeTask);
    g_compositeTask = InvalidEncoderTask;
  }
}
} // namespace aurora::gfx::sprite_pass
