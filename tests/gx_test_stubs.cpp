// Stub implementations for renderer symbols that the GX/FIFO/command_processor code
// references but that live in the full renderer (gfx, gx.cpp, model/shader.cpp, etc.).
// These allow the test binary to link without pulling in WebGPU runtime.
//
// With AURORA_GX_TEST_LINK_GX defined, the test binary links lib/gx/gx.cpp and
// lib/gx/shader_info.cpp (plus the WebGPU library for the symbols they reference),
// so the stubs those files provide are left out and the few symbols they need are
// stubbed instead.

#include "gfx/hash.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/resource_cache.hpp"
#include "gx/gx.hpp"
#include "gfx/clear.hpp"
#include "gfx/resources.hpp"
#include "gfx/depth_peek.hpp"
#include "gfx/frame.hpp"
#include "gfx/recording.hpp"
#include "gfx/tex_copy_conv.hpp"
#include "gfx/tex_palette_conv.hpp"
#include "gfx/texture.hpp"
#include "gfx/texture_replacement.hpp"
#include "gx/pipeline.hpp"
#include "gx/shader_info.hpp"
#include "gx/texture.hpp"
#include "internal.hpp"
#include "webgpu/gpu.hpp"

#include <atomic>
#include <cstdio>
#include <functional>
#include <thread>
#include <fmt/format.h>

// --- aurora::g_config ---
namespace aurora {
AuroraConfig g_config{};
} // namespace aurora

// --- aurora::log_internal ---
namespace aurora {
void log_internal(AuroraLogLevel level, const char* module, const char* message, unsigned int len) noexcept {
  fprintf(stderr, "[%d] %s: %.*s\n", static_cast<int>(level), module, len, message);
}
} // namespace aurora

// --- fmt::formatter<AuroraLogLevel> ---
auto fmt::formatter<AuroraLogLevel>::format(AuroraLogLevel level, format_context& ctx) const
    -> format_context::iterator {
  return fmt::format_to(ctx.out(), "{}", static_cast<int>(level));
}

// --- GPU resources (default-constructed, not used in tests) ---
namespace aurora::gfx::detail {
Resources& resources() noexcept {
  static Resources resources;
  return resources;
}

void increment_merged_draw_count() noexcept {}
} // namespace aurora::gfx::detail

namespace aurora::webgpu {
GraphicsConfig g_graphicsConfig{};
#ifdef AURORA_GX_TEST_LINK_GX
wgpu::Device g_device;
bool g_hasCoreFeatures = false;
#endif
} // namespace aurora::webgpu

// --- GXState ---
namespace aurora::gx {
#ifndef AURORA_GX_TEST_LINK_GX
GXState g_gxState{};
void set_viewport_policy(AuroraViewportPolicy policy) noexcept {}
#endif
} // namespace aurora::gx

namespace aurora::vi {
Vec2<uint32_t> configured_fb_size() noexcept { return {640, 480}; }
void configure(const GXRenderModeObj*) noexcept {}
} // namespace aurora::vi

// --- get_texture ---
namespace aurora::gx {
#ifndef AURORA_GX_TEST_LINK_GX
const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[id]; }
#else
void clear_copy_texture_cache() noexcept {}
#endif
namespace texture {
void invalidate_bindings() noexcept {}
uint64_t current_bind_generation() noexcept { return 1; }
// Description hashes standing in for the texture cache's identities (FNV-1a; never 0).
u32 texture_object_identity(const GXTexObj_& obj) noexcept {
  const uint64_t fields[] = {reinterpret_cast<uintptr_t>(obj.data),
                             obj.width(),
                             obj.height(),
                             obj.format(),
                             obj.mode0 & 0x00FFFFFFu,
                             obj.mode1 & 0xFFFFu,
                             obj.flags & 3u,
                             static_cast<u32>(obj.tlut)};
  u32 hash = 0x811C9DC5u;
  for (const uint64_t field : fields) {
    for (int shift = 0; shift < 64; shift += 8) {
      hash = (hash ^ static_cast<u32>((field >> shift) & 0xFFu)) * 0x01000193u;
    }
  }
  return hash != 0 ? hash : 1;
}
u32 tlut_object_identity(const GXTlutObj_& tlut) noexcept {
  const uint64_t fields[] = {reinterpret_cast<uintptr_t>(tlut.data), static_cast<u32>(tlut.format), tlut.numEntries};
  u32 hash = 0x811C9DC5u;
  for (const uint64_t field : fields) {
    for (int shift = 0; shift < 64; shift += 8) {
      hash = (hash ^ static_cast<u32>((field >> shift) & 0xFFu)) * 0x01000193u;
    }
  }
  return hash != 0 ? hash : 1;
}
#ifdef AURORA_GX_TEST_LINK_GX
void shutdown() noexcept {}
#endif
} // namespace texture
void evict_texture_object(u32 texObjId) noexcept {
  for (auto& obj : g_gxState.loadedTextures) {
    if (obj.texObjId == texObjId) {
      obj.set_no_cache(true);
    }
  }
}
void evict_tlut_object(u32 tlutObjId) noexcept {
  for (auto& obj : g_gxState.loadedTluts) {
    if (obj.tlutObjId == tlutObjId) {
      obj.set_no_cache(true);
    }
  }
}
void evict_copy_texture(const void* dest) noexcept {
  g_gxState.copyTextures.erase(dest);
  for (auto it = g_gxState.copyTextureCache.begin(); it != g_gxState.copyTextureCache.end();) {
    if (it->first.dest == dest) {
      g_gxState.copyTextureCache.erase(it++);
    } else {
      ++it;
    }
  }
}
#ifndef AURORA_GX_TEST_LINK_GX
void shutdown() noexcept {}
Vec2<uint32_t> logical_fb_size() noexcept { return {640, 480}; }
gfx::Viewport map_logical_viewport(const gfx::Viewport& logicalViewport) noexcept { return logicalViewport; }
gfx::ClipRect map_logical_scissor(const gfx::ClipRect& logicalScissor) noexcept { return logicalScissor; }
void set_logical_viewport(const gfx::Viewport& viewport) noexcept {
  g_gxState.logicalViewport = viewport;
  set_render_viewport(map_logical_viewport(viewport));
}
void set_render_viewport(const gfx::Viewport& viewport) noexcept { g_gxState.renderViewport = viewport; }
void set_logical_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.logicalScissor = scissor;
  set_render_scissor(map_logical_scissor(scissor));
}
void set_render_scissor(const gfx::ClipRect& scissor) noexcept { g_gxState.renderScissor = scissor; }
#endif
} // namespace aurora::gx

// --- Shader/pipeline stubs ---
namespace aurora::gx {
#ifndef AURORA_GX_TEST_LINK_GX
void populate_pipeline_config(PipelineConfig& config, GXPrimitive primitive, GXVtxFmt fmt) noexcept {
  // No-op for tests
}
GXBindGroups build_bind_groups(const ShaderInfo& info) noexcept { return {}; }
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept { return {}; }
gfx::Range build_uniform(const ShaderInfo& info) noexcept { return {.size = 1}; }
#endif
void resolve_sampled_textures(const ShaderInfo& info) noexcept {}
bool resolve_sampled_textures(const ShaderInfo& info) noexcept { return false; }
} // namespace aurora::gx

// --- Buffer push stubs ---
namespace aurora::gfx {
Range push_verts(const uint8_t* data, size_t length, size_t alignment) { return {}; }
Range push_indices(const uint8_t* data, size_t length, size_t alignment) { return {}; }
Range push_uniform(const uint8_t* data, size_t length) { return {}; }
Range push_storage(const uint8_t* data, size_t length) { return {}; }

Vec2<uint32_t> get_render_target_size() noexcept { return {640, 480}; }
void set_viewport(const Viewport& viewport) noexcept {}
void set_scissor(uint32_t x, uint32_t y, uint32_t w, uint32_t h) noexcept {}
uint32_t get_sample_count() noexcept { return 1; }
#ifdef AURORA_GX_TEST_LINK_GX
void set_scissor(const ClipRect& scissor) noexcept {}
uint32_t align_uniform(uint32_t value) { return value; }
BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor) { return 0; }
wgpu::Sampler sampler_ref(const wgpu::SamplerDescriptor& descriptor) { return {}; }
#endif
RenderTargetLayout get_render_target_layout() noexcept {
  return {
      .colorAttachmentCount = 1,
      .colorAttachments = {{{ColorAttachmentSemantic::SceneColor, wgpu::TextureFormat::RGBA8Unorm}}},
      .depthStencilFormat = wgpu::TextureFormat::Depth24Plus,
      .sampleCount = 1,
  };
}
} // namespace aurora::gfx

// --- Pipeline/draw command stubs ---
namespace aurora::gfx {
namespace clear {
PipelineConfig make_pipeline_config(const RenderTargetLayout& layout, bool clearColor, bool clearAlpha,
                                    bool clearDepth) noexcept {
  return {
      .targetLayoutKey = layout.key,
      .depthStencilFormat = layout.depthStencilFormat,
      .colorAttachmentCount = layout.colorAttachmentCount,
      .msaaSamples = layout.sampleCount,
      .clearColor = clearColor,
      .clearAlpha = clearAlpha,
      .clearDepth = clearDepth,
  };
}
} // namespace clear

template <>
PipelineRef pipeline_ref<clear::PipelineConfig>(const clear::PipelineConfig& config) {
  return 0;
}
template <>
void push_draw_command<clear::DrawData>(clear::DrawData data) {
  // No-op
}
template <>
PipelineRef pipeline_ref<gx::PipelineConfig>(const gx::PipelineConfig& config) {
  // Same key as the pipeline cache derives, without creating a pipeline.
  return xxh3_hash(config, static_cast<HashType>(ShaderType::GX));
}
gx::DrawData g_testLastDraw{};
uint32_t g_testDrawCount = 0;
std::atomic<uint32_t> g_testProcessedDrawCount{0};

template <>
void push_draw_command<gx::DrawData>(gx::DrawData data) {
  g_testLastDraw = data;
  ++g_testDrawCount;
  g_testProcessedDrawCount.fetch_add(1, std::memory_order_release);
}
template <>
gx::DrawData* get_last_draw_command() {
  return nullptr;
}
} // namespace aurora::gfx

// --- TextureBind::get_descriptor ---
namespace aurora::gfx {
wgpu::SamplerDescriptor TextureBind::get_descriptor() const noexcept { return wgpu::SamplerDescriptor{}; }
} // namespace aurora::gfx

// --- Texture creation/write/replacement stubs ---
namespace aurora::gfx {
TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                    ArrayRef<uint8_t> data, bool tlut, const char* label,
                                    std::optional<TextureClass> textureClass) noexcept {
  return {};
}
TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat, const char* label,
                                     std::optional<TextureClass> textureClass) noexcept {
  return {};
}
TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  return {};
}
TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept { return {}; }
void write_texture(TextureRef& ref, ArrayRef<uint8_t> data) noexcept {}
void queue_texture_upload(TextureUpload upload) {}
void queue_texture_upload_data(const uint8_t* data, size_t length, uint32_t bytesPerRow, uint32_t rowsPerImage,
                               wgpu::TexelCopyTextureInfo tex, wgpu::Extent3D size) {}
void queue_palette_conv(tex_palette_conv::ConvRequest req) {}
namespace testing {
std::atomic<uint32_t> beginOffscreenCount{0};
std::atomic<uint32_t> endOffscreenCount{0};
std::atomic<uint32_t> resolvePassCount{0};
std::atomic<uint32_t> offscreenWidth{0};
std::atomic<uint32_t> offscreenHeight{0};
} // namespace testing

void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat) {
  testing::resolvePassCount.fetch_add(1, std::memory_order_release);
}
void begin_offscreen(uint32_t width, uint32_t height) {
  testing::offscreenWidth.store(width, std::memory_order_relaxed);
  testing::offscreenHeight.store(height, std::memory_order_relaxed);
  testing::beginOffscreenCount.fetch_add(1, std::memory_order_release);
}
void end_offscreen() { testing::endOffscreenCount.fetch_add(1, std::memory_order_release); }
bool is_offscreen() noexcept { return false; }

// --- Asynchronous frame markers ---
namespace testing {
std::atomic<uint32_t> reservedFrameBeginCount{0};
std::atomic<uint32_t> reservedFrameBeginSlot{UINT32_MAX};
std::atomic<uint32_t> reservedFrameBeginBpReg41{0};
std::atomic<uint32_t> deferredFrameEndCount{0};
std::atomic<uint32_t> finishCount{0};
std::atomic<size_t> deferredFrameEndThreadHash{0};
} // namespace testing

void begin_reserved_frame(uint32_t frameSlot) {
  testing::reservedFrameBeginSlot.store(frameSlot, std::memory_order_relaxed);
  testing::reservedFrameBeginBpReg41.store(gx::g_gxState.bpRegCache[0x41], std::memory_order_relaxed);
  testing::reservedFrameBeginCount.fetch_add(1, std::memory_order_release);
}
void finish() { testing::finishCount.fetch_add(1, std::memory_order_release); }
void end_deferred_frame() {
  testing::deferredFrameEndThreadHash.store(std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                            std::memory_order_relaxed);
  testing::deferredFrameEndCount.fetch_add(1, std::memory_order_release);
}
} // namespace aurora::gfx

namespace aurora::gx::texture {
void end_frame() noexcept {}
} // namespace aurora::gx::texture

namespace aurora::gfx::depth_peek {
namespace {
bool s_snapshotRequested = false;
uint32_t s_width = 0;
uint32_t s_height = 0;
std::vector<uint32_t> s_data;
} // namespace

void initialize() {}
void shutdown() {}
void request_snapshot() noexcept { s_snapshotRequested = true; }
void poll() noexcept {}
void encode_frame_snapshot(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView,
                           wgpu::Extent3D sourceSize, uint32_t msaaSamples) noexcept {}
void after_submit() noexcept {}

bool read_latest(uint16_t x, uint16_t y, uint32_t& z) noexcept {
  if (x >= s_width || y >= s_height || s_data.empty()) {
    return false;
  }
  z = s_data[static_cast<size_t>(y) * s_width + x] & 0x00ffffffu;
  return true;
}

namespace testing {
void reset() noexcept {
  s_snapshotRequested = false;
  s_width = 0;
  s_height = 0;
  s_data.clear();
}

bool snapshot_requested() noexcept { return s_snapshotRequested; }

void set_latest(uint32_t width, uint32_t height, const std::vector<uint32_t>& data) {
  s_width = width;
  s_height = height;
  s_data = data;
}
} // namespace testing
} // namespace aurora::gfx::depth_peek

namespace aurora::gfx::tex_copy_conv {
bool needs_conversion(GXTexFmt fmt) { return false; }
} // namespace aurora::gfx::tex_copy_conv

namespace aurora::gfx::tex_palette_conv {
void queue(ConvRequest req) {}
} // namespace aurora::gfx::tex_palette_conv

namespace aurora::gfx::texture_replacement {
u32 compute_texture_upload_size(const GXTexObj_& obj) noexcept { return 0; }
bool has_replacement(const GXTexObj_&) noexcept { return false; }
bool has_replacement(const GXTexObj_&, const GXTlutObj_&) noexcept { return false; }
} // namespace aurora::gfx::texture_replacement

// --- Window stub ---
#include "../lib/window.hpp"
namespace aurora::window {
AuroraWindowSize get_window_size() { return {640, 480, 640, 480, 640, 480, 1.0f}; }
void set_frame_buffer_aspect_fit(bool) {}
} // namespace aurora::window

// --- WebGPU C API stubs (prevent linker errors from wgpu:: destructors) ---
#ifndef AURORA_GX_TEST_LINK_GX
extern "C" {
void wgpuDeviceRelease(WGPUDevice) {}
void wgpuQueueRelease(WGPUQueue) {}
void wgpuSurfaceRelease(WGPUSurface) {}
void wgpuBufferRelease(WGPUBuffer) {}
void wgpuTextureRelease(WGPUTexture) {}
void wgpuTextureViewRelease(WGPUTextureView) {}
void wgpuSamplerRelease(WGPUSampler) {}
void wgpuShaderModuleRelease(WGPUShaderModule) {}
void wgpuRenderPipelineRelease(WGPURenderPipeline) {}
void wgpuBindGroupRelease(WGPUBindGroup) {}
void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout) {}
void wgpuPipelineLayoutRelease(WGPUPipelineLayout) {}
void wgpuInstanceRelease(WGPUInstance) {}
void wgpuDeviceAddRef(WGPUDevice) {}
void wgpuQueueAddRef(WGPUQueue) {}
void wgpuSurfaceAddRef(WGPUSurface) {}
void wgpuBufferAddRef(WGPUBuffer) {}
void wgpuTextureAddRef(WGPUTexture) {}
void wgpuTextureViewAddRef(WGPUTextureView) {}
void wgpuInstanceAddRef(WGPUInstance) {}
}
#endif

void aurora::gfx::push_debug_group(std::string) {}
void push_debug_group(const char*) {}
void pop_debug_group() {}
void aurora::gfx::insert_debug_marker(std::string) {}
