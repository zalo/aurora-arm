#include <aurora/aurora.h>
#include <aurora/time.hpp>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#ifdef AURORA_ENABLE_GX
#include "gfx/resources.hpp"
#include "gfx/frame.hpp"
#include "gfx/gles_direct.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/recording.hpp"
#include "gfx/render_worker.hpp"
#include "gx/command_processor.hpp"
#include "gx/fifo.hpp"
#include "gx/gx.hpp"
#include "gx/texture.hpp"
#include "imgui.hpp"
#include "webgpu/gpu.hpp"
#include "webgpu/gpu_prof.hpp"
#include <webgpu/webgpu_cpp.h>
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "input.hpp"
#include "internal.hpp"
#include "thread.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include "system_info.hpp"
#include "tracy/Tracy.hpp"

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
constexpr Module Log{"aurora"};

#ifdef AURORA_ENABLE_GX
// GPU
using webgpu::g_device;
using webgpu::g_queue;
using webgpu::g_surface;

uint32_t clamp_scissor_coord(double value, uint32_t maximum) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  return static_cast<uint32_t>(std::clamp(value, 0.0, static_cast<double>(maximum)));
}

void set_present_viewport(const wgpu::RenderPassEncoder& pass, const gfx::Viewport& viewport, uint32_t surfaceWidth,
                          uint32_t surfaceHeight) noexcept {
  pass.SetViewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);
  const auto scissorX = clamp_scissor_coord(std::floor(viewport.left), surfaceWidth);
  const auto scissorY = clamp_scissor_coord(std::floor(viewport.top), surfaceHeight);
  const auto scissorRight = clamp_scissor_coord(std::ceil(viewport.left + viewport.width), surfaceWidth);
  const auto scissorBottom = clamp_scissor_coord(std::ceil(viewport.top + viewport.height), surfaceHeight);
  pass.SetScissorRect(scissorX, scissorY, scissorRight - scissorX, scissorBottom - scissorY);
}
#endif

#ifdef AURORA_ENABLE_GX
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
#ifdef DAWN_ENABLE_BACKEND_OPENGLES
    BACKEND_OPENGLES,
#endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
#else
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  log_system_information();
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  if (g_config.textureVerifyInterval == 0) {
    g_config.textureVerifyInterval = 1;
  }
  AURORA_ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  AURORA_ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  AURORA_ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  AURORA_ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  AURORA_ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();
  thread::set_current({
      .name = "Main thread",
      .affinity = thread::Affinity::SharedCache,
  });

#ifdef AURORA_ENABLE_GX
  gfx::initialize();
  gx::fifo::init();
  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

void shutdown() noexcept {
#ifdef AURORA_ENABLE_GX
  gx::fifo::shutdown();
  gfx::render_worker::synchronize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
#endif
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
  }
#ifdef AURORA_ENABLE_GX
  gx::update();
#endif
  return window::poll_events();
}

#ifdef AURORA_ENABLE_GX
// The Drain processing mode only translates at drain(), which asynchronous frames never call.
bool async_frames() noexcept {
  return g_config.asyncFrames && gx::fifo::processing_mode() != gx::fifo::ProcessingMode::Drain;
}

void finish_frame(gfx::EndFrameCallback callback) {
  if (async_frames()) {
    // The processor finishes the frame and hands the callback to the render worker when it
    // reaches the end marker; store the callback before the marker is published.
    gfx::defer_end_frame(std::move(callback));
    gx::fifo::end_frame_async();
  } else {
    gfx::end_frame(std::move(callback));
  }
}
#endif

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  {
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (!g_surface) {
      webgpu::refresh_surface(true);
      if (!g_surface) {
        return false;
      }
    }
  }

  imgui::new_frame(window::get_window_size());
  gx::fifo::recycle();
  if (async_frames()) {
    // The processor begins recording into the slot when it reaches the marker.
    uint32_t frameSlot;
    if (!gfx::reserve_frame(frameSlot)) {
      return false;
    }
    gx::fifo::begin_frame_async(frameSlot);
  } else {
    if (!gfx::begin_frame()) {
      return false;
    }
    gx::fifo::begin_frame();
  }
#endif
  return true;
}

void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  const bool asyncFrame = async_frames();
  if (!asyncFrame) {
    gx::fifo::drain();
    gx::fifo::end_frame();
    gx::texture::end_frame();
    gfx::finish();
  }
  auto imguiDrawData = imgui::freeze();

  const auto& presentSource = webgpu::present_source();
  const auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                           webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                           presentSource.size.width, presentSource.size.height);

  wgpu::BindGroup rmlBindGroup;
  bool rmlOverlay = false;
#if AURORA_ENABLE_RMLUI
  if (rmlui::is_initialized()) {
    if (asyncFrame) {
      // RmlUi records into the frame from this thread: let the processor catch up first.
      gx::fifo::drain();
    }
    auto rmlFrame = rmlui::record_frame(viewport);
    rmlBindGroup = std::move(rmlFrame.bindGroup);
    rmlOverlay = rmlFrame.overlay;
  }
#endif

  finish_frame([rmlBindGroup = std::move(rmlBindGroup), rmlOverlay, viewport, imguiDrawData = std::move(imguiDrawData)](
                   wgpu::CommandEncoder& encoder, std::vector<gfx::AfterSubmitCallback> afterSubmitCallbacks) {
    wgpu::Texture currentTexture;
    wgpu::TextureView currentView;
    auto surfaceStatus = wgpu::SurfaceGetCurrentTextureStatus::Error;
    // Renderer phase timing (renderStats): acquire, encode, submit and present per frame.
    auto phaseNow = std::chrono::steady_clock::now();
    const auto phase = [&](double& accumulator) {
      if (!g_config.renderStats) {
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      accumulator += std::chrono::duration<double, std::milli>(now - phaseNow).count();
      phaseNow = now;
    };
    static double sAcquireMs = 0, sEncodeMs = 0, sSubmitMs = 0, sPresentMs = 0;
    static unsigned sPhaseFrames = 0;
    // Scene on surface: the frame's EFB passes already rendered into a presented texture acquired at encode time.
    bool sceneOnSurface = false;
    if (auto [surfaceTexture, surfaceView] = gfx::take_frame_surface(); surfaceTexture) {
      currentTexture = std::move(surfaceTexture);
      currentView = std::move(surfaceView);
      surfaceStatus = wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal;
      sceneOnSurface = true;
    } else {
      window::SurfaceLock surfaceLock;
      if (window::is_presentable() && g_surface) {
        ZoneScopedN("Acquire texture");
        wgpu::SurfaceTexture surfaceTexture;
        g_surface.GetCurrentTexture(&surfaceTexture);
        surfaceStatus = surfaceTexture.status;
        if (surfaceStatus == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal) {
          currentTexture = std::move(surfaceTexture.texture);
          currentView = currentTexture.CreateView();
        }
      }
    }
    phase(sAcquireMs);

    const bool canPresent = currentTexture && currentView;
    const auto& surfaceConfig = webgpu::g_graphicsConfig.surfaceConfiguration;
    const auto viewportWidth = static_cast<uint32_t>(viewport.width + 0.5f);
    const auto viewportHeight = static_cast<uint32_t>(viewport.height + 0.5f);
    const bool fullSurfaceViewport = viewport.left == 0.f && viewport.top == 0.f &&
                                     viewportWidth == surfaceConfig.width && viewportHeight == surfaceConfig.height;
    // The scene is already in the presented texture: skip the present copy pass when nothing else would be
    // composited into it. AURORA_SCENE_MIRROR=1 copies it back into the EFB texture for capture tooling.
    const bool skipPresentCopy = sceneOnSurface && !(rmlBindGroup && rmlOverlay) && fullSurfaceViewport;
    if (canPresent && skipPresentCopy) {
      static const bool mirror = [] {
        const char* value = std::getenv("AURORA_SCENE_MIRROR");
        return value != nullptr && value[0] == '1';
      }();
      if (mirror) {
        const auto& efb = webgpu::present_source();
        const wgpu::TexelCopyTextureInfo src{.texture = currentTexture};
        const wgpu::TexelCopyTextureInfo dst{.texture = efb.texture};
        const wgpu::Extent3D size{efb.size.width, efb.size.height, 1};
        encoder.CopyTextureToTexture(&src, &dst, &size);
      }
    }
    if (canPresent && !skipPresentCopy) {
      wgpu::BindGroup presentBindGroup;
      if (rmlBindGroup && !rmlOverlay) {
        presentBindGroup = rmlBindGroup;
      } else {
        // At 1:1 the resample pass is an identity: copy the EFB directly and save a full-screen render pass.
        const auto& efbSource = webgpu::present_source();
        if (efbSource.size.width == viewportWidth && efbSource.size.height == viewportHeight) {
          presentBindGroup = webgpu::create_copy_bind_group(efbSource);
        } else {
          const auto& resampledSource = webgpu::resample_present_source(encoder, viewport);
          presentBindGroup = webgpu::create_copy_bind_group(resampledSource);
        }
      }
      {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = currentView,
                .loadOp = wgpu::LoadOp::Clear,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "EFB copy render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("Present blit"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        // Copy EFB -> XFB (swapchain)
        pass.SetPipeline(webgpu::g_CopyPipeline);
        pass.SetBindGroup(0, presentBindGroup, 0, nullptr);
        set_present_viewport(pass, viewport, webgpu::g_graphicsConfig.surfaceConfiguration.width,
                             webgpu::g_graphicsConfig.surfaceConfiguration.height);

        pass.Draw(3);
        if (rmlBindGroup && rmlOverlay) {
          pass.SetPipeline(webgpu::g_CopyPremultipliedAlphaPipeline);
          pass.SetBindGroup(0, rmlBindGroup, 0, nullptr);
          pass.Draw(3);
        }
        pass.End();
      }
    }
    if (canPresent) {
      // An empty overlay still costs a full render pass (load, store, and on
      // tile-based mobile drivers ~0.5 ms of driver time). Skip it when ImGui
      // recorded nothing this frame.
      if (imgui::has_draws(imguiDrawData)) {
        const std::array attachments{
            wgpu::RenderPassColorAttachment{
                .view = currentView,
                .loadOp = wgpu::LoadOp::Load,
                .storeOp = wgpu::StoreOp::Store,
            },
        };
        const wgpu::RenderPassDescriptor renderPassDescriptor{
            .label = "ImGui render pass",
            .colorAttachmentCount = attachments.size(),
            .colorAttachments = attachments.data(),
            .timestampWrites = webgpu::gpu_prof::pass_writes("ImGui"),
        };
        const auto pass = encoder.BeginRenderPass(&renderPassDescriptor);
        pass.SetViewport(0.f, 0.f, static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.width),
                         static_cast<float>(webgpu::g_graphicsConfig.surfaceConfiguration.height), 0.f, 1.f);
        imgui::render(pass, imguiDrawData);
        pass.End();
      }
    } else {
      Log.info("Skipping present; window not presentable");
    }
    webgpu::gpu_prof::frame_end(encoder);
    const wgpu::CommandBufferDescriptor cmdBufDescriptor{.label = "Redraw command buffer"};
    const auto buffer = encoder.Finish(&cmdBufDescriptor);
    phase(sEncodeMs);
    {
      ZoneScopedN("Queue Submit");
      // OpenGL ES direct submission: the render pass callback is live only while this command buffer executes.
      gfx::gles_direct::install_frame();
      g_queue.Submit(1, &buffer);
      gfx::gles_direct::uninstall_frame();
    }
    phase(sSubmitMs);
    webgpu::gpu_prof::after_submit();
    if (canPresent && g_surface) {
      ZoneScopedN("Present");
      wgpu::ConvertibleStatus status = wgpu::Status::Error;
      {
        window::SurfaceLock surfaceLock;
        if (window::is_presentable()) {
          status = g_surface.Present();
        }
      }
      phase(sPresentMs);
      if (g_config.renderStats && ++sPhaseFrames % 120 == 0) {
        std::fprintf(stderr, "[render-phase] acquire_ms=%.3f encode_ms=%.3f submit_ms=%.3f present_ms=%.3f\n",
                     sAcquireMs / 120, sEncodeMs / 120, sSubmitMs / 120, sPresentMs / 120);
        sAcquireMs = sEncodeMs = sSubmitMs = sPresentMs = 0;
      }
      if (status) {
        gfx::after_present();
      } else {
        Log.warn("Surface present failed");
        webgpu::release_surface();
      }
    } else if (g_surface) {
      switch (surfaceStatus) {
      case wgpu::SurfaceGetCurrentTextureStatus::Timeout:
        Log.warn("Surface texture acquisition timed out");
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal:
      case wgpu::SurfaceGetCurrentTextureStatus::Outdated:
        Log.info("Surface texture is {}, reconfiguring swapchain", magic_enum::enum_name(surfaceStatus));
        window::push_custom_event(window::CustomEvent::RefreshSurface);
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Lost:
        Log.warn("Surface texture is {}, releasing surface", magic_enum::enum_name(surfaceStatus));
        webgpu::release_surface();
        break;
      case wgpu::SurfaceGetCurrentTextureStatus::Error:
        Log.warn("Surface texture is {}, dropping surface", magic_enum::enum_name(surfaceStatus));
        g_surface = {};
        break;
      default:
        if (!window::is_presentable()) {
          webgpu::release_surface();
        } else {
          Log.error("Failed to get surface texture: {}", magic_enum::enum_name(surfaceStatus));
        }
        break;
      }
    }
    for (auto& callback : afterSubmitCallbacks) {
      if (callback) {
        callback();
      }
    }
    gfx::after_submit();

    TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
    TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);

    const auto& stats = gfx::detail::resources().stats;
    TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(stats.queuedPipelines));
    TracyPlot("aurora: createdPipelines", static_cast<int64_t>(stats.createdPipelines));
    TracyPlot("aurora: drawCallCount", static_cast<int64_t>(stats.drawCallCount));
    TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(stats.mergedDrawCallCount));
    TracyPlot("aurora: lastVertSize", static_cast<int64_t>(stats.lastVertSize));
    TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(stats.lastUniformSize));
    TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(stats.lastIndexSize));
    TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(stats.lastStorageSize));
    TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(stats.lastTextureUploadSize));
  });

#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
#ifdef AURORA_ENABLE_GX
uint64_t aurora_render_stats_fifo_wait_ns() { return aurora::gx::fifo::wait_ns(); }
uint64_t aurora_render_stats_fifo_process_ns() { return aurora::gx::fifo::process_ns(); }
uint64_t aurora_render_stats_render_worker_busy_ns() { return aurora::gfx::render_worker::busy_ns(); }
uint64_t aurora_render_stats_pipeline_wait_ns() { return aurora::gfx::pipeline_wait_ns(); }
uint64_t aurora_render_stats_pipeline_wait_count() { return aurora::gfx::pipeline_wait_count(); }
#else
uint64_t aurora_render_stats_fifo_wait_ns() { return 0; }
uint64_t aurora_render_stats_fifo_process_ns() { return 0; }
uint64_t aurora_render_stats_render_worker_busy_ns() { return 0; }
uint64_t aurora_render_stats_pipeline_wait_ns() { return 0; }
uint64_t aurora_render_stats_pipeline_wait_count() { return 0; }
#endif
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
void aurora_set_resampler(AuroraSampler sampler) {
#ifdef AURORA_ENABLE_GX
  aurora::webgpu::set_resampler(sampler);
#else
  (void)sampler;
#endif
}
void aurora_set_timescale(float scale) { aurora::time::set_scale(scale); }
float aurora_get_timescale() { return aurora::time::scale(); }
