#ifndef AURORA_AURORA_H
#define AURORA_AURORA_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

extern "C" {
#else
#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"
#endif

typedef enum {
  SAMPLER_BILINEAR,
  SAMPLER_AREA,
} AuroraSampler;

typedef enum {
  BACKEND_AUTO,
  BACKEND_D3D11,
  BACKEND_D3D12,
  BACKEND_METAL,
  BACKEND_VULKAN,
  BACKEND_OPENGL,
  BACKEND_OPENGLES,
  BACKEND_WEBGPU,
  BACKEND_NULL,
} AuroraBackend;

typedef enum {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_FATAL,
} AuroraLogLevel;

typedef struct {
  int32_t x;
  int32_t y;
} AuroraWindowPos;

typedef struct {
  uint32_t width;
  uint32_t height;

  /**
   * Width of the main GX framebuffer.
   */
  uint32_t fb_width;

  /**
   * Height of the main GX framebuffer.
   */
  uint32_t fb_height;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_width if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_width;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_height if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_height;
  float scale;
} AuroraWindowSize;

typedef struct SDL_Window SDL_Window;
typedef struct AuroraEvent AuroraEvent;

typedef void (*AuroraLogCallback)(AuroraLogLevel level, const char* module, const char* message, unsigned int len);
typedef void (*AuroraImGuiInitCallback)(const AuroraWindowSize* size);

#define MEM1_DEFAULT_SIZE (24 * 1024 * 1024)
#define ARAM_DEFAULT_SIZE (16 * 1024 * 1024)

typedef struct {
  const char* appName;
  const char* userPath;
  const char* cachePath;
  const char* resourcesPath;
  AuroraBackend desiredBackend;
  uint32_t msaa;
  uint16_t maxTextureAnisotropy;
  bool vsync;
  bool startFullscreen;
  bool allowJoystickBackgroundEvents;
  bool pauseOnFocusLost;
  bool allowTextureDumps;
  bool allowCpuAdapter;

  /*
   * Pack single-mip textures that clamp on both axes into shared 1024x1024
   * atlas layers so that more draws share a texture bind group and can merge.
   * The shader remaps texel coordinates into the atlas cell, which can shift
   * sampled values by one LSB compared to sampling the texture directly;
   * leave off for bit-exact output.
   */
  bool textureAtlas;
  int32_t windowPosX;
  int32_t windowPosY;
  uint32_t windowWidth;
  uint32_t windowHeight;
  void* iconRGBA8;
  uint32_t iconWidth;
  uint32_t iconHeight;
  AuroraLogCallback logCallback;
  AuroraLogLevel logLevel;
  AuroraImGuiInitCallback imGuiInitCallback;

  /*
   * The size of the GameCube's main memory, or MEM1 on the Wii.
   * Note that it will not be allocated at the exact 0x80000000 address, as that cannot be guaranteed.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem1Size;

  /*
   * The size of the GameCube's ARAM, or MEM2 on the Wii.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem2Size;

  /*
   * How often, in frames, the GX texture cache re-checks the source contents of a texture object it has already
   * uploaded. Texture identities are derived from the object description, so an image rewritten in place without
   * GXInitTexObjData is only caught by this sampled content check.
   * 0 or 1 (default) checks every frame the object is resolved. A larger value checks at most once every N frames,
   * which saves CPU on slow hosts but can show such an in-place rewrite up to N-1 frames late.
   * GXInitTexObjData / GXInitTlutObjData changes always invalidate immediately.
   */
  uint32_t textureVerifyInterval;

  /*
   * Translate GX commands asynchronously to the game thread: aurora_end_frame() no longer
   * waits for the FIFO processor, so the CPU frame time becomes the slowest of the game
   * thread, the FIFO processor and the render worker instead of game + translation.
   * The game thread may then run up to one frame ahead of translation; GXSetDrawDone /
   * GXWaitDrawDone and AuroraGXSync wait for a specific point of the stream.
   */
  bool asyncFrames;

  /*
   * Disables fusing adjacent small render-to-texture EFB passes (shadow maps, reflections) into one render pass.
   * Fusion is exact and on by default; this is a debugging switch for bisecting rendering differences.
   */
  bool disableRenderPassFusion;

  /*
   * Decode GX vertex attributes on the CPU into conventional vertex buffers instead of fetching
   * them from storage buffers in the vertex shader. Produces the same vertex values; intended for
   * GLES-class GPUs where vertex-shader storage buffer reads are slow or unavailable.
   */
  bool cpuVertexDecode;

  /*
   * Keep geometry decoded from GXCallDisplayList resident on the GPU (requires cpuVertexDecode).
   * Each distinct display list is decoded once and later calls only reference it. The application
   * must call GXInvalidateResidentGeometry() before it frees or rewrites memory that held display
   * lists or the vertex arrays they index; rewriting the first or last 64 bytes of a list in place
   * is detected without it. GXInvalidateVtxCache() does not affect resident geometry.
   */
  bool residentDisplayLists;

  /*
   * Bytes of decoded display-list geometry kept resident before the cache is emptied and rebuilt
   * (at a frame boundary). 0 selects the default of 64 MiB.
   */
  uint32_t residentGeometryBudget;

  /*
   * Place each GX draw's uniform record in a 4 KiB slot of a 64 KiB uniform window and bind one
   * window per sixteen records instead of re-binding the uniform buffer at a new dynamic offset for
   * every draw; the shader indexes the record within the window. Removes most per-draw buffer
   * binding work on GLES drivers and is required by batchDraws. Bit-exact.
   */
  bool uniformTable;

  /*
   * Merge adjacent GX draws that share a pipeline, texture bind group, destination alpha, fog range
   * table and uniform window into one draw call, even when their uniform records differ (requires
   * uniformTable and cpuVertexDecode: the record index travels in bits 8-23 of each decoded vertex's
   * matrix word). Bit-exact; on a Mali-G52 handheld running a GX title this roughly halves the draw
   * calls of a frame.
   */
  bool batchDraws;

  /*
   * Submit GX render passes directly through OpenGL ES (build option AURORA_GLES_DIRECT, needs Dawn's
   * native GL interop extension). Dawn keeps resource ownership, uploads and render pass setup; the GX
   * draws of an eligible pass are issued by Aurora in the device's GL context with a redundant-state
   * filter, the uniform table bound once per window and texture state folded into the texture objects.
   * Ineligible passes (custom draws, MSAA, fog range tables, a pipeline still compiling) take the WebGPU
   * path unchanged. Requires cpuVertexDecode and implies uniformTable and batchDraws.
   * 0 = automatic (on when the OpenGL ES backend is selected and the extension is available), 1 = on,
   * -1 = off. Measured on a Mali-G52 handheld running a GX title: render work 47 -> ~13 ms per frame.
   */
  int8_t glesDirectSubmission;

  /*
   * With direct submission, record the frame's uniform records, indices and decoded vertices straight into
   * persistently mapped GL buffers (GL_EXT_buffer_storage) with one fenced slot per staging buffer, so the
   * direct path binds them without a staging copy. 0 = automatic (on with direct submission), 1 = on,
   * -1 = off (frames stay staged through WebGPU buffers).
   */
  int8_t glesMappedStreams;

  /*
   * Direct submission only: reorder runs of consecutive opaque, depth-tested and depth-written draws by
   * pipeline, texture bind group and uniform window so the driver sees fewer state changes. Not exact when
   * opaque surfaces share depth values; off by default.
   */
  bool sortOpaqueDraws;

  /*
   * Print renderer diagnostics to stderr every 120 frames and account the time the game thread waits for
   * GX translation, the translation worker spends processing, the render worker spends busy and the
   * translation worker blocks on pipeline creation (aurora_render_stats_*). Off by default; the counters
   * cost a clock read per item when on.
   */
  bool renderStats;

  /*
   * Render the frame's EFB passes straight into the presented swapchain texture instead of the EFB texture,
   * skipping the full-screen present copy pass when the whole surface is covered and no overlay is
   * composited (the presented texture is acquired when the first such pass is encoded, so the swapchain
   * needs TextureBinding and copy usage). The EFB texture then no longer holds the finished scene; capture
   * tooling that reads it can set AURORA_SCENE_MIRROR=1 to have the surface copied back into it each frame.
   * Applies to full-size, non-MSAA EFB passes only; other passes render as usual. On tile-based mobile
   * drivers a render pass costs ~0.5-0.7 ms of driver time, which is what this saves.
   */
  bool sceneOnSurface;

  /*
   * Half-resolution sprite pass (requires batchDraws): when the previous frame drew at least this many GX
   * point sprites, runs of eligible point draws (alpha-tested or SRCALPHA-blended, color writes on, no
   * destination alpha) render into a half-size target with a downsampled scene depth and are composited
   * back with premultiplied alpha. Depth-writing sprites then do not occlude geometry drawn after them, so
   * this is an approximation; 0 (default) disables it. A particle-heavy stage (~25k sprites per frame) went
   * from 26.6 to 23.9 ms mean frame time on a Mali-G52 handheld with a threshold of 4000.
   */
  uint32_t halfResolutionSpritePoints;

  /*
   * Render small (<= 128x128) color-format render-to-texture passes whose EFB copy clears color (reflection
   * cameras) only every Nth frame; in between the pass is dropped and the copy texture keeps its previous
   * image. 0 or 1 (default) renders every frame. A quality trade-off for scenes that re-render the world
   * into a small reflection every frame.
   */
  uint32_t smallCopyPassInterval;
} AuroraConfig;

typedef struct {
  AuroraBackend backend;
  const char* userPath;
  const char* cachePath;
  SDL_Window* window;
  AuroraWindowSize windowSize;
} AuroraInfo;

AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config);
void aurora_shutdown();

/*
 * Renderer time accounting (AuroraConfig::renderStats; all zero when it is off). Monotonic totals in
 * nanoseconds since initialization, for the application's own frame-time breakdown:
 * - fifo_wait: game thread blocked waiting for the GX translation worker (drain, GXWaitDrawDone, sync)
 * - fifo_process: translation worker time spent processing GX commands
 * - render_worker_busy: render worker time spent executing queued items (encoding, submit, present)
 * - pipeline_wait / pipeline_wait_count: translation worker blocked on pipeline creation, and how often
 */
uint64_t aurora_render_stats_fifo_wait_ns(void);
uint64_t aurora_render_stats_fifo_process_ns(void);
uint64_t aurora_render_stats_render_worker_busy_ns(void);
uint64_t aurora_render_stats_pipeline_wait_ns(void);
uint64_t aurora_render_stats_pipeline_wait_count(void);
uint64_t aurora_render_stats_created_pipelines(void);
/*
 * Non-NULL once the GL driver probe found a driver that drops draws and switched the renderer to per-draw
 * texture-fetch barriers (correct but several times slower): the user-facing notice Aurora also draws on
 * screen. Lets the application log or report the degraded mode. NULL on other backends and healthy drivers.
 */
const char* aurora_gl_driver_notice(void);
const AuroraEvent* aurora_update();
bool aurora_begin_frame();
void aurora_end_frame();

void aurora_set_log_level(AuroraLogLevel level);
void aurora_set_pause_on_focus_lost(bool value);
void aurora_set_background_input(bool value);
void aurora_set_resampler(AuroraSampler sampler);
/** Sets the clock timescale. Default 1.0f. 0.0f is paused. Range 0.0f-16.0f. */
void aurora_set_timescale(float scale);

AuroraBackend aurora_get_backend();
const AuroraBackend* aurora_get_available_backends(size_t* count);
float aurora_get_timescale();

#ifdef __cplusplus
}
#endif

#endif
