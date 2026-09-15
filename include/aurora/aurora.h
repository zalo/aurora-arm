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
