#include "gles_direct.hpp"

#include "clear.hpp"
#include "frame_packet.hpp"
#include "pipeline_cache.hpp"
#include "profile.hpp"
#include "resource_cache.hpp"
#include "resources.hpp"
#include "../gx/gx.hpp"
#include "../gx/pipeline.hpp"
#include "../gx/resident_geometry.hpp"
#include "../gx/vertex_loader.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if AURORA_GLES_DIRECT
#include <dawn/native/OpenGLBackend.h>
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#endif

namespace aurora::gfx::gles_direct {
namespace {
constexpr Module Log{"aurora::gfx::gles_direct"};
bool sEnabled = false;
bool sMappedStreams = false;
std::vector<EncoderTaskId> sStreamIndependentTasks;
} // namespace

bool available() noexcept {
#if AURORA_GLES_DIRECT
  return true;
#else
  return false;
#endif
}

bool enabled() noexcept { return sEnabled; }
bool mapped_streams_enabled() noexcept { return sEnabled && sMappedStreams; }

namespace {
extern std::atomic_bool sSceneOnSurfaceBlocked;
extern std::atomic_bool sDriverNoticeReady;
extern std::string sDriverNotice;
} // namespace
bool scene_on_surface_allowed() noexcept { return !sSceneOnSurfaceBlocked; }
const char* driver_notice() noexcept { return sDriverNoticeReady ? sDriverNotice.c_str() : nullptr; }

void configure(wgpu::BackendType backend) {
  auto& config = g_config;
  const bool gles = backend == wgpu::BackendType::OpenGLES;
  bool direct = config.glesDirectSubmission > 0 || (config.glesDirectSubmission == 0 && gles);
  if (direct && !available()) {
    if (config.glesDirectSubmission > 0) {
      Log.warn("glesDirectSubmission requested, but this build has no OpenGL ES direct submission "
               "(AURORA_GLES_DIRECT off or Dawn without the native GL interop extension)");
    }
    direct = false;
  }
  if (direct && !gles) {
    if (config.glesDirectSubmission > 0) {
      Log.warn("glesDirectSubmission requested on a non-OpenGL ES backend; ignored");
    }
    direct = false;
  }
  if (direct && !config.cpuVertexDecode) {
    Log.warn("glesDirectSubmission requires cpuVertexDecode; direct submission disabled");
    direct = false;
  }
  sEnabled = direct;
  if (direct) {
    // The record index rides in the decoded vertices: the direct path replays batched draws only.
    config.uniformTable = true;
    config.batchDraws = true;
  }
  sMappedStreams = direct && config.glesMappedStreams >= 0;
  if (config.glesMappedStreams > 0 && !direct) {
    Log.warn("glesMappedStreams requires glesDirectSubmission; ignored");
  }
  if (direct) {
    Log.info("OpenGL ES direct submission of GX render passes enabled (mapped streams {})",
             sMappedStreams ? "on" : "off");
  }
}

void register_stream_independent_task(EncoderTaskId type) {
  if (type != InvalidEncoderTask &&
      std::find(sStreamIndependentTasks.begin(), sStreamIndependentTasks.end(), type) == sStreamIndependentTasks.end()) {
    sStreamIndependentTasks.push_back(type);
  }
}

#if AURORA_GLES_DIRECT
using namespace detail;
using dawn::native::opengl::GLInteropPipelineInfo;
using dawn::native::opengl::GLInteropTextureInfo;

namespace {
// GL_EXT_disjoint_timer_query (the GLES headers only declare the core API).
constexpr GLenum GL_TIME_ELAPSED_EXT = 0x88BF;
constexpr GLenum GL_GPU_DISJOINT_EXT = 0x8FBB;
using GetQueryObjectui64Proc = void(GL_APIENTRY*)(GLuint, GLenum, GLuint64*);
constexpr GLuint Unknown = std::numeric_limits<GLuint>::max();

bool stats_enabled() noexcept { return g_config.renderStats; }

// One recorded command of a planned pass. GX and clear draws are decoded once, at plan time.
struct PlanCommand {
  CommandType type = CommandType::DebugMarker;
  Viewport viewport{};
  ClipRect scissor{};
  gx::DrawData draw{};
  clear::DrawData clear{};
  bool isClear = false;
  // Opaque, depth-tested and depth-written with LESS/LEQUAL, no blend, no polygon offset: order independent
  // barring equal depths (sortOpaqueDraws).
  bool sortable = false;
};
struct PassPlan {
  std::string label;
  std::vector<PlanCommand> commands;
  uint32_t width = 0;
  uint32_t height = 0;
  Vec4<float> clearColor{0.f, 0.f, 0.f, 0.f};
  float clearDepth = 1.f;
  uint32_t draws = 0; // GX and clear draws
  bool eligible = false;
};
std::vector<PassPlan> sPlans;
size_t sNextPlan = 0;
size_t sProbePlan = SIZE_MAX; // plan of this frame the driver probe replays (probe_driver)
const MappedSlot* sMapped = nullptr; // this frame's mapped streams, or null
GLuint sVao = 0;
GLuint sClearProgram = 0;
GLuint sPassEbo = 0;
uint32_t sEnabledAttributes = 0;
uint64_t sFrameNumber = 0;

// Per-frame GL call counts (renderStats, [gles-direct-gl-calls]).
struct GlCallStats {
  unsigned programs = 0, pipelineStates = 0, textures = 0, texParams = 0, samplers = 0, ubos = 0, vbos = 0,
           layouts = 0, immediates = 0, draws = 0;
};
GlCallStats sGlCalls;

struct TextureBinding {
  uint32_t binding = 0;
  std::vector<uint32_t> textures;
  std::vector<uint32_t> samplers;
};
// What the direct path needs to know about one GX pipeline: its GL program, vertex layout, uniform block
// binding, immediates location and the texture units of each texture/sampler pair.
struct PreparedPipeline {
  gx::PipelineConfig config{};
  wgpu::RenderPipeline owner;
  GLInteropPipelineInfo gl{};
  gx::DecodedVertexLayout layout{};
  GLuint uniformBinding = 0;
  GLint immediates = -1;
  // Batched shaders read the matrix index and record from the vertices; only the record base (_pad) is
  // read from the immediates, so they are re-sent only when it changes.
  bool immediatesUnused = false;
  uint32_t lastImmediatePad = UINT32_MAX;
  std::array<uint32_t, 2> storageBindings{UINT32_MAX, UINT32_MAX}; // group 0 bindings 0 and 1
  std::vector<TextureBinding> textures;
};
std::unordered_map<PipelineRef, PreparedPipeline> sPrepared;

// Redundant-state memos, reset at every pass start (Dawn may change GL state between passes).
GLuint sLastProgram = Unknown, sLastUniform = Unknown, sLastUniformBinding = Unknown;
uint32_t sLastUniformOffset = Unknown;
uint32_t sLastLayout = Unknown;
PipelineRef sLastPipeline = UINTPTR_MAX;
GLuint sLastVertexBuffer = Unknown;
uint32_t sLastVertexOffset = UINT32_MAX, sLastVertexStride = UINT32_MAX;
GLuint sBindingDivisor = Unknown; // binding 0 step: 1 for instanced point sprites
GLuint sActiveUnit = Unknown;
std::array<uint32_t, 2> sLastStorage{Unknown, Unknown};

struct ResidentArenaGL {
  WGPUBuffer buffer = nullptr;
  GLuint name = 0;
};
std::vector<ResidentArenaGL> sResidentGL;
GLuint resident_gl_buffer(uint32_t arena) {
  if (sResidentGL.size() < arena) {
    sResidentGL.resize(arena);
  }
  auto& entry = sResidentGL[arena - 1];
  uint64_t size = 0;
  const auto& buffer = gx::resident::arena_buffer(arena - 1, size);
  // The arena buffer is replaced when it grows; look the GL name up again then.
  if (entry.buffer != buffer.Get() || entry.name == 0) {
    entry.buffer = buffer.Get();
    entry.name = buffer ? dawn::native::opengl::GetGLInteropBuffer(buffer.Get()) : 0;
  }
  return entry.name;
}

struct TextureUnitState {
  bool valid = false;
  GLInteropTextureInfo texture{};
  GLuint sampler = Unknown;
};
std::array<TextureUnitState, 32> sTextureState;

struct PassTimer {
  using Clock = std::chrono::steady_clock;
  Clock::time_point start;
  GLuint query = 0;
  uint64_t frame = sFrameNumber;
  struct Pending {
    GLuint query;
    uint64_t frame;
  };
  static inline std::vector<Pending> pending;
  static inline GetQueryObjectui64Proc get64 = nullptr;
  static inline bool initialized = false;
  PassTimer() {
    if (!stats_enabled()) {
      return;
    }
    if (!initialized) {
      const char* extensions = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
      if (extensions != nullptr && std::strstr(extensions, "GL_EXT_disjoint_timer_query") != nullptr) {
        get64 = reinterpret_cast<GetQueryObjectui64Proc>(eglGetProcAddress("glGetQueryObjectui64vEXT"));
      }
      initialized = true;
    }
    if (get64 != nullptr) {
      GLint disjoint = 0;
      glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
      if (disjoint) {
        for (auto& result : pending) {
          glDeleteQueries(1, &result.query);
        }
        pending.clear();
      }
      for (auto it = pending.begin(); it != pending.end();) {
        GLuint ready = 0;
        glGetQueryObjectuiv(it->query, GL_QUERY_RESULT_AVAILABLE, &ready);
        if (!ready) {
          ++it;
          continue;
        }
        GLuint64 ns = 0;
        get64(it->query, GL_QUERY_RESULT, &ns);
        if (!disjoint) {
          std::fprintf(stderr, "[gles-direct-gpu] frame=%llu gpu_ms=%.3f\n", static_cast<unsigned long long>(it->frame),
                       static_cast<double>(ns) / 1e6);
        }
        glDeleteQueries(1, &it->query);
        it = pending.erase(it);
      }
      // Sampled asynchronously; the submit path never waits for a GPU result.
      if (frame % 60 == 0 && pending.size() < 8) {
        glGenQueries(1, &query);
        glBeginQuery(GL_TIME_ELAPSED_EXT, query);
      }
    }
    start = Clock::now();
  }
  ~PassTimer() {
    if (!stats_enabled()) {
      return;
    }
    if (query != 0) {
      glEndQuery(GL_TIME_ELAPSED_EXT);
      pending.push_back({query, frame});
    }
    if (frame % 60 == 0) {
      std::fprintf(stderr, "[gles-direct-cpu] frame=%llu wall_ms=%.3f\n", static_cast<unsigned long long>(frame),
                   std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
  }
};

// Per-draw error checks only with renderStats; every pass is checked at its end.
bool gl_ok(const char* section, uint32_t passIndex, uint32_t drawIndex, bool always = false) {
  if (!always && !stats_enabled()) {
    return true;
  }
  const GLenum error = glGetError();
  if (error == GL_NO_ERROR) {
    return true;
  }
  std::fprintf(stderr, "[gles-direct-error] frame=%llu pass=%u draw=%u section=%s error=0x%x\n",
               static_cast<unsigned long long>(sFrameNumber), passIndex, drawIndex, section, error);
  while (glGetError() != GL_NO_ERROR) {
  }
  return false;
}

GLuint compile_shader(GLenum type, const char* source, const char* label) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (ok == GL_TRUE) {
    return shader;
  }
  char log[2048]{};
  GLsizei length = 0;
  glGetShaderInfoLog(shader, sizeof(log), &length, log);
  Log.error("{} shader compile failed: {}", label, std::string_view{log, static_cast<size_t>(length)});
  glDeleteShader(shader);
  return 0;
}

// Full-target clear draw (clear.cpp) as a native program: one triangle, color through the blend constant.
GLuint clear_program() {
  if (sClearProgram != 0) {
    return sClearProgram;
  }
  constexpr const char* vertex = R"GLSL(#version 310 es
precision highp float;
const vec2 positions[3] = vec2[3](vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));
void main() { gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0); }
)GLSL";
  constexpr const char* fragment = R"GLSL(#version 310 es
precision highp float;
layout(location = 0) out vec4 color;
void main() { color = vec4(1.0); }
)GLSL";
  const GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex, "clear vertex");
  const GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment, "clear fragment");
  if (vs == 0 || fs == 0) {
    if (vs != 0) {
      glDeleteShader(vs);
    }
    if (fs != 0) {
      glDeleteShader(fs);
    }
    return 0;
  }
  const GLuint program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  glDeleteShader(vs);
  glDeleteShader(fs);
  GLint ok = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (ok != GL_TRUE) {
    char log[2048]{};
    GLsizei length = 0;
    glGetProgramInfoLog(program, sizeof(log), &length, log);
    Log.error("clear program link failed: {}", std::string_view{log, static_cast<size_t>(length)});
    glDeleteProgram(program);
    return 0;
  }
  sClearProgram = program;
  return program;
}

GLenum compare_func(GXCompare f) {
  switch (f) {
  case GX_NEVER:
    return GL_NEVER;
  case GX_LESS:
    return gx::UseReversedZ ? GL_GREATER : GL_LESS;
  case GX_EQUAL:
    return GL_EQUAL;
  case GX_LEQUAL:
    return gx::UseReversedZ ? GL_GEQUAL : GL_LEQUAL;
  case GX_GREATER:
    return gx::UseReversedZ ? GL_LESS : GL_GREATER;
  case GX_NEQUAL:
    return GL_NOTEQUAL;
  case GX_GEQUAL:
    return gx::UseReversedZ ? GL_LEQUAL : GL_GEQUAL;
  case GX_ALWAYS:
  default:
    return GL_ALWAYS;
  }
}

GLenum blend_factor(GXBlendFactor f, bool destination) {
  switch (f) {
  case GX_BL_ZERO:
    return GL_ZERO;
  case GX_BL_ONE:
    return GL_ONE;
  case GX_BL_SRCCLR:
    return destination ? GL_SRC_COLOR : GL_DST_COLOR;
  case GX_BL_INVSRCCLR:
    return destination ? GL_ONE_MINUS_SRC_COLOR : GL_ONE_MINUS_DST_COLOR;
  case GX_BL_SRCALPHA:
    return GL_SRC_ALPHA;
  case GX_BL_INVSRCALPHA:
    return GL_ONE_MINUS_SRC_ALPHA;
  case GX_BL_DSTALPHA:
    return GL_DST_ALPHA;
  case GX_BL_INVDSTALPHA:
    return GL_ONE_MINUS_DST_ALPHA;
  default:
    return GL_ONE;
  }
}

// Redundant-state filter. Consecutive GX pipelines usually differ only in their TEV program, so most
// fixed-function state calls can be skipped; mobile drivers validate every state call.
struct GLStateCache {
  bool valid = false;
  bool cull = false;
  GLenum cullFace = 0;
  bool depth = false;
  GLenum depthFunc = 0;
  GLboolean depthMask = GL_FALSE;
  GLboolean colorMask[4] = {GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE};
  bool blend = false;
  GLenum equation = 0, src = 0, dst = 0;
  bool separate = false;
  bool accumulate = false;
  float constantAlpha = -1.f;
  bool polygonOffset = false;
  float slope = 0, bias = 0;
};
GLStateCache sGLState;

// Mirrors gx::build_pipeline's fixed-function state (gx.cpp) in GL terms.
void apply_pipeline_state(const gx::PipelineConfig& c) {
  auto& st = sGLState;
  const bool fresh = !st.valid;
  // GX pipelines use WebGPU CW winding; Tint inverts clip-space Y for GLES, which reverses the winding.
  if (fresh) {
    glFrontFace(GL_CCW);
  }
  const bool cull = c.cullMode != GX_CULL_NONE;
  const GLenum cullFace = c.cullMode == GX_CULL_FRONT ? GL_FRONT : GL_BACK;
  if (fresh || st.cull != cull) {
    if (cull) {
      glEnable(GL_CULL_FACE);
    } else {
      glDisable(GL_CULL_FACE);
    }
  }
  if (cull && (fresh || !st.cull || st.cullFace != cullFace)) {
    glCullFace(cullFace);
  }
  const bool depth = c.depthCompare;
  const GLenum depthFunc = compare_func(c.depthFunc);
  if (fresh || st.depth != depth) {
    if (depth) {
      glEnable(GL_DEPTH_TEST);
    } else {
      glDisable(GL_DEPTH_TEST);
    }
  }
  if (depth && (fresh || !st.depth || st.depthFunc != depthFunc)) {
    glDepthFunc(depthFunc);
  }
  const GLboolean depthMask = c.depthCompare && c.depthUpdate ? GL_TRUE : GL_FALSE;
  if (fresh || st.depthMask != depthMask) {
    glDepthMask(depthMask);
  }
  // Half-resolution sprite accumulation (sprite_pass.hpp): rgb blends as the game asked, alpha accumulates
  // coverage and is always written.
  const bool accumulate = c.shaderConfig.spriteAccumulate;
  const GLboolean colorMask[4] = {static_cast<GLboolean>(c.colorUpdate), static_cast<GLboolean>(c.colorUpdate),
                                  static_cast<GLboolean>(c.colorUpdate),
                                  static_cast<GLboolean>(accumulate || c.alphaUpdate)};
  if (fresh || std::memcmp(st.colorMask, colorMask, sizeof(colorMask)) != 0) {
    glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
  }
  GLenum src = GL_ONE, dst = GL_ZERO, equation = GL_FUNC_ADD;
  if (c.blendMode == GX_BM_BLEND) {
    src = blend_factor(c.blendFacSrc, false);
    dst = blend_factor(c.blendFacDst, true);
  } else if (c.blendMode == GX_BM_SUBTRACT) {
    src = dst = GL_ONE;
    equation = GL_FUNC_REVERSE_SUBTRACT;
  } else if (c.blendMode == GX_BM_LOGIC) {
    if (c.blendOp == GX_LO_CLEAR) {
      src = dst = GL_ZERO;
    } else if (c.blendOp == GX_LO_NOOP) {
      src = GL_ZERO;
      dst = GL_ONE;
    }
  }
  const bool blend = c.blendMode != GX_BM_NONE || c.dstAlpha != UINT32_MAX || accumulate;
  if (accumulate && c.blendMode == GX_BM_NONE) {
    // Opaque (alpha-tested) sprites store premultiplied color and coverage.
    src = GL_SRC_ALPHA;
    dst = GL_ZERO;
  }
  if (fresh || st.blend != blend) {
    if (blend) {
      glEnable(GL_BLEND);
    } else {
      glDisable(GL_BLEND);
    }
  }
  if (blend) {
    if (fresh || !st.blend || st.equation != equation) {
      glBlendEquationSeparate(equation, equation);
    }
    if (c.dstAlpha != UINT32_MAX) {
      const float constantAlpha = static_cast<float>(c.dstAlpha) / 255.f;
      if (fresh || st.constantAlpha != constantAlpha) {
        glBlendColor(0, 0, 0, constantAlpha);
      }
      if (fresh || !st.blend || !st.separate || st.accumulate || st.src != src || st.dst != dst) {
        glBlendFuncSeparate(src, dst, GL_CONSTANT_ALPHA, GL_ZERO);
      }
      st.separate = true;
      st.accumulate = false;
      st.constantAlpha = constantAlpha;
    } else if (accumulate) {
      const bool opaque = c.blendMode == GX_BM_NONE;
      const bool attenuates = dst == GL_ONE_MINUS_SRC_ALPHA;
      if (fresh || !st.blend || !st.separate || !st.accumulate || st.src != src || st.dst != dst) {
        glBlendFuncSeparate(src, dst, opaque || attenuates ? GL_ONE : GL_ZERO,
                            opaque ? GL_ZERO : attenuates ? GL_ONE_MINUS_SRC_ALPHA : GL_ONE);
      }
      st.separate = true;
      st.accumulate = true;
    } else {
      if (fresh || !st.blend || st.separate || st.src != src || st.dst != dst) {
        glBlendFunc(src, dst);
      }
      st.separate = false;
      st.accumulate = false;
    }
    st.equation = equation;
    st.src = src;
    st.dst = dst;
  }
  const float bias = std::round((gx::UseReversedZ ? -1.f : 1.f) * std::bit_cast<float>(c.polygonOffsetBits));
  const float slope = (gx::UseReversedZ ? -1.f : 1.f) * std::bit_cast<float>(c.polygonOffsetScaleBits);
  const bool polygonOffset = bias != 0.f || slope != 0.f;
  if (fresh || st.polygonOffset != polygonOffset) {
    if (polygonOffset) {
      glEnable(GL_POLYGON_OFFSET_FILL);
    } else {
      glDisable(GL_POLYGON_OFFSET_FILL);
    }
  }
  if (polygonOffset && (fresh || !st.polygonOffset || st.slope != slope || st.bias != bias)) {
    glPolygonOffset(slope, bias);
  }
  st.valid = true;
  st.cull = cull;
  st.cullFace = cullFace;
  st.depth = depth;
  st.depthFunc = depthFunc;
  st.depthMask = depthMask;
  std::memcpy(st.colorMask, colorMask, sizeof(colorMask));
  st.blend = blend;
  st.polygonOffset = polygonOffset;
  st.slope = slope;
  st.bias = bias;
}

// Texture parameters (mip range, swizzle) are texture object state; within a direct pass nothing else
// touches these objects, so they are set once per object. Sampler state is folded into the texture object
// too (a GX texture almost always uses one sampler), so draws do not rebind sampler objects.
struct SamplerParams {
  GLint wrapS, wrapT, minFilter, magFilter;
  GLfloat minLod, maxLod;
  GLint compareMode, compareFunc;
  GLfloat anisotropy;
};
const SamplerParams& sampler_params(GLuint sampler) {
  static std::unordered_map<GLuint, SamplerParams> cache;
  const auto it = cache.find(sampler);
  if (it != cache.end()) {
    return it->second;
  }
  SamplerParams p{};
  glGetSamplerParameteriv(sampler, GL_TEXTURE_WRAP_S, &p.wrapS);
  glGetSamplerParameteriv(sampler, GL_TEXTURE_WRAP_T, &p.wrapT);
  glGetSamplerParameteriv(sampler, GL_TEXTURE_MIN_FILTER, &p.minFilter);
  glGetSamplerParameteriv(sampler, GL_TEXTURE_MAG_FILTER, &p.magFilter);
  glGetSamplerParameterfv(sampler, GL_TEXTURE_MIN_LOD, &p.minLod);
  glGetSamplerParameterfv(sampler, GL_TEXTURE_MAX_LOD, &p.maxLod);
  glGetSamplerParameteriv(sampler, GL_TEXTURE_COMPARE_MODE, &p.compareMode);
  glGetSamplerParameteriv(sampler, GL_TEXTURE_COMPARE_FUNC, &p.compareFunc);
  p.anisotropy = 1.f;
  glGetSamplerParameterfv(sampler, 0x84FE /* GL_TEXTURE_MAX_ANISOTROPY_EXT */, &p.anisotropy);
  while (glGetError() != GL_NO_ERROR) {
  }
  return cache.emplace(sampler, p).first->second;
}
std::unordered_map<GLuint, GLuint> sTextureSamplers; // texture object -> sampler whose state it carries
void apply_sampler_to_texture(GLenum target, GLuint texture, GLuint sampler) {
  auto& carried = sTextureSamplers[texture];
  if (carried == sampler) {
    return;
  }
  const auto& p = sampler_params(sampler);
  ++sGlCalls.texParams;
  glTexParameteri(target, GL_TEXTURE_WRAP_S, p.wrapS);
  glTexParameteri(target, GL_TEXTURE_WRAP_T, p.wrapT);
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, p.minFilter);
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, p.magFilter);
  glTexParameterf(target, GL_TEXTURE_MIN_LOD, p.minLod);
  glTexParameterf(target, GL_TEXTURE_MAX_LOD, p.maxLod);
  glTexParameteri(target, GL_TEXTURE_COMPARE_MODE, p.compareMode);
  if (p.compareMode != GL_NONE) {
    glTexParameteri(target, GL_TEXTURE_COMPARE_FUNC, p.compareFunc);
  }
  carried = sampler;
}
struct TextureParams {
  GLint baseMipLevel, maxMipLevel;
  GLenum swizzle[4];
};
std::unordered_map<GLuint, TextureParams> sTextureParams;

void bind_texture_unit(const GLInteropTextureInfo& texture, GLuint unit) {
  if (unit < sTextureState.size()) {
    auto& state = sTextureState[unit];
    const auto& last = state.texture;
    if (state.valid && last.texture == texture.texture && last.target == texture.target &&
        last.baseMipLevel == texture.baseMipLevel && last.maxMipLevel == texture.maxMipLevel &&
        std::equal(std::begin(last.swizzle), std::end(last.swizzle), std::begin(texture.swizzle))) {
      return;
    }
    // Mip limits and swizzles belong to the texture object, not the unit: a different view of an aliased
    // object invalidates the other units' memos.
    for (auto& other : sTextureState) {
      if (&other != &state && other.valid && other.texture.texture == texture.texture) {
        other.valid = false;
      }
    }
    state.valid = true;
    state.texture = texture;
  }
  if (sActiveUnit != unit) {
    glActiveTexture(GL_TEXTURE0 + unit);
    sActiveUnit = unit;
  }
  glBindTexture(texture.target, texture.texture);
  ++sGlCalls.textures;
  const TextureParams params{static_cast<GLint>(texture.baseMipLevel), static_cast<GLint>(texture.maxMipLevel),
                             {texture.swizzle[0], texture.swizzle[1], texture.swizzle[2], texture.swizzle[3]}};
  const auto [it, inserted] = sTextureParams.try_emplace(texture.texture, params);
  if (!inserted) {
    if (std::memcmp(&it->second, &params, sizeof(params)) == 0) {
      return;
    }
    it->second = params;
  }
  ++sGlCalls.texParams;
  glTexParameteri(texture.target, GL_TEXTURE_BASE_LEVEL, static_cast<GLint>(texture.baseMipLevel));
  glTexParameteri(texture.target, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(texture.maxMipLevel));
  if (texture.swizzle[0] != GL_NONE) {
    glTexParameteri(texture.target, GL_TEXTURE_SWIZZLE_R, static_cast<GLint>(texture.swizzle[0]));
    glTexParameteri(texture.target, GL_TEXTURE_SWIZZLE_G, static_cast<GLint>(texture.swizzle[1]));
    glTexParameteri(texture.target, GL_TEXTURE_SWIZZLE_B, static_cast<GLint>(texture.swizzle[2]));
    glTexParameteri(texture.target, GL_TEXTURE_SWIZZLE_A, static_cast<GLint>(texture.swizzle[3]));
  }
}

// GL names behind one (texture bind group, pipeline) pair. Bind groups are immutable; entries are dropped
// when the bind group cache evicts anything (the generation changes).
struct ResolvedTextures {
  uint64_t generation = 0;
  struct Entry {
    GLInteropTextureInfo texture;
    GLuint sampler;
  };
  std::vector<Entry> entries; // one per PreparedPipeline::textures element
};
std::unordered_map<uint64_t, ResolvedTextures> sResolvedTextures;
const ResolvedTextures& resolve_textures(BindGroupRef group, PipelineRef pipeline, const PreparedPipeline& p) {
  using namespace dawn::native::opengl;
  const uint64_t generation = bind_group_cache_generation();
  const uint64_t key = group * 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(pipeline);
  auto it = sResolvedTextures.find(key);
  if (it != sResolvedTextures.end() && it->second.generation == generation) {
    return it->second;
  }
  if (it == sResolvedTextures.end()) {
    it = sResolvedTextures.emplace(key, ResolvedTextures{}).first;
  }
  auto& resolved = it->second;
  resolved.generation = generation;
  resolved.entries.clear();
  const auto bindGroup = find_bind_group(group);
  for (const auto& t : p.textures) {
    resolved.entries.push_back({GetGLInteropBindGroupTexture(bindGroup.Get(), t.binding),
                                GetGLInteropBindGroupSampler(bindGroup.Get(), t.binding + 1)});
  }
  return resolved;
}

// Pipelines the direct path can replay: batched (record index in the vertices), no fog range table (read
// through the storage buffer immediates) and no depth bias clamp (Dawn's GL backend emulates it through an
// internal immediate whose layout the direct path does not mirror).
bool pipeline_eligible(const gx::PipelineConfig& c) {
  return c.shaderConfig.cpuVertexDecode && c.shaderConfig.batchDraws && !c.shaderConfig.fogRangeEnabled &&
         std::bit_cast<float>(c.polygonOffsetClampBits) == 0.f;
}

PreparedPipeline* prepare_pipeline(PipelineRef ref) {
  using namespace dawn::native::opengl;
  if (const auto it = sPrepared.find(ref); it != sPrepared.end()) {
    return &it->second;
  }
  PreparedPipeline p;
  if (!gx::find_pipeline_config(ref, p.config) || !get_pipeline(ref, p.owner)) {
    return nullptr;
  }
  p.gl = GetGLInteropRenderPipeline(p.owner.Get());
  p.uniformBinding = GetGLInteropBufferBinding(p.owner.Get(), 1, 0);
  p.storageBindings = {GetGLInteropBufferBinding(p.owner.Get(), 0, 0), GetGLInteropBufferBinding(p.owner.Get(), 0, 1)};
  p.immediates = glGetUniformLocation(p.gl.program, "tint_immediates");
  p.immediatesUnused = pipeline_eligible(p.config);
  p.layout = gx::decoded_vertex_layout(p.config.shaderConfig);
  const auto units = [&](uint32_t group, uint32_t binding, bool sampler) {
    uint32_t values[32];
    const auto count = GetGLInteropTextureUnits(p.owner.Get(), group, binding, sampler, values, 32);
    return std::vector<uint32_t>(values, values + std::min(count, 32u));
  };
  for (uint32_t binding = 0; binding < gx::MaxTextures * 2; binding += 2) {
    TextureBinding t{.binding = binding, .textures = units(2, binding, false), .samplers = units(2, binding + 1, true)};
    if (!t.textures.empty()) {
      p.textures.push_back(std::move(t));
    }
  }
  return &sPrepared.emplace(ref, std::move(p)).first->second;
}

void vertex_attrib_format(const wgpu::VertexAttribute& attr) {
  const auto loc = static_cast<GLuint>(attr.shaderLocation);
  const auto offset = static_cast<GLuint>(attr.offset);
  switch (attr.format) {
  case wgpu::VertexFormat::Uint32x3:
    glVertexAttribIFormat(loc, 3, GL_UNSIGNED_INT, offset);
    break;
  case wgpu::VertexFormat::Uint32x2:
    glVertexAttribIFormat(loc, 2, GL_UNSIGNED_INT, offset);
    break;
  case wgpu::VertexFormat::Uint32:
    glVertexAttribIFormat(loc, 1, GL_UNSIGNED_INT, offset);
    break;
  case wgpu::VertexFormat::Float32x2:
    glVertexAttribFormat(loc, 2, GL_FLOAT, GL_FALSE, offset);
    break;
  case wgpu::VertexFormat::Float32x3:
    glVertexAttribFormat(loc, 3, GL_FLOAT, GL_FALSE, offset);
    break;
  case wgpu::VertexFormat::Float32x4:
    glVertexAttribFormat(loc, 4, GL_FLOAT, GL_FALSE, offset);
    break;
  default:
    Log.fatal("unsupported decoded vertex attribute format {}", static_cast<uint32_t>(attr.format));
  }
}

// AURORA_GLES_VERTEX_API=pointer sets attributes with glVertexAttribPointer/Divisor like Dawn's GL backend
// instead of the ES 3.1 vertex binding API (diagnostic for drivers that mishandle binding offsets).
bool vertex_attrib_pointers() {
  static const bool pointers = [] {
    const char* value = std::getenv("AURORA_GLES_VERTEX_API");
    return value != nullptr && std::strcmp(value, "pointer") == 0;
  }();
  return pointers;
}

void vertex_attrib_pointer(const wgpu::VertexAttribute& attr, uint32_t base, uint32_t stride, GLuint divisor) {
  const auto loc = static_cast<GLuint>(attr.shaderLocation);
  const auto pointer = reinterpret_cast<const void*>(static_cast<uintptr_t>(base + attr.offset));
  const auto s = static_cast<GLsizei>(stride);
  switch (attr.format) {
  case wgpu::VertexFormat::Uint32x3:
    glVertexAttribIPointer(loc, 3, GL_UNSIGNED_INT, s, pointer);
    break;
  case wgpu::VertexFormat::Uint32x2:
    glVertexAttribIPointer(loc, 2, GL_UNSIGNED_INT, s, pointer);
    break;
  case wgpu::VertexFormat::Uint32:
    glVertexAttribIPointer(loc, 1, GL_UNSIGNED_INT, s, pointer);
    break;
  case wgpu::VertexFormat::Float32x2:
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, s, pointer);
    break;
  case wgpu::VertexFormat::Float32x3:
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, s, pointer);
    break;
  case wgpu::VertexFormat::Float32x4:
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, s, pointer);
    break;
  default:
    Log.fatal("unsupported decoded vertex attribute format {}", static_cast<uint32_t>(attr.format));
  }
  glVertexAttribDivisor(loc, divisor);
  glEnableVertexAttribArray(loc);
}

// AURORA_GLES_TEXTURE_SAMPLERS=0 binds Dawn's sampler objects instead of carrying sampler state on the
// texture object (diagnostic for drivers that apply texture parameters lazily across a job).
bool carry_samplers_on_textures() {
  static const bool carry = [] {
    const char* value = std::getenv("AURORA_GLES_TEXTURE_SAMPLERS");
    return value == nullptr || value[0] != '0';
  }();
  return carry;
}

void bind_draw_resources(const gx::DrawData& d, PreparedPipeline& p) {
  using namespace dawn::native::opengl;
  const auto& res = resources();
  // Uniform window: the record's 64 KiB window of the mapped slot or of Dawn's uniform buffer.
  {
    const GLuint buffer = sMapped != nullptr ? sMapped->uniforms : GetGLInteropBuffer(res.uniformBuffer.Get());
    const uint32_t offset = gx::uniform_window_index(d.uniformRange.offset) * gx::UniformWindowSize;
    if (sLastUniform != buffer || sLastUniformBinding != p.uniformBinding || sLastUniformOffset != offset) {
      glBindBufferRange(GL_UNIFORM_BUFFER, p.uniformBinding, buffer, offset, gx::UniformWindowSize);
      ++sGlCalls.ubos;
      sLastUniform = buffer;
      sLastUniformBinding = p.uniformBinding;
      sLastUniformOffset = offset;
    }
  }
  // Storage blocks the program still declares (fetch buffers; unused by decoded-vertex shaders without a
  // fog range table, but bound so the driver never sees an incomplete binding).
  for (size_t i = 0; i < p.storageBindings.size(); ++i) {
    const uint32_t binding = p.storageBindings[i];
    if (binding == UINT32_MAX || sLastStorage[i] == binding) {
      continue;
    }
    const auto& buffer = i == 0 ? res.vertexBuffer : res.storageBuffer;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, GetGLInteropBuffer(buffer.Get()));
    sLastStorage[i] = binding;
  }
  if (d.bindGroups.textureBindGroup != 0) {
    const auto& resolved = resolve_textures(d.bindGroups.textureBindGroup, d.pipeline, p);
    for (size_t i = 0; i < p.textures.size(); ++i) {
      const auto& t = p.textures[i];
      const auto& texture = resolved.entries[i].texture;
      const GLuint sampler = resolved.entries[i].sampler;
      for (auto unit : t.textures) {
        bind_texture_unit(texture, unit);
      }
      if (texture.texture != 0 && carry_samplers_on_textures()) {
        // The texture object carries the sampler state; the unit samplers stay 0.
        for (auto unit : t.textures) {
          if (sActiveUnit != unit) {
            glActiveTexture(GL_TEXTURE0 + unit);
            sActiveUnit = unit;
          }
          apply_sampler_to_texture(texture.target, texture.texture, sampler);
        }
        for (auto unit : t.samplers) {
          if (unit >= sTextureState.size() || sTextureState[unit].sampler != 0) {
            glBindSampler(unit, 0);
            ++sGlCalls.samplers;
            if (unit < sTextureState.size()) {
              sTextureState[unit].sampler = 0;
            }
          }
        }
      } else {
        for (auto unit : t.samplers) {
          if (unit >= sTextureState.size() || sTextureState[unit].sampler != sampler) {
            glBindSampler(unit, sampler);
            ++sGlCalls.samplers;
            if (unit < sTextureState.size()) {
              sTextureState[unit].sampler = sampler;
            }
          }
        }
      }
    }
  }
  // Vertex stream: the frame's decoded records (mapped slot or Dawn's vertex buffer) or a resident arena.
  {
    const auto& layout = p.layout;
    GLuint vertex;
    uint32_t vertexOffset;
    if (d.residentArena != 0) {
      vertex = resident_gl_buffer(d.residentArena);
      vertexOffset = 0;
    } else {
      vertex = sMapped != nullptr ? sMapped->vertices : GetGLInteropBuffer(res.vertexBuffer.Get());
      vertexOffset = d.vertRange.offset;
    }
    uint32_t used = 0;
    for (unsigned a = 0; a < layout.count; ++a) {
      used |= 1u << layout.attributes[a].shaderLocation;
    }
    const GLuint divisor = p.config.shaderConfig.lineMode == 3 ? 1u : 0u;
    if (vertex_attrib_pointers()) {
      if (sLastVertexBuffer != vertex || sLastVertexOffset != vertexOffset || sLastVertexStride != layout.stride ||
          sLastLayout != used || sBindingDivisor != divisor) {
        glBindBuffer(GL_ARRAY_BUFFER, vertex);
        for (unsigned a = 0; a < layout.count; ++a) {
          vertex_attrib_pointer(layout.attributes[a], vertexOffset, layout.stride, divisor);
        }
        for (unsigned loc = 0; loc < gx::MaxDecodedVertexAttrs; ++loc) {
          if ((sEnabledAttributes & (1u << loc)) != 0 && (used & (1u << loc)) == 0) {
            glDisableVertexAttribArray(loc);
          }
        }
        sEnabledAttributes = used;
        sLastVertexBuffer = vertex;
        sLastVertexOffset = vertexOffset;
        sLastVertexStride = layout.stride;
        sLastLayout = used;
        sBindingDivisor = divisor;
        ++sGlCalls.vbos;
      }
    } else {
    if (sLastVertexBuffer != vertex || sLastVertexOffset != vertexOffset || sLastVertexStride != layout.stride) {
      glBindVertexBuffer(0, vertex, vertexOffset, static_cast<GLsizei>(layout.stride));
      ++sGlCalls.vbos;
      sLastVertexBuffer = vertex;
      sLastVertexOffset = vertexOffset;
      sLastVertexStride = layout.stride;
    }
    if (sLastLayout != used) {
      for (unsigned a = 0; a < layout.count; ++a) {
        const auto& attr = layout.attributes[a];
        vertex_attrib_format(attr);
        glVertexAttribBinding(static_cast<GLuint>(attr.shaderLocation), 0);
        glEnableVertexAttribArray(static_cast<GLuint>(attr.shaderLocation));
      }
      for (unsigned loc = 0; loc < gx::MaxDecodedVertexAttrs; ++loc) {
        if ((sEnabledAttributes & (1u << loc)) != 0 && (used & (1u << loc)) == 0) {
          glDisableVertexAttribArray(loc);
        }
      }
      sEnabledAttributes = used;
      sLastLayout = used;
      ++sGlCalls.layouts;
    }
    // Instanced point sprites step binding 0 per instance (one record per point).
    if (sBindingDivisor != divisor) {
      glVertexBindingDivisor(0, divisor);
      sBindingDivisor = divisor;
    }
    }
  }
  if (p.immediates >= 0) {
    // Batched shaders read only _pad (record base): streamed draws pass 0, resident draws their record.
    const uint32_t pad = d.residentArena != 0 ? gx::uniform_record_index(d.uniformRange.offset) : 0;
    if (!p.immediatesUnused || p.lastImmediatePad != pad) {
      auto immediates = d.immediateData;
      immediates._pad = pad;
      glUniform1uiv(p.immediates, sizeof(immediates) / sizeof(uint32_t), reinterpret_cast<const GLuint*>(&immediates));
      ++sGlCalls.immediates;
      p.lastImmediatePad = pad;
    }
  }
}

// Some Mali-G52 drivers (libmali g13p0 and g24p0 on a g18p0 kernel) lose geometry (stale 128px tiles, missing
// panels) unless a texture-fetch barrier follows each draw. The driver probe (probe_driver) detects that at
// startup and turns the per-draw barrier on. AURORA_GLES_DRAW_BARRIER overrides the probe: 0 = off, N = after
// every Nth draw, pass = once at the end of each render pass.
struct DrawBarrierPolicy {
  uint32_t every = 0;
  bool pass = false;
  bool finish = false; // AURORA_GLES_DRAW_BARRIER=finish: glFinish at the end of each render pass
  bool flush = false;  // AURORA_GLES_DRAW_BARRIER=flush: glFlush at the end of each render pass
  // Startup driver probes left (AURORA_GLES_DRIVER_PROBE=N, 0 = none); none with an explicit barrier override.
  uint32_t probes = 0;
};
// Forces a barrier after every draw while the driver probe renders its reference image.
bool sProbeBarrier = false;
// Mali r13p0 (Anbernic RG351P, AmberELEC) can hang glCopyImageSubData forever when the scene renders into a
// presented surface texture that the present worker's shared context samples (the Nintendo logo froze in about
// half of runs, with or without flushes on either context). Detected with the barrier policy on the first direct
// draw; later passes render into the EFB texture instead. AURORA_GLES_SCENE_ON_SURFACE=1 keeps it on.
std::atomic_bool sSceneOnSurfaceBlocked = false;

bool sBarrierPolicyResolved = false;
bool draw_barrier_policy_resolved() noexcept { return sBarrierPolicyResolved; }

// Resolved on the GL thread (it reads GL_VERSION).
DrawBarrierPolicy& draw_barrier_policy() {
  static DrawBarrierPolicy policy;
  if (!sBarrierPolicyResolved) {
    sBarrierPolicyResolved = true;
    const char* override = std::getenv("AURORA_GLES_DRAW_BARRIER");
    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    const char* surfaceOverride = std::getenv("AURORA_GLES_SCENE_ON_SURFACE");
    if (surfaceOverride != nullptr ? std::strcmp(surfaceOverride, "0") == 0
                                   : version != nullptr && std::strstr(version, "r13p0") != nullptr) {
      sSceneOnSurfaceBlocked = true;
      Log.info("Scene on surface disabled (GL_VERSION {})", version != nullptr ? version : "unknown");
    }
    if (override != nullptr && std::strcmp(override, "pass") == 0) {
      policy.pass = true;
    } else if (override != nullptr && std::strcmp(override, "finish") == 0) {
      policy.finish = true;
    } else if (override != nullptr && std::strcmp(override, "flush") == 0) {
      policy.flush = true;
    } else if (override != nullptr && *override != '\0') {
      policy.every = static_cast<uint32_t>(std::strtoul(override, nullptr, 10));
    } else {
      const char* probes = std::getenv("AURORA_GLES_DRIVER_PROBE");
      policy.probes = probes != nullptr ? static_cast<uint32_t>(std::strtoul(probes, nullptr, 10)) : 12u;
    }
    Log.info("Texture-fetch barrier: every {} draws, per pass {}, driver probes {} (GL_VERSION {})", policy.every,
             policy.pass, policy.probes, version != nullptr ? version : "unknown");
  }
  return policy;
}

// Bisection scope for the every-N barrier: AURORA_GLES_BARRIER_PASSES=i,j limits it to those plan indices of
// a frame, AURORA_GLES_BARRIER_FIRST=N to the first N draws of each pass.
uint32_t sBarrierPlan = 0;
uint32_t sBarrierDrawInPass = 0;
bool barrier_in_scope() {
  static const uint64_t passes = []() -> uint64_t {
    const char* value = std::getenv("AURORA_GLES_BARRIER_PASSES");
    if (value == nullptr || *value == '\0') {
      return ~0ull;
    }
    uint64_t mask = 0;
    for (char* end = nullptr;; value = end + 1) {
      const auto index = std::strtoul(value, &end, 10);
      if (end == value) {
        break;
      }
      if (index < 64) {
        mask |= 1ull << index;
      }
      if (*end != ',') {
        break;
      }
    }
    return mask;
  }();
  static const uint32_t first = [] {
    const char* value = std::getenv("AURORA_GLES_BARRIER_FIRST");
    return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : UINT32_MAX;
  }();
  return sBarrierPlan < 64 && (passes >> sBarrierPlan & 1) != 0 && sBarrierDrawInPass < first;
}

// AURORA_GLES_BARRIER_PRIMS=N issues the barrier once the draws since the last one reach N index/vertex
// elements (diagnostic: whether the driver's loss tracks the primitives in a job rather than its draw count).
uint32_t barrier_element_budget() {
  static const uint32_t budget = [] {
    const char* value = std::getenv("AURORA_GLES_BARRIER_PRIMS");
    return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
  }();
  return budget;
}

void draw_barrier(uint32_t elements) {
  static uint32_t draws = 0;
  static uint64_t pending = 0;
  if (const uint32_t budget = barrier_element_budget(); budget != 0) {
    pending += elements;
    if (pending >= budget) {
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
      pending = 0;
    }
    return;
  }
  const auto& policy = draw_barrier_policy();
  if (sProbeBarrier || (policy.every != 0 && barrier_in_scope() && ++draws % policy.every == 0)) {
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
  }
}

// AURORA_GLES_SCISSOR=0 ignores GX scissor boxes (diagnostic for drivers that size a job's render area from
// the scissor in effect when the job starts).
bool apply_gx_scissor() {
  static const bool apply = [] {
    const char* value = std::getenv("AURORA_GLES_SCISSOR");
    return value == nullptr || value[0] != '0';
  }();
  return apply;
}

// AURORA_GLES_INDEX_DRAW=plain|instanced replaces glDrawRangeElements for single-instance indexed draws
// (diagnostic for drivers whose index-range handling differs).
enum class IndexDrawMode { Range, Plain, Instanced };
IndexDrawMode index_draw_mode() {
  static const IndexDrawMode mode = [] {
    const char* value = std::getenv("AURORA_GLES_INDEX_DRAW");
    if (value != nullptr && std::strcmp(value, "plain") == 0) {
      return IndexDrawMode::Plain;
    }
    if (value != nullptr && std::strcmp(value, "instanced") == 0) {
      return IndexDrawMode::Instanced;
    }
    return IndexDrawMode::Range;
  }();
  return mode;
}

bool render_draw(const gx::DrawData& d, uint32_t passIndex, uint32_t drawIndex) {
  if (profile::enabled()) {
    profile::state.tag = (static_cast<uint64_t>(passIndex) << 32) | drawIndex;
  }
  profile::Scope drawProfile("gl_draw_total");
  auto* prepared = prepare_pipeline(d.pipeline);
  if (prepared == nullptr) {
    std::fprintf(stderr, "[gles-direct-error] frame=%llu pass=%u draw=%u section=pipeline-lookup\n",
                 static_cast<unsigned long long>(sFrameNumber), passIndex, drawIndex);
    return false;
  }
  const auto& native = prepared->gl;
  if (sLastProgram != native.program) {
    glUseProgram(native.program);
    sLastProgram = native.program;
    ++sGlCalls.programs;
  }
  if (sLastPipeline != d.pipeline) {
    apply_pipeline_state(prepared->config);
    sLastPipeline = d.pipeline;
    ++sGlCalls.pipelineStates;
  }
  if (!gl_ok("pipeline-state", passIndex, drawIndex)) {
    return true;
  }
  {
    profile::Scope resourcesProfile("gl_resources");
    bind_draw_resources(d, *prepared);
  }
  if (!gl_ok("resources", passIndex, drawIndex)) {
    return true;
  }
  profile::Scope drawCallProfile("gl_draw_call");
  ++sGlCalls.draws;
  const GLenum indexType = d.residentArena != 0 ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT;
  const auto indices = reinterpret_cast<const void*>(static_cast<uintptr_t>(d.idxRange.offset));
  if (d.instanceCount == 1) {
    if (d.indexCount != 0) {
      // Explicit vertex range: the driver need not scan the index data (and reads persistently mapped index
      // data correctly on the drivers this was measured on, where plain glDrawElements produced stale draws).
      GLuint start = 0;
      GLuint end = d.vtxCount != 0 ? d.vtxCount - 1 : 0;
      if (d.residentArena != 0) {
        const uint32_t stride = prepared->layout.stride;
        start = d.vertRange.offset / stride;
        end = (d.vertRange.offset + d.vertRange.size) / stride;
        end = end != 0 ? end - 1 : 0;
      }
      switch (index_draw_mode()) {
      case IndexDrawMode::Plain:
        glDrawElements(native.topology, static_cast<GLsizei>(d.indexCount), indexType, indices);
        break;
      case IndexDrawMode::Instanced:
        glDrawElementsInstanced(native.topology, static_cast<GLsizei>(d.indexCount), indexType, indices, 1);
        break;
      default:
        glDrawRangeElements(native.topology, start, end, static_cast<GLsizei>(d.indexCount), indexType, indices);
        break;
      }
    } else {
      glDrawArrays(native.topology, 0, static_cast<GLsizei>(d.vtxCount));
    }
  } else if (d.indexCount != 0) {
    glDrawElementsInstanced(native.topology, static_cast<GLsizei>(d.indexCount), indexType, indices,
                            static_cast<GLsizei>(d.instanceCount));
  } else {
    glDrawArraysInstanced(native.topology, 0, static_cast<GLsizei>(d.vtxCount), static_cast<GLsizei>(d.instanceCount));
  }
  draw_barrier((d.indexCount != 0 ? d.indexCount : d.vtxCount) * std::max(d.instanceCount, 1u));
  return gl_ok("draw", passIndex, drawIndex);
}

// AURORA_GLES_CLEAR=gl issues GX clears as glClear with the same masks, color and depth instead of a
// full-framebuffer triangle (diagnostic for drivers that mishandle the triangle's tile coverage).
bool clear_with_gl_clear() {
  static const bool gl = [] {
    const char* value = std::getenv("AURORA_GLES_CLEAR");
    return value != nullptr && std::strcmp(value, "gl") == 0;
  }();
  return gl;
}

bool render_clear(const clear::DrawData& d, uint32_t width, uint32_t height, uint32_t passIndex, uint32_t drawIndex) {
  clear::PipelineConfig config;
  if (!clear::find_pipeline_config(d.pipeline, config)) {
    std::fprintf(stderr, "[gles-direct-error] frame=%llu pass=%u draw=%u section=clear-pipeline-lookup\n",
                 static_cast<unsigned long long>(sFrameNumber), passIndex, drawIndex);
    return false;
  }
  const GLuint program = clear_program();
  if (program == 0) {
    return false;
  }
  glUseProgram(program);
  sLastProgram = Unknown;
  sLastPipeline = UINTPTR_MAX;
  sGLState = GLStateCache{};
  glDisable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
  glBlendFuncSeparate(GL_CONSTANT_COLOR, GL_ZERO, GL_CONSTANT_ALPHA, GL_ZERO);
  glBlendColor(static_cast<float>(d.color.r), static_cast<float>(d.color.g), static_cast<float>(d.color.b),
               static_cast<float>(d.color.a));
  glDisable(GL_STENCIL_TEST);
  glDisable(GL_POLYGON_OFFSET_FILL);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_ALWAYS);
  glColorMask(config.clearColor, config.clearColor, config.clearColor, config.clearAlpha);
  glDepthMask(config.clearDepth ? GL_TRUE : GL_FALSE);
  glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
  glDepthRangef(d.depth, d.depth);
  glScissor(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
  if (clear_with_gl_clear()) {
    glClearColor(static_cast<float>(d.color.r), static_cast<float>(d.color.g), static_cast<float>(d.color.b),
                 static_cast<float>(d.color.a));
    glClearDepthf(d.depth);
    glClear((config.clearColor || config.clearAlpha ? GL_COLOR_BUFFER_BIT : 0) |
            (config.clearDepth ? GL_DEPTH_BUFFER_BIT : 0));
  } else {
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }
  draw_barrier(3);
  return gl_ok("clear", passIndex, drawIndex);
}

void reset_memos() {
  sLastProgram = sLastUniform = sLastUniformBinding = sLastLayout = Unknown;
  sLastUniformOffset = Unknown;
  sLastPipeline = UINTPTR_MAX;
  sTextureState = {};
  sGLState = GLStateCache{};
  sActiveUnit = Unknown;
  sLastVertexBuffer = Unknown;
  sLastVertexOffset = sLastVertexStride = UINT32_MAX;
  sBindingDivisor = Unknown;
  sLastStorage = {Unknown, Unknown};
  for (auto& [ref, prepared] : sPrepared) {
    prepared.lastImmediatePad = UINT32_MAX;
  }
}

// Replays a plan's commands into the bound framebuffer; returns the number of draws.
uint32_t replay_plan(const PassPlan& plan, uint32_t passIndex) {
  uint32_t drawIndex = 0;
  for (const auto& command : plan.commands) {
    switch (command.type) {
    case CommandType::SetViewport: {
      const auto& v = command.viewport;
      glViewport(static_cast<GLint>(v.left), static_cast<GLint>(v.top), static_cast<GLsizei>(v.width),
                 static_cast<GLsizei>(v.height));
      const float nearDepth = gx::UseReversedZ ? 1.f - v.zfar : v.znear;
      const float farDepth = gx::UseReversedZ ? 1.f - v.znear : v.zfar;
      glDepthRangef(nearDepth, farDepth);
    } break;
    case CommandType::SetScissor: {
      if (!apply_gx_scissor()) {
        glScissor(0, 0, static_cast<GLsizei>(plan.width), static_cast<GLsizei>(plan.height));
        break;
      }
      const auto& s = command.scissor;
      const auto x = std::min(static_cast<uint32_t>(std::max(s.x, 0)), plan.width);
      const auto y = std::min(static_cast<uint32_t>(std::max(s.y, 0)), plan.height);
      glScissor(static_cast<GLint>(x), static_cast<GLint>(y),
                static_cast<GLsizei>(std::min(static_cast<uint32_t>(std::max(s.width, 0)), plan.width - x)),
                static_cast<GLsizei>(std::min(static_cast<uint32_t>(std::max(s.height, 0)), plan.height - y)));
    } break;
    case CommandType::Draw:
      sBarrierDrawInPass = drawIndex;
      if (command.isClear) {
        render_clear(command.clear, plan.width, plan.height, passIndex, drawIndex);
      } else {
        render_draw(command.draw, passIndex, drawIndex);
      }
      ++drawIndex;
      break;
    default:
      break;
    }
  }
  return drawIndex;
}

// Driver probe. A driver that drops draws within a job (see DrawBarrierPolicy) renders a busy pass differently
// with and without a texture-fetch barrier after each draw, while a correct driver renders both identically.
// For the first few busy frames the plan with the most draws is rendered into a private framebuffer both ways
// and read back; any real difference turns the per-draw barrier on for the rest of the run and publishes a
// notice (driver_notice). Both renderings replay the same recorded draws from the same buffers and textures.
struct DriverProbe {
  GLuint framebuffer = 0, color = 0, depth = 0;
  uint32_t width = 0, height = 0;
  uint64_t nextFrame = 0;
  std::vector<uint8_t> plain, reference;
};
DriverProbe sProbe;
constexpr uint32_t ProbeMinDraws = 8;
constexpr uint64_t ProbeSpacingFrames = 20;
std::string sDriverNotice;
std::atomic_bool sDriverNoticeReady = false;

void release_probe_target() {
  auto& p = sProbe;
  if (p.framebuffer != 0) {
    glDeleteFramebuffers(1, &p.framebuffer);
  }
  if (p.color != 0) {
    glDeleteRenderbuffers(1, &p.color);
  }
  if (p.depth != 0) {
    glDeleteRenderbuffers(1, &p.depth);
  }
  p.framebuffer = p.color = p.depth = 0;
  p.width = p.height = 0;
  p.plain = {};
  p.reference = {};
}

// Binds the probe framebuffer, (re)creating it at the plan's size.
bool bind_probe_target(uint32_t width, uint32_t height) {
  auto& p = sProbe;
  if (p.framebuffer == 0 || p.width != width || p.height != height) {
    release_probe_target();
    glGenRenderbuffers(1, &p.color);
    glBindRenderbuffer(GL_RENDERBUFFER, p.color);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
    glGenRenderbuffers(1, &p.depth);
    glBindRenderbuffer(GL_RENDERBUFFER, p.depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, static_cast<GLsizei>(width),
                          static_cast<GLsizei>(height));
    glGenFramebuffers(1, &p.framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, p.framebuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, p.color);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, p.depth);
    p.width = width;
    p.height = height;
  }
  glBindFramebuffer(GL_FRAMEBUFFER, p.framebuffer);
  return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
}

// Renders the plan into the probe framebuffer and reads it back; returns the GPU-complete render time in ms.
double render_probe_image(const PassPlan& plan, uint32_t passIndex, bool barrier, std::vector<uint8_t>& out) {
  glFinish();
  const auto start = std::chrono::steady_clock::now();
  reset_memos();
  glDisable(GL_SCISSOR_TEST);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glDepthMask(GL_TRUE);
  glClearColor(plan.clearColor.x(), plan.clearColor.y(), plan.clearColor.z(), plan.clearColor.w());
  glClearDepthf(plan.clearDepth);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_SCISSOR_TEST);
  glViewport(0, 0, static_cast<GLsizei>(plan.width), static_cast<GLsizei>(plan.height));
  glScissor(0, 0, static_cast<GLsizei>(plan.width), static_cast<GLsizei>(plan.height));
  sProbeBarrier = barrier;
  replay_plan(plan, passIndex);
  sProbeBarrier = false;
  glFinish();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  out.resize(static_cast<size_t>(plan.width) * plan.height * 4);
  glReadPixels(0, 0, static_cast<GLsizei>(plan.width), static_cast<GLsizei>(plan.height), GL_RGBA, GL_UNSIGNED_BYTE,
               out.data());
  return ms;
}

void probe_driver(const PassPlan& plan, uint32_t passIndex) {
  auto& policy = draw_barrier_policy();
  GLint drawFramebuffer = 0, readFramebuffer = 0, renderbuffer = 0, packBuffer = 0, packAlignment = 4;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFramebuffer);
  glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFramebuffer);
  glGetIntegerv(GL_RENDERBUFFER_BINDING, &renderbuffer);
  glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &packBuffer);
  glGetIntegerv(GL_PACK_ALIGNMENT, &packAlignment);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  bool finished = false;
  if (!bind_probe_target(plan.width, plan.height)) {
    Log.warn("Driver probe: framebuffer incomplete; probe skipped");
    finished = true;
  } else {
    const double plainMs = render_probe_image(plan, passIndex, false, sProbe.plain);
    const double barrierMs = render_probe_image(plan, passIndex, true, sProbe.reference);
    size_t differing = 0;
    for (size_t i = 0; i < sProbe.plain.size(); i += 4) {
      for (size_t c = 0; c < 4; ++c) {
        if (std::abs(static_cast<int>(sProbe.plain[i + c]) - static_cast<int>(sProbe.reference[i + c])) > 8) {
          ++differing;
          break;
        }
      }
    }
    const size_t pixels = static_cast<size_t>(plan.width) * plan.height;
    const bool faulty = differing > std::max<size_t>(64, pixels / 1000);
    Log.info("Driver probe: frame {} pass {} ({}x{}, {} draws): {} of {} pixels differ; {:.1f} ms plain, {:.1f} ms "
             "with per-draw barriers",
             sFrameNumber, passIndex, plan.width, plan.height, plan.draws, differing, pixels, plainMs, barrierMs);
    if (faulty) {
      const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
      policy.every = 1;
      const bool slow = barrierMs > plainMs * 1.5 && barrierMs - plainMs > 4.0;
      Log.warn("Driver probe: the GPU driver drops draws without texture-fetch barriers; per-draw barrier enabled "
               "(GL_VERSION {})",
               version != nullptr ? version : "unknown");
      sDriverNotice = std::string{"GPU driver update needed\n\nThis device is supported, but its GPU driver has a "
                                  "rendering bug. A workaround is active"} +
                      (slow ? ", so the game runs slowly." : ".") +
                      " Updating the GPU driver restores correct, full-speed rendering.\n\nDriver: " +
                      (version != nullptr ? version : "unknown");
      sDriverNoticeReady = true;
      finished = true;
    } else if (--policy.probes == 0) {
      Log.info("Driver probe: no rendering fault found");
      finished = true;
    }
    sProbe.nextFrame = sFrameNumber + ProbeSpacingFrames;
  }
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(drawFramebuffer));
  glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(readFramebuffer));
  glBindRenderbuffer(GL_RENDERBUFFER, static_cast<GLuint>(renderbuffer));
  glBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(packBuffer));
  glPixelStorei(GL_PACK_ALIGNMENT, packAlignment);
  if (finished) {
    policy.probes = 0;
    release_probe_target();
  }
  while (glGetError() != GL_NO_ERROR) {
  }
}

// Dawn's render pass callback: the framebuffer is bound and cleared and the dynamic state is at its defaults.
bool render_pass(void*, uint32_t passIndex, const char* label) {
  if (sNextPlan >= sPlans.size()) {
    return false;
  }
  auto& plan = sPlans[sNextPlan];
  if (label == nullptr || plan.label != label) {
    return false;
  }
  const bool probe = sNextPlan == sProbePlan;
  sBarrierPlan = static_cast<uint32_t>(sNextPlan);
  ++sNextPlan;
  if (!plan.eligible) {
    return false;
  }
  profile::Scope passProfile("gl_pass", passIndex);
  PassTimer timer;
  while (glGetError() != GL_NO_ERROR) {
  }
  if (sVao == 0) {
    glGenVertexArrays(1, &sVao);
  }
  glBindVertexArray(sVao);
  sPassEbo = sMapped != nullptr ? sMapped->indices : dawn::native::opengl::GetGLInteropBuffer(resources().indexBuffer.Get());
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sPassEbo);
  // sEnabledAttributes persists: the vertex array object is ours and keeps its enabled attributes between passes.
  glDisable(GL_STENCIL_TEST);
  if (probe) {
    probe_driver(plan, passIndex);
  }
  reset_memos();
  glEnable(GL_SCISSOR_TEST);
  if (draw_barrier_policy().pass) {
    // Dawn's copies and uploads since the previous pass must resolve before this pass samples them.
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
  }
  gl_ok("pass-state", passIndex, 0);
  const uint32_t drawIndex = replay_plan(plan, passIndex);
  if (draw_barrier_policy().pass) {
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
  }
  if (draw_barrier_policy().finish) {
    glFinish();
  } else if (draw_barrier_policy().flush) {
    glFlush();
  }
  gl_ok("pass-end", passIndex, drawIndex, true);
  // Once an eligible pass is intercepted it must not also be replayed by Dawn, even after a GL error.
  return true;
}

bool stream_independent_task(EncoderTaskId type) {
  return std::find(sStreamIndependentTasks.begin(), sStreamIndependentTasks.end(), type) != sStreamIndependentTasks.end();
}

void report_gl_calls() {
  static GlCallStats sum;
  static unsigned frames = 0;
  sum.programs += sGlCalls.programs;
  sum.pipelineStates += sGlCalls.pipelineStates;
  sum.textures += sGlCalls.textures;
  sum.texParams += sGlCalls.texParams;
  sum.samplers += sGlCalls.samplers;
  sum.ubos += sGlCalls.ubos;
  sum.vbos += sGlCalls.vbos;
  sum.layouts += sGlCalls.layouts;
  sum.immediates += sGlCalls.immediates;
  sum.draws += sGlCalls.draws;
  if (++frames % 120 == 0) {
    std::fprintf(stderr,
                 "[gles-direct-gl-calls] per frame: draws=%.1f programs=%.1f pipeline-states=%.1f textures=%.1f "
                 "tex-params=%.1f samplers=%.1f ubos=%.1f vbos=%.1f layouts=%.1f immediates=%.1f\n",
                 sum.draws / 120.0, sum.programs / 120.0, sum.pipelineStates / 120.0, sum.textures / 120.0,
                 sum.texParams / 120.0, sum.samplers / 120.0, sum.ubos / 120.0, sum.vbos / 120.0, sum.layouts / 120.0,
                 sum.immediates / 120.0);
    sum = {};
  }
  sGlCalls = {};
}
} // namespace

bool encode_pass_resources(const wgpu::RenderPassEncoder& encoder, RenderPass& pass, std::string_view label) {
  pass.directLabel = label;
  if (!sEnabled || pass.msaaSamples != 1 || pass.colorAttachmentCount != 1) {
    return false;
  }
  // Decide for the whole immutable pass before recording anything: failure preserves every reference command,
  // never a partially intercepted pass.
  std::unordered_set<PipelineRef> seenPipelines;
  std::vector<wgpu::RenderPipeline> pipelines;
  std::unordered_set<uint32_t> arenas;
  bool hasDraw = false;
  for (const auto& command : pass.commands) {
    if (command.type == CommandType::CustomDraw) {
      return false;
    }
    if (command.type != CommandType::Draw) {
      continue;
    }
    gx::DrawData d;
    clear::DrawData clearDraw;
    if (decode_clear_draw(command.data.draw, clearDraw)) {
      continue;
    }
    if (!decode_gx_draw(command.data.draw, d)) {
      return false;
    }
    hasDraw = true;
    if (seenPipelines.insert(d.pipeline).second) {
      gx::PipelineConfig c;
      wgpu::RenderPipeline p;
      if (!get_pipeline(d.pipeline, p) || !gx::find_pipeline_config(d.pipeline, c) || !pipeline_eligible(c)) {
        return false;
      }
      pipelines.push_back(std::move(p));
    }
    if (d.residentArena != 0) {
      arenas.insert(d.residentArena);
    }
  }
  if (!hasDraw) {
    return false;
  }
  // The direct path issues the GL work itself; Dawn only needs enough recorded state to keep the pass setup,
  // lazy clears and resource lifetimes correct. Pipelines, uniform windows and texture bind groups stay alive
  // through the GX caches, so recording every one of them (hundreds per pass) would be pure overhead.
  const auto& res = resources();
  for (auto arena : arenas) {
    uint64_t size = 0;
    const auto& buffer = gx::resident::arena_buffer(arena - 1, size);
    if (buffer) {
      encoder.SetVertexBuffer(0, buffer);
    }
  }
  encoder.SetVertexBuffer(0, res.vertexBuffer);
  encoder.SetIndexBuffer(res.indexBuffer, wgpu::IndexFormat::Uint16);
  encoder.SetBindGroup(0, res.staticBindGroup);
  encoder.SetPipeline(pipelines.front());
  pass.directResourcesOnly = true;
  return true;
}

void prepare_frame(FramePacket& frame) {
  profile::begin(frame.frameId, "render");
  sPlans.clear();
  sNextPlan = 0;
  ++sFrameNumber;
  if (stats_enabled()) {
    report_gl_calls();
  } else {
    sGlCalls = {};
  }
  if (!sEnabled) {
    return;
  }
  // Texture-object memos persist across passes and frames; forget entries whose GL names died so a recycled
  // name starts clean.
  GLuint dead[256];
  while (const uint32_t n = dawn::native::opengl::GetGLInteropDestroyedTextures(webgpu::g_device.Get(), dead, 256)) {
    for (uint32_t i = 0; i < n; ++i) {
      sTextureParams.erase(dead[i]);
      sTextureSamplers.erase(dead[i]);
      for (auto& unit : sTextureState) {
        if (unit.valid && unit.texture.texture == dead[i]) {
          unit.valid = false;
        }
      }
    }
    if (n < 256) {
      break;
    }
  }
  sMapped = mapped_slot(frame);
  bool streamsNeeded = false; // some consumer reads the streams through Dawn's buffers
  for (const auto& task : frame.encoderTasks) {
    if (!stream_independent_task(task.type)) {
      streamsNeeded = true;
    }
  }
  static unsigned sSortRuns = 0, sSortedDraws = 0, sSortFrames = 0;
  for (auto& pass : frame.renderPasses) {
    if (!pass.sealed || pass.discardable || pass.colorAttachmentCount == 0) {
      continue;
    }
    PassPlan plan{
        .label = pass.directLabel,
        .width = pass.colorAttachments[0].size.width,
        .height = pass.colorAttachments[0].size.height,
        .clearColor = pass.colorAttachments[0].clearValue,
        .clearDepth = pass.clearDepthValue,
        .eligible = pass.msaaSamples == 1 && pass.colorAttachmentCount == 1,
    };
    bool hasGxDraw = false;
    uint32_t gxDraws = 0, clearDraws = 0, otherDraws = 0, customDraws = 0;
    for (const auto& command : pass.commands) {
      PlanCommand out{.type = command.type};
      if (command.type == CommandType::SetViewport) {
        out.viewport = command.data.setViewport;
      } else if (command.type == CommandType::SetScissor) {
        out.scissor = command.data.setScissor;
      } else if (command.type == CommandType::Draw) {
        if (decode_gx_draw(command.data.draw, out.draw)) {
          hasGxDraw = true;
          ++gxDraws;
          gx::PipelineConfig config;
          if (!gx::find_pipeline_config(out.draw.pipeline, config) || !pipeline_eligible(config)) {
            plan.eligible = false;
          } else {
            out.sortable = config.depthCompare && config.depthUpdate &&
                           (config.depthFunc == GX_LESS || config.depthFunc == GX_LEQUAL) &&
                           config.blendMode == GX_BM_NONE && config.colorUpdate && config.polygonOffsetBits == 0;
          }
        } else if (decode_clear_draw(command.data.draw, out.clear)) {
          out.isClear = true;
          ++clearDraws;
        } else {
          plan.eligible = false;
          ++otherDraws;
        }
      } else if (command.type == CommandType::CustomDraw) {
        plan.eligible = false;
        ++customDraws;
      }
      plan.commands.push_back(out);
    }
    plan.eligible &= hasGxDraw;
    plan.draws = gxDraws + clearDraws;
    if (hasGxDraw && !pass.directResourcesOnly) {
      streamsNeeded = true;
    }
    if (customDraws != 0) {
      streamsNeeded = true;
    }
    // sortOpaqueDraws: runs of consecutive opaque, depth-ordered draws are grouped by program, textures and
    // uniform window so the driver sees fewer state changes. Not exact when opaque surfaces share depth values.
    if (g_config.sortOpaqueDraws && plan.eligible) {
      auto& cmds = plan.commands;
      size_t i = 0;
      while (i < cmds.size()) {
        if (cmds[i].type != CommandType::Draw || cmds[i].isClear || !cmds[i].sortable) {
          ++i;
          continue;
        }
        size_t j = i;
        while (j < cmds.size() && cmds[j].type == CommandType::Draw && !cmds[j].isClear && cmds[j].sortable) {
          ++j;
        }
        if (j - i > 1) {
          std::stable_sort(cmds.begin() + static_cast<std::ptrdiff_t>(i), cmds.begin() + static_cast<std::ptrdiff_t>(j),
                           [](const PlanCommand& a, const PlanCommand& b) {
                             if (a.draw.pipeline != b.draw.pipeline) {
                               return a.draw.pipeline < b.draw.pipeline;
                             }
                             if (a.draw.bindGroups.textureBindGroup != b.draw.bindGroups.textureBindGroup) {
                               return a.draw.bindGroups.textureBindGroup < b.draw.bindGroups.textureBindGroup;
                             }
                             return gx::uniform_window_index(a.draw.uniformRange.offset) <
                                    gx::uniform_window_index(b.draw.uniformRange.offset);
                           });
          ++sSortRuns;
          sSortedDraws += static_cast<unsigned>(j - i);
        }
        i = j;
      }
    }
    AURORA_ASSERT(!pass.directResourcesOnly || plan.eligible, "direct resource-only pass lost eligibility");
    if (stats_enabled() && sFrameNumber % 120 == 0) {
      std::fprintf(stderr,
                   "[gles-direct-plan] frame=%llu pass=%zu eligible=%u gx=%u clear=%u other=%u custom=%u commands=%zu\n",
                   static_cast<unsigned long long>(sFrameNumber), sPlans.size(), plan.eligible ? 1u : 0u, gxDraws,
                   clearDraws, otherDraws, customDraws, plan.commands.size());
    }
    sPlans.push_back(std::move(plan));
  }
  // The render worker resolves the policy on the first direct pass; probes start from the frame after.
  sProbePlan = SIZE_MAX;
  if (draw_barrier_policy_resolved() && draw_barrier_policy().probes != 0 && sFrameNumber >= sProbe.nextFrame) {
    uint32_t most = ProbeMinDraws - 1;
    for (size_t i = 0; i < sPlans.size(); ++i) {
      if (sPlans[i].eligible && sPlans[i].draws > most && sPlans[i].width != 0 && sPlans[i].height != 0) {
        most = sPlans[i].draws;
        sProbePlan = i;
      }
    }
  }
  if (g_config.sortOpaqueDraws && stats_enabled() && ++sSortFrames % 360 == 0) {
    std::fprintf(stderr, "[gles-direct-sort] per frame: runs=%.1f sorted-draws=%.1f\n", sSortRuns / 360.0,
                 sSortedDraws / 360.0);
    sSortRuns = sSortedDraws = 0;
  }
  // Frames recorded into mapped GL storage reach Dawn's buffers only when something reads them through Dawn.
  if (sMapped != nullptr && streamsNeeded) {
    const auto& res = resources();
    const auto upload = [](const wgpu::Buffer& dst, const ByteBuffer& src) {
      if (src.size() != 0) {
        webgpu::g_queue.WriteBuffer(dst, 0, src.data(), AURORA_ALIGN(src.size(), 4));
      }
    };
    upload(res.vertexBuffer, frame.verts);
    upload(res.uniformBuffer, frame.uniforms);
    upload(res.indexBuffer, frame.indices);
  }
}

void install_frame() {
  if (sEnabled) {
    dawn::native::opengl::SetGLInteropRenderPassCallback(render_pass, nullptr);
  }
}

void uninstall_frame() {
  if (sEnabled) {
    dawn::native::opengl::SetGLInteropRenderPassCallback(nullptr, nullptr);
  }
  profile::end();
}

void shutdown_mapped_slots(); // gles_mapped_streams.cpp

void shutdown() {
  uninstall_frame();
  if (!sEnabled) {
    return;
  }
  shutdown_mapped_slots();
  dawn::native::opengl::RunGLInterop(
      webgpu::g_device.Get(),
      [](void*) {
        for (auto& pending : PassTimer::pending) {
          glDeleteQueries(1, &pending.query);
        }
        PassTimer::pending.clear();
        PassTimer::get64 = nullptr;
        PassTimer::initialized = false;
        if (sVao != 0) {
          glDeleteVertexArrays(1, &sVao);
        }
        if (sClearProgram != 0) {
          glDeleteProgram(sClearProgram);
        }
        release_probe_target();
        sVao = sClearProgram = sPassEbo = sEnabledAttributes = 0;
      },
      nullptr);
  sPrepared.clear();
  sPlans.clear();
  sNextPlan = 0;
  sFrameNumber = 0;
  sResolvedTextures.clear();
  sTextureParams.clear();
  sTextureSamplers.clear();
  sResidentGL.clear();
  sMapped = nullptr;
}
#else
namespace {
std::atomic_bool sSceneOnSurfaceBlocked = false;
std::string sDriverNotice;
std::atomic_bool sDriverNoticeReady = false;
} // namespace
bool encode_pass_resources(const wgpu::RenderPassEncoder&, detail::RenderPass& pass, std::string_view label) {
  pass.directLabel = label;
  return false;
}
void prepare_frame(detail::FramePacket&) {}
void install_frame() {}
void uninstall_frame() {}
void shutdown() {}
#endif
} // namespace aurora::gfx::gles_direct
