#include "gles_direct.hpp"

#include "frame_packet.hpp"
#include "resources.hpp"
#include "../gx/gx.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if AURORA_GLES_DIRECT
#include <dawn/native/OpenGLBackend.h>
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#endif

namespace aurora::gfx::gles_direct {
namespace {
constexpr Module Log{"aurora::gfx::gles_direct"};
std::vector<MappedSlot> sSlots;
std::atomic<bool> sReady{false};
} // namespace

#if AURORA_GLES_DIRECT
namespace {
using BufferStorageProc = void(GL_APIENTRY*)(GLenum, GLsizeiptr, const void*, GLbitfield);
constexpr GLbitfield MapPersistentBit = 0x0040; // GL_MAP_PERSISTENT_BIT_EXT
BufferStorageProc sBufferStorage = nullptr;

// Persistent, explicitly flushed mapping: the FIFO processor writes from another core and the
// written ranges are flushed on the render thread before submit, rather than relying on coherent
// mapping semantics for cross-thread writes.
bool make_stream(GLuint& name, uint8_t*& mapped, size_t bytes) {
  constexpr GLbitfield storageFlags = GL_MAP_WRITE_BIT | MapPersistentBit;
  constexpr GLbitfield mapFlags = GL_MAP_WRITE_BIT | MapPersistentBit | GL_MAP_FLUSH_EXPLICIT_BIT;
  glGenBuffers(1, &name);
  glBindBuffer(GL_COPY_WRITE_BUFFER, name);
  sBufferStorage(GL_COPY_WRITE_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, storageFlags);
  mapped = static_cast<uint8_t*>(glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, static_cast<GLsizeiptr>(bytes), mapFlags));
  if (glGetError() != GL_NO_ERROR || mapped == nullptr) {
    mapped = nullptr;
    return false;
  }
  std::memset(mapped, 0, bytes);
  return true;
}

// The Mali blob on Allwinner H700 handhelds (libmali.so.0.20, fbdev) grows a table per index buffer inside
// glDrawRangeElements (20-byte entries, doubling) and only lets go of it with the buffer. Fed by a stream whose
// draws land at new offsets every frame, the three slots' tables reached 20 MiB each after 22 minutes of play and
// kept doubling: an hour cost 200 MB of process memory. So a slot's index stream is replaced with fresh storage
// after this many frames, which keeps each table under 3 MiB. AURORA_GLES_INDEX_STREAM_FRAMES overrides; 0 keeps
// the storage.
uint32_t index_stream_frames() {
  static const uint32_t frames = [] {
    const char* value = std::getenv("AURORA_GLES_INDEX_STREAM_FRAMES");
    return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 256u;
  }();
  return frames;
}

// The slot is idle here: its fence has signalled and it has not been handed back to the recorder.
void renew_index_stream(MappedSlot& slot) {
  const uint32_t frames = index_stream_frames();
  if (frames == 0 || ++slot.indexFrames < frames) {
    return;
  }
  slot.indexFrames = 0;
  GLuint fresh = 0;
  uint8_t* data = nullptr;
  if (make_stream(fresh, data, stream_sizes().index)) {
    glBindBuffer(GL_COPY_WRITE_BUFFER, slot.indices);
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);
    glDeleteBuffers(1, &slot.indices);
    slot.indices = fresh;
    slot.indexData = data;
  } else {
    // Out of memory or a driver refusal: keep the storage in use.
    while (glGetError() != GL_NO_ERROR) {
    }
    if (fresh != 0) {
      glDeleteBuffers(1, &fresh);
    }
  }
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

struct CreateRequest {
  size_t count;
};

void create_slots_gl(void* data) {
  const auto& request = *static_cast<CreateRequest*>(data);
  const char* extensions = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
  BufferStorageProc bufferStorage = nullptr;
  if (extensions != nullptr && std::strstr(extensions, "GL_EXT_buffer_storage") != nullptr) {
    bufferStorage = reinterpret_cast<BufferStorageProc>(eglGetProcAddress("glBufferStorageEXT"));
  }
  if (bufferStorage == nullptr) {
    Log.info("GL_EXT_buffer_storage unavailable; frame streams stay staged");
    return;
  }
  sBufferStorage = bufferStorage;
  const auto& sizes = stream_sizes();
  std::vector<MappedSlot> slots(request.count);
  for (auto& slot : slots) {
    // A whole extra window lets a draw in the last window bind a full 64 KiB range.
    if (!make_stream(slot.uniforms, slot.uniformData, sizes.uniform + gx::UniformWindowSize) ||
        !make_stream(slot.indices, slot.indexData, sizes.index) ||
        !make_stream(slot.vertices, slot.vertexData, sizes.vertex)) {
      Log.warn("persistent mapping failed; frame streams stay staged");
      while (glGetError() != GL_NO_ERROR) {
      }
      glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
      return;
    }
  }
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
  sSlots = std::move(slots);
  Log.info("uniform, index and vertex streams recorded into persistently mapped GL storage ({} slots)",
           request.count);
  sReady.store(true, std::memory_order_release);
}

struct FlushRequest {
  const MappedSlot* slot;
  size_t uniformBytes;
  size_t indexBytes;
  size_t vertexBytes;
};

void flush_slot_gl(void* data) {
  const auto& request = *static_cast<FlushRequest*>(data);
  const auto flush = [](GLuint name, size_t bytes) {
    if (bytes == 0) {
      return;
    }
    glBindBuffer(GL_COPY_WRITE_BUFFER, name);
    glFlushMappedBufferRange(GL_COPY_WRITE_BUFFER, 0, static_cast<GLsizeiptr>((bytes + 3) & ~size_t{3}));
  };
  flush(request.slot->uniforms, request.uniformBytes);
  flush(request.slot->indices, request.indexBytes);
  flush(request.slot->vertices, request.vertexBytes);
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void fence_slot_gl(void* data) {
  auto& slot = *static_cast<MappedSlot*>(data);
  if (slot.fence != nullptr) {
    glDeleteSync(static_cast<GLsync>(slot.fence));
  }
  slot.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  glFlush();
}

struct FenceQuery {
  MappedSlot* slot;
  bool block;
  bool done;
};

void check_fence_gl(void* data) {
  auto& query = *static_cast<FenceQuery*>(data);
  if (query.slot->fence == nullptr) {
    query.done = true;
    return;
  }
  aurora_render_phase = query.block ? "fence-wait-block" : "fence-poll";
  GLenum status;
  int timeouts = 0;
  do {
    status = glClientWaitSync(static_cast<GLsync>(query.slot->fence), GL_SYNC_FLUSH_COMMANDS_BIT,
                              query.block ? 1000000000 : 0);
  } while (query.block && status == GL_TIMEOUT_EXPIRED && ++timeouts < 3);
  // Mali r13p0 can leave a fence unsignalled forever (the RG froze on the Nintendo logo); after three seconds the
  // slot is released anyway, risking one torn frame instead of a hang.
  if (query.block && status == GL_TIMEOUT_EXPIRED) {
    Log.warn("mapped-slot fence not signalled after {}s; releasing the slot", timeouts);
    status = GL_WAIT_FAILED;
  }
  query.done = status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED || status == GL_WAIT_FAILED;
  if (query.done) {
    glDeleteSync(static_cast<GLsync>(query.slot->fence));
    query.slot->fence = nullptr;
    renew_index_stream(*query.slot);
  }
}

void destroy_slots_gl(void*) {
  for (auto& slot : sSlots) {
    if (slot.fence != nullptr) {
      glClientWaitSync(static_cast<GLsync>(slot.fence), GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000);
      glDeleteSync(static_cast<GLsync>(slot.fence));
    }
    for (GLuint name : {slot.uniforms, slot.indices, slot.vertices}) {
      if (name != 0) {
        glBindBuffer(GL_COPY_WRITE_BUFFER, name);
        glUnmapBuffer(GL_COPY_WRITE_BUFFER);
        glDeleteBuffers(1, &name);
      }
    }
  }
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
  sSlots.clear();
}
} // namespace

void create_mapped_slots(size_t count) {
  if (!enabled() || !mapped_streams_enabled() || sReady.load(std::memory_order_acquire)) {
    return;
  }
  CreateRequest request{count};
  dawn::native::opengl::RunGLInterop(webgpu::g_device.Get(), create_slots_gl, &request);
}

void flush_mapped_slot(const detail::FramePacket& frame) {
  const auto* slot = mapped_slot(frame);
  if (slot == nullptr) {
    return;
  }
  FlushRequest request{slot, frame.uniforms.size(), frame.indices.size(), frame.verts.size()};
  dawn::native::opengl::RunGLInterop(webgpu::g_device.Get(), flush_slot_gl, &request);
}

void fence_mapped_slot(size_t stagingSlot) {
  if (stagingSlot >= sSlots.size()) {
    return;
  }
  dawn::native::opengl::RunGLInterop(webgpu::g_device.Get(), fence_slot_gl, &sSlots[stagingSlot]);
}

bool mapped_slot_fence_done(size_t stagingSlot, bool block) {
  if (stagingSlot >= sSlots.size()) {
    return true;
  }
  FenceQuery query{&sSlots[stagingSlot], block, false};
  if (!dawn::native::opengl::RunGLInterop(webgpu::g_device.Get(), check_fence_gl, &query)) {
    return true;
  }
  return query.done;
}

void shutdown_mapped_slots() {
  if (sSlots.empty()) {
    return;
  }
  sReady.store(false, std::memory_order_release);
  dawn::native::opengl::RunGLInterop(webgpu::g_device.Get(), destroy_slots_gl, nullptr);
  sSlots.clear();
}
#else
void create_mapped_slots(size_t) {}
void flush_mapped_slot(const detail::FramePacket&) {}
void fence_mapped_slot(size_t) {}
bool mapped_slot_fence_done(size_t, bool) { return true; }
void shutdown_mapped_slots() {}
#endif

bool mapped_slots_ready() noexcept { return sReady.load(std::memory_order_acquire); }

const MappedSlot* mapped_slot(size_t stagingSlot) noexcept {
  if (!sReady.load(std::memory_order_acquire) || stagingSlot >= sSlots.size()) {
    return nullptr;
  }
  return &sSlots[stagingSlot];
}

const MappedSlot* mapped_slot(const detail::FramePacket& frame) noexcept {
  return frame.mappedStreams ? mapped_slot(frame.stagingBuffer) : nullptr;
}
} // namespace aurora::gfx::gles_direct
