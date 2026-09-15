#pragma once
// OpenGL ES direct submission (AuroraConfig::glesDirectSubmission, CMake option AURORA_GLES_DIRECT).
//
// On draw-call-bound GLES devices (Mali, Adreno, VideoCore class GPUs in handhelds, Raspberry Pi
// boards and weaker phones) the WebGPU command executor's per-draw work dominates the render
// thread: on a Mali-G52 handheld running a GX title, replaying ~180 GX draws through Dawn's GL backend
// cost ~47 ms of render work per frame. This path keeps Dawn's resource ownership, uploads, texture
// cache and render pass setup, but issues the GL calls of eligible GX render passes itself through
// Dawn's native GL interop extension (dawn/native/OpenGLBackend.h): once Dawn has bound and cleared
// the pass's framebuffer its render pass callback replays the recorded GX draws with a redundant-
// state filter, the uniform table bound once per window, texture state folded into the texture
// objects and glDrawRangeElements. With the same scenes that brought render work to ~13 ms.
//
// Eligibility is decided per pass before anything is recorded, and a pass that is not eligible (a
// custom draw, an MSAA or multi-target pass, a fog-range shader, a pipeline still compiling) goes
// through the WebGPU path unchanged; a frame is never partially intercepted.
//
// glesMappedStreams records the frame's uniform records, indices and decoded vertices straight into
// persistently mapped GL buffers (GL_EXT_buffer_storage), one fenced slot per staging buffer, so the
// direct path binds the GL names the FIFO processor wrote without a staging copy. Frames in which a
// pass falls back are uploaded to the WebGPU buffers once at frame end.
//
// Requires cpuVertexDecode and implies uniformTable and batchDraws (the record index rides in the
// vertices). Everything here runs on the render worker unless noted.
#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx::detail {
struct FramePacket;
struct RenderPass;
} // namespace aurora::gfx::detail

namespace aurora::gfx::gles_direct {
// Compiled in (AURORA_GLES_DIRECT) with a Dawn that has the native GL interop extension.
bool available() noexcept;
// Resolves the automatic AuroraConfig fields for the selected backend: glesDirectSubmission and
// glesMappedStreams (0 = automatic) and, when direct submission is on, the uniform table and draw
// batching it relies on. Called by webgpu::initialize once the adapter is known, before the device
// toggles are chosen.
void configure(wgpu::BackendType backend);
bool enabled() noexcept;
bool mapped_streams_enabled() noexcept;

// Encode time. Records only the resources Dawn must keep tracking for the pass and marks it for direct
// submission; returns false, recording nothing, when the pass must take the WebGPU path.
bool encode_pass_resources(const wgpu::RenderPassEncoder& encoder, detail::RenderPass& pass,
                           std::string_view label);
// Frame end, before the presentation callback submits: plans the frame's passes from the recorded
// packet. For a frame recorded into mapped streams, uploads the streams to the WebGPU buffers when
// any pass with GX draws takes the WebGPU path (or an encoder task may read them).
void prepare_frame(detail::FramePacket& frame);
// Encoder tasks that never read the streaming buffers (they only touch their own targets) do not
// force that upload.
void register_stream_independent_task(EncoderTaskId type);
// Around the frame's queue submit: the render pass callback is installed only while the frame's
// command buffer executes.
void install_frame();
void uninstall_frame();
void shutdown();

// Persistently mapped GL streams (glesMappedStreams). One slot per staging buffer; a slot is written
// by the FIFO processor while its frame is recorded and reused only after the fence created behind
// that frame's submission has signalled.
struct MappedSlot {
  unsigned uniforms = 0; // GL buffer names
  unsigned indices = 0;
  unsigned vertices = 0;
  uint8_t* uniformData = nullptr;
  uint8_t* indexData = nullptr;
  uint8_t* vertexData = nullptr;
  void* fence = nullptr; // GLsync of the last frame that used the slot
};
// Creates and maps the slots (through the interop, from the render worker). Leaves the streams staged
// when GL_EXT_buffer_storage is unavailable.
void create_mapped_slots(size_t count);
bool mapped_slots_ready() noexcept;
const MappedSlot* mapped_slot(size_t stagingSlot) noexcept;
// Null when the frame was not recorded into mapped storage.
const MappedSlot* mapped_slot(const detail::FramePacket& frame) noexcept;
// Before submit: makes the FIFO processor's writes visible to the GPU (explicitly flushed mapping).
void flush_mapped_slot(const detail::FramePacket& frame);
// After submit: fences the slot; done() polls (or waits for) that fence.
void fence_mapped_slot(size_t stagingSlot);
bool mapped_slot_fence_done(size_t stagingSlot, bool block);
} // namespace aurora::gfx::gles_direct
