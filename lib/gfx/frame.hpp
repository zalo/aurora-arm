#pragma once

#include "resources.hpp"
#include "frame_packet.hpp"

#include <optional>
#include <utility>

namespace aurora::gfx::detail {

inline constexpr size_t FrameSlotCount = 2;
#ifdef MELEE_MIYOO_FLIP
inline constexpr size_t StagingBufferCount = FrameSlotCount + 1; // 1 GiB device: three 38 MiB staging maps
#else
inline constexpr size_t StagingBufferCount = FrameSlotCount + 3;
#endif
inline constexpr uint64_t StagingBufferSize = UniformBufferSize + VertexBufferSize + IndexBufferSize +
                                              StorageBufferSize + (UseTextureBuffer ? TextureUploadSize : 0);

// Where each stream lives in a staging buffer. When the frame streams are recorded into persistently
// mapped GL storage (gles_direct mapped streams, decided in initialize()) the vertex, uniform and index
// regions are never written and are left out (streams = false): Mesa/Panfrost backs every page of a
// buffer when it is created, so on a 1 GiB device the three unused 22 MiB regions cost 66 MiB of pinned
// memory that pushed the RK3326 and RK3566 into swap-less thrashing.
struct StagingLayout {
  uint64_t vertex = 0;
  uint64_t uniform = 0;
  uint64_t index = 0;
  uint64_t storage = 0;
  uint64_t textureUpload = 0;
  uint64_t size = 0;
  bool streams = true;
};
const StagingLayout& staging_layout() noexcept;

const wgpu::Buffer& staging_buffer(size_t slot);

struct RegisteredDrawType {
  DrawCallback draw = nullptr;
  void* userdata = nullptr;
};

struct RegisteredEncoderTaskType {
  EncoderTaskCallback callback = nullptr;
  void* userdata = nullptr;
  EncoderTaskCompletionCallback afterSubmit = nullptr;
};

std::optional<RegisteredDrawType> find_runtime_draw_type(DrawTypeId id);
std::optional<RegisteredEncoderTaskType> find_runtime_encoder_task_type(EncoderTaskId id);

} // namespace aurora::gfx::detail

namespace aurora::gfx {
void initialize();
void shutdown();
bool begin_frame();
void end_frame(EndFrameCallback callback);
// Asynchronous frames: the producer only reserves the frame; recording begins and ends on
// the FIFO processor when it reaches the frame's GX_AURORA_FRAME_BEGIN / FRAME_END markers.
// The producer stores the presentation callback with defer_end_frame() before writing the
// end marker; end_deferred_frame() hands it to the render worker.
bool reserve_frame(uint32_t& frameSlot);
void begin_reserved_frame(uint32_t frameSlot);
void defer_end_frame(EndFrameCallback callback);
void end_deferred_frame();
uint32_t current_frame() noexcept;
// Scene on surface: the presented texture the frame's EFB passes rendered into, set by the render worker
// before the presentation callback runs and taken by it (empty when the scene rendered into the EFB).
std::pair<wgpu::Texture, wgpu::TextureView> take_frame_surface() noexcept;
void after_submit() noexcept;
void gpu_synchronize();
void after_present() noexcept;
float calculate_fps() noexcept;
} // namespace aurora::gfx
