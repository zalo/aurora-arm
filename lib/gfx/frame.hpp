#pragma once

#include "resources.hpp"
#include "frame_packet.hpp"

#include <optional>

namespace aurora::gfx::detail {

inline constexpr size_t FrameSlotCount = 2;
#ifdef MELEE_MIYOO_FLIP
inline constexpr size_t StagingBufferCount = FrameSlotCount + 1; // 1 GiB device: three 38 MiB staging maps
#else
inline constexpr size_t StagingBufferCount = FrameSlotCount + 3;
#endif
inline constexpr uint64_t StagingBufferSize = UniformBufferSize + VertexBufferSize + IndexBufferSize +
                                              StorageBufferSize + (UseTextureBuffer ? TextureUploadSize : 0);

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
void after_submit() noexcept;
void gpu_synchronize();
void after_present() noexcept;
float calculate_fps() noexcept;
} // namespace aurora::gfx
