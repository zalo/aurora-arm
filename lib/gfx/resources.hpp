#pragma once

#include "types.hpp"

#include <array>
#include <atomic>

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;
#ifdef MELEE_MIYOO_FLIP
// Bounded pools for the Flip's 1 GiB shared CPU/GPU memory. The vertex pool is larger than
// upstream's because CPU-decoded float records are wider than the raw GX stream.
inline constexpr uint64_t UniformBufferSize = 8 * 1024 * 1024;
inline constexpr uint64_t VertexBufferSize = 12 * 1024 * 1024;
inline constexpr uint64_t IndexBufferSize = 2 * 1024 * 1024;
inline constexpr uint64_t StorageBufferSize = 4 * 1024 * 1024;
inline constexpr uint64_t TextureUploadSize = 12 * 1024 * 1024;
#else
inline constexpr uint64_t UniformBufferSize = 25165824; // 24 MiB
#ifdef AURORA_VERTEX_BUFFER_SIZE
// Set from the AURORA_VERTEX_BUFFER_MIB CMake variable. CPU-decoded vertex records (cpuVertexDecode)
// are wider than raw GX vertices; titles that stream heavy geometry without resident display lists
// need more than the 5 MiB default.
inline constexpr uint64_t VertexBufferSize = AURORA_VERTEX_BUFFER_SIZE;
#else
inline constexpr uint64_t VertexBufferSize = 5242880;   // 5 MiB
#endif
inline constexpr uint64_t IndexBufferSize = 2097152;    // 2 MiB
inline constexpr uint64_t StorageBufferSize = 8388608;  // 8 MiB
inline constexpr uint64_t TextureUploadSize = 25165824; // 24 MiB
#endif

// The sizes in use. They are the constants above unless AURORA_STREAM_KB=<uniform>,<vertex>,<index>,<storage>,
// <texture upload> (KiB each, any separator, 0 keeps a default) overrides them: every stream exists once per staging slot and
// once more as a Dawn buffer, so on a 1 GiB device each MiB here is several MiB of GPU memory.
struct StreamSizes {
  uint64_t uniform = UniformBufferSize;
  uint64_t vertex = VertexBufferSize;
  uint64_t index = IndexBufferSize;
  uint64_t storage = StorageBufferSize;
  uint64_t textureUpload = TextureUploadSize;
};
const StreamSizes& stream_sizes() noexcept;

// What the frames recorded so far needed of each stream, for sizing them (AURORA_MEM_LOG).
struct StreamUsage {
  std::atomic<uint64_t> frames{0};
  std::atomic<uint64_t> maxUniform{0};
  std::atomic<uint64_t> maxVertex{0};
  std::atomic<uint64_t> maxIndex{0};
  std::atomic<uint64_t> maxStorage{0};
  std::atomic<uint64_t> maxTextureUpload{0};
  std::atomic<uint64_t> overflowUploads{0}; // texture uploads that did not fit the stream and got a buffer of their own
  std::atomic<uint64_t> overflowUploadBytes{0};
  std::atomic<uint64_t> droppedPushes{0}; // draw data refused by a full stream
  // Frames whose mapped GL streams were also written into Dawn's buffers (gles_direct), and the bytes written.
  std::atomic<uint64_t> dawnStreamFrames{0};
  std::atomic<uint64_t> dawnStreamBytes{0};
  // Why: passes with GX draws that Dawn replayed (a draw or pipeline the direct path does not take) and passes
  // with custom draws.
  std::atomic<uint64_t> dawnGxPasses{0};
  std::atomic<uint64_t> dawnCustomPasses{0};
  // What kept those GX passes from the direct path; a pass counts once per reason it has.
  enum DawnPassReason : size_t {
    Multisample,  // MSAA or more than one colour attachment
    OtherDraw,    // a draw that is neither GX nor a clear
    NoConfig,     // a pipeline whose configuration is unknown
    VertexPath,   // a pipeline without CPU vertex decode or draw batching
    FogRange,     // range-adjusted fog
    OffsetClamp,  // polygon offset clamp
    EncoderTask,  // a frame task that reads the streams through Dawn (not a pass)
    DawnPassReasonCount,
  };
  std::array<std::atomic<uint64_t>, DawnPassReasonCount> dawnPassReasons{};
  std::atomic<uint64_t> pendingPipelineDraws{0}; // direct draws skipped because their pipeline was not ready
};
StreamUsage& stream_usage() noexcept;

namespace detail {
struct Resources {
  wgpu::Buffer vertexBuffer;
  wgpu::Buffer uniformBuffer;
  wgpu::Buffer indexBuffer;
  wgpu::Buffer storageBuffer;
  wgpu::BindGroupLayout staticBindGroupLayout;
  wgpu::BindGroup staticBindGroup;
  wgpu::BindGroupLayout uniformBindGroupLayout;
  wgpu::BindGroup uniformBindGroup;
  // Uniform table: the uniform buffer bound as 64 KiB windows (gx::UniformWindowSize) at dynamic offsets.
  wgpu::BindGroup uniformWindowBindGroup;
  wgpu::Limits limits;
  AuroraStats stats{};
};

Resources& resources() noexcept;
} // namespace detail
} // namespace aurora::gfx
