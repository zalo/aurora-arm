#pragma once

#include "types.hpp"

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;
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
  wgpu::Limits limits;
  AuroraStats stats{};
};

Resources& resources() noexcept;
} // namespace detail
} // namespace aurora::gfx
