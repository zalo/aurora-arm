#pragma once

#include "types.hpp"

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
inline constexpr uint64_t VertexBufferSize = 5242880;   // 5 MiB
inline constexpr uint64_t IndexBufferSize = 2097152;    // 2 MiB
inline constexpr uint64_t StorageBufferSize = 8388608;  // 8 MiB
inline constexpr uint64_t TextureUploadSize = 25165824; // 24 MiB
#endif

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
