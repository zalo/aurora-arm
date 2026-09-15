#pragma once

#include "types.hpp"

namespace aurora::gfx {

BindGroupRef bind_group_ref(const WGPUBindGroupDescriptor& descriptor);
wgpu::BindGroup find_bind_group(BindGroupRef id);
wgpu::Sampler sampler_ref(const wgpu::SamplerDescriptor& descriptor);

namespace detail {
void clear_bind_group_cache();
void expire_cached_bind_groups();
// Increments whenever a cached bind group is destroyed; caches derived from a BindGroupRef must be dropped
// when it changes.
uint64_t bind_group_cache_generation() noexcept;
void shutdown_resource_cache();
} // namespace detail

} // namespace aurora::gfx
