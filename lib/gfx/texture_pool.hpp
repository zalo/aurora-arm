#pragma once

#include <algorithm>
#include <cstdint>

// Allocation policy of the GX texture layer pool (texture.cpp): plain
// bookkeeping with no GPU dependencies so it can be unit tested.
namespace aurora::gfx::texture_pool {
// A slab (one 2D array texture) stops growing at this many layers or bytes.
constexpr uint32_t MaxSlabLayers = 32;
constexpr uint64_t MaxSlabBytes = 2u << 20;

// Layers of the next slab of a texture class, given the layer count of the
// class's newest slab (0 for the first) and the bytes of one layer.
//
// Only textures in the same slab let adjacent draws share a bind group, so the
// first slab starts with room for neighbours, scaled by texture size; later
// slabs double. Memory then tracks the textures actually in use: on a 1 GiB
// handheld (Miyoo Flip), 32 layers per class up front cost 670 MB of free RAM,
// this policy 30-50 MB.
constexpr uint32_t next_slab_layers(uint32_t width, uint32_t height, uint32_t previousSlabLayers,
                                    uint64_t layerBytes) noexcept {
  uint32_t layers = 1;
  if (previousSlabLayers != 0) {
    layers = std::min(previousSlabLayers * 2, MaxSlabLayers);
  } else {
    const uint64_t area = static_cast<uint64_t>(width) * height;
    layers = area <= 64 * 64 ? 8 : area <= 128 * 128 ? 4 : area <= 256 * 256 ? 2 : 1;
  }
  const uint64_t byBudget = std::clamp<uint64_t>(MaxSlabBytes / std::max<uint64_t>(layerBytes, 1), 1, MaxSlabLayers);
  return static_cast<uint32_t>(std::min<uint64_t>(layers, byBudget));
}
} // namespace aurora::gfx::texture_pool
