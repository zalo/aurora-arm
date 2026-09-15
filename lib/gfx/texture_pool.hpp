#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

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

// Atlas layers for single-mip textures that clamp on both axes: squares of
// AtlasSize texels, each texture in a cell surrounded by AtlasGutter texels of
// replicated edge, so that clamp filtering inside the cell matches the
// standalone texture. Atlas slabs double per class up to MaxAtlasSlabLayers.
constexpr uint32_t AtlasSize = 1024;
constexpr uint32_t AtlasGutter = 1;
constexpr uint32_t MaxAtlasSlabLayers = 8;

constexpr uint32_t next_atlas_slab_layers(uint32_t previousSlabLayers) noexcept {
  return previousSlabLayers == 0 ? 1 : std::min(previousSlabLayers * 2, MaxAtlasSlabLayers);
}

// Shelf packer for one atlas layer: cells (texture plus gutter) go left to
// right on a shelf of similar height, new shelves open below. Cells never
// move; the layer is reused as a whole once every cell has been released.
class ShelfPacker {
public:
  // Places a cell for a width x height texture. On success (x, y) is the texel
  // origin of the texture itself, inside its gutter.
  bool place(uint32_t width, uint32_t height, uint32_t& x, uint32_t& y) noexcept {
    const uint32_t cellWidth = width + 2 * AtlasGutter;
    const uint32_t cellHeight = height + 2 * AtlasGutter;
    if (cellWidth > AtlasSize || cellHeight > AtlasSize) {
      return false;
    }
    for (auto& shelf : m_shelves) {
      if (cellHeight <= shelf.height && cellHeight * 2 > shelf.height && shelf.nextX + cellWidth <= AtlasSize) {
        x = shelf.nextX + AtlasGutter;
        y = shelf.y + AtlasGutter;
        shelf.nextX += cellWidth;
        ++m_live;
        return true;
      }
    }
    if (m_nextY + cellHeight > AtlasSize) {
      return false;
    }
    m_shelves.push_back({.y = m_nextY, .height = cellHeight, .nextX = cellWidth});
    x = AtlasGutter;
    y = m_nextY + AtlasGutter;
    m_nextY += cellHeight;
    ++m_live;
    return true;
  }

  // Releases one cell; the layer is empty again when none is left.
  void release() noexcept {
    if (m_live > 0 && --m_live == 0) {
      m_shelves.clear();
      m_nextY = 0;
    }
  }

  [[nodiscard]] uint32_t live() const noexcept { return m_live; }

private:
  struct Shelf {
    uint32_t y;
    uint32_t height;
    uint32_t nextX;
  };
  std::vector<Shelf> m_shelves;
  uint32_t m_nextY = 0;
  uint32_t m_live = 0;
};

// Copies a width x height image into `out` with AtlasGutter texels of
// replicated edge around it and returns the padded row pitch in bytes.
inline uint32_t pad_with_gutter(const uint8_t* src, uint32_t srcBytesPerRow, uint32_t width, uint32_t height,
                                uint32_t bytesPerTexel, std::vector<uint8_t>& out) {
  const uint32_t paddedWidth = width + 2 * AtlasGutter;
  const uint32_t paddedHeight = height + 2 * AtlasGutter;
  const uint32_t outBytesPerRow = paddedWidth * bytesPerTexel;
  out.resize(static_cast<size_t>(outBytesPerRow) * paddedHeight);
  for (uint32_t y = 0; y < paddedHeight; ++y) {
    const uint32_t srcY = y < AtlasGutter ? 0 : y >= height + AtlasGutter ? height - 1 : y - AtlasGutter;
    const uint8_t* srcRow = src + static_cast<size_t>(srcY) * srcBytesPerRow;
    uint8_t* dstRow = out.data() + static_cast<size_t>(y) * outBytesPerRow;
    std::memcpy(dstRow + AtlasGutter * bytesPerTexel, srcRow, static_cast<size_t>(width) * bytesPerTexel);
    for (uint32_t g = 0; g < AtlasGutter; ++g) {
      std::memcpy(dstRow + g * bytesPerTexel, srcRow, bytesPerTexel);
      std::memcpy(dstRow + (width + AtlasGutter + g) * bytesPerTexel,
                  srcRow + static_cast<size_t>(width - 1) * bytesPerTexel, bytesPerTexel);
    }
  }
  return outBytesPerRow;
}
} // namespace aurora::gfx::texture_pool
