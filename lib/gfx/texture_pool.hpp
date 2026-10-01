#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// Allocation policy of the GX texture layer pool (texture.cpp): plain
// bookkeeping with no GPU dependencies so it can be unit tested.
namespace aurora::gfx::texture_pool {
// A slab (one 2D array texture) stops growing at this many layers or bytes.
constexpr uint32_t MaxSlabLayers = 32;
constexpr uint64_t MaxSlabBytes = 2u << 20;

// How far slabs grow. A slab is one GPU allocation that is only released as a whole, so the layers
// a class once needed stay allocated while a single texture of the slab is alive: on a 1 GiB
// handheld the default growth backed 58 MiB of cached textures with 155 MiB of slabs after a dozen
// matches. The small limits trade some bind-group sharing for slabs that drain.
struct SlabLimits {
  uint32_t maxLayers = MaxSlabLayers;
  uint64_t maxBytes = MaxSlabBytes;
  uint32_t maxAtlasLayers = 8;
};
constexpr SlabLimits DefaultSlabLimits{};
constexpr SlabLimits SmallSlabLimits{.maxLayers = 8, .maxBytes = 512u << 10, .maxAtlasLayers = 2};

// Layers of the next slab of a texture class, given the layer count of the
// class's newest slab (0 for the first) and the bytes of one layer.
//
// Only textures in the same slab let adjacent draws share a bind group, so the
// first slab starts with room for neighbours, scaled by texture size; later
// slabs double. Memory then tracks the textures actually in use: on a 1 GiB
// handheld (Miyoo Flip), 32 layers per class up front cost 670 MB of free RAM,
// this policy 30-50 MB.
constexpr uint32_t next_slab_layers(uint32_t width, uint32_t height, uint32_t previousSlabLayers, uint64_t layerBytes,
                                    const SlabLimits& limits = DefaultSlabLimits) noexcept {
  uint32_t layers = 1;
  if (previousSlabLayers != 0) {
    layers = std::min(previousSlabLayers * 2, limits.maxLayers);
  } else {
    const uint64_t area = static_cast<uint64_t>(width) * height;
    layers = area <= 64 * 64 ? 8 : area <= 128 * 128 ? 4 : area <= 256 * 256 ? 2 : 1;
  }
  const uint64_t byBudget =
      std::clamp<uint64_t>(limits.maxBytes / std::max<uint64_t>(layerBytes, 1), 1, limits.maxLayers);
  return static_cast<uint32_t>(std::min<uint64_t>(layers, byBudget));
}

// A slab with no layer in use is released after this many frames. Without that the pool only ever grows: every
// size class keeps the most layers it ever held at once (a movie's frames, the previous stage's textures), which on a
// 1 GiB handheld is memory the next scene needs.
constexpr uint64_t SlabIdleFrames = 120;
constexpr uint64_t SlabTrimPeriod = 16;

// Atlas layers for single-mip textures that clamp on both axes: squares of
// AtlasSize texels, each texture in a cell surrounded by AtlasGutter texels of
// replicated edge, so that clamp filtering inside the cell matches the
// standalone texture. Atlas slabs double per class up to SlabLimits::maxAtlasLayers.
constexpr uint32_t AtlasSize = 1024;
constexpr uint32_t AtlasGutter = 1;
constexpr uint32_t MaxAtlasSlabLayers = DefaultSlabLimits.maxAtlasLayers;

constexpr uint32_t next_atlas_slab_layers(uint32_t previousSlabLayers,
                                          const SlabLimits& limits = DefaultSlabLimits) noexcept {
  return previousSlabLayers == 0 ? 1 : std::min(previousSlabLayers * 2, limits.maxAtlasLayers);
}

// Shelf packer for one atlas layer: cells (texture plus gutter) go left to
// right on a shelf of similar height, new shelves open below. Cells never
// move, but a released cell is free for the next texture that fits it and an
// emptied shelf gives its rows back, so a layer that keeps a few long-lived
// textures does not stay full of dead cells.
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
    // A shelf of similar height: a released cell first, then the unused tail.
    for (auto& shelf : m_shelves) {
      if (shelf.cells.empty() || cellHeight > shelf.height || cellHeight * 2 <= shelf.height) {
        continue;
      }
      if (take_cell(shelf, cellWidth, x)) {
        y = shelf.y + AtlasGutter;
        ++m_live;
        return true;
      }
    }
    // The tightest emptied shelf, cut down to this cell's height.
    size_t best = m_shelves.size();
    for (size_t i = 0; i < m_shelves.size(); ++i) {
      const auto& shelf = m_shelves[i];
      if (shelf.cells.empty() && shelf.height >= cellHeight &&
          (best == m_shelves.size() || shelf.height < m_shelves[best].height)) {
        best = i;
      }
    }
    if (best == m_shelves.size()) {
      if (m_nextY + cellHeight > AtlasSize) {
        return false;
      }
      m_shelves.push_back({.y = m_nextY, .height = cellHeight});
      m_nextY += cellHeight;
    } else if (m_shelves[best].height > cellHeight) {
      const Shelf rest{.y = m_shelves[best].y + cellHeight, .height = m_shelves[best].height - cellHeight};
      m_shelves[best].height = cellHeight;
      m_shelves.insert(m_shelves.begin() + static_cast<ptrdiff_t>(best) + 1, rest);
    }
    auto& shelf = m_shelves[best];
    shelf.cells.push_back({.x = 0, .width = cellWidth, .used = true});
    x = AtlasGutter;
    y = shelf.y + AtlasGutter;
    ++m_live;
    return true;
  }

  // Releases the cell of the texture placed at (x, y).
  void release(uint32_t x, uint32_t y) noexcept {
    const auto shelfIt = std::find_if(m_shelves.begin(), m_shelves.end(),
                                      [&](const Shelf& shelf) { return shelf.y + AtlasGutter == y; });
    if (shelfIt == m_shelves.end()) {
      return;
    }
    auto& cells = shelfIt->cells;
    auto cell = std::find_if(cells.begin(), cells.end(),
                             [&](const Cell& c) { return c.used && c.x + AtlasGutter == x; });
    if (cell == cells.end()) {
      return;
    }
    cell->used = false;
    --m_live;
    // Free neighbours become one cell; a free cell at the end goes back to the tail.
    if (cell + 1 != cells.end() && !(cell + 1)->used) {
      cell->width += (cell + 1)->width;
      cells.erase(cell + 1);
    }
    if (cell != cells.begin() && !(cell - 1)->used) {
      (cell - 1)->width += cell->width;
      cell = cells.erase(cell) - 1;
    }
    if (cell + 1 == cells.end()) {
      cells.erase(cell);
    }
    if (cells.empty()) {
      release_shelf(static_cast<size_t>(shelfIt - m_shelves.begin()));
    }
  }

  [[nodiscard]] uint32_t live() const noexcept { return m_live; }
  // Rows already given to shelves; rows below are still free.
  [[nodiscard]] uint32_t used_rows() const noexcept { return m_nextY; }

private:
  struct Cell {
    uint32_t x;
    uint32_t width;
    bool used;
  };
  struct Shelf {
    uint32_t y;
    uint32_t height;
    std::vector<Cell> cells; // left to right without gaps; empty for a shelf whose rows are free
  };

  static bool take_cell(Shelf& shelf, uint32_t cellWidth, uint32_t& x) noexcept {
    for (auto cell = shelf.cells.begin(); cell != shelf.cells.end(); ++cell) {
      if (cell->used || cell->width < cellWidth) {
        continue;
      }
      x = cell->x + AtlasGutter;
      if (cell->width > cellWidth) {
        const Cell rest{.x = cell->x + cellWidth, .width = cell->width - cellWidth, .used = false};
        cell->width = cellWidth;
        cell = shelf.cells.insert(cell + 1, rest) - 1;
      }
      cell->used = true;
      return true;
    }
    const uint32_t nextX = shelf.cells.back().x + shelf.cells.back().width;
    if (nextX + cellWidth > AtlasSize) {
      return false;
    }
    shelf.cells.push_back({.x = nextX, .width = cellWidth, .used = true});
    x = nextX + AtlasGutter;
    return true;
  }

  // The shelf at `index` lost its last cell: its rows join emptied neighbours, and emptied rows at
  // the bottom are unshelved.
  void release_shelf(size_t index) noexcept {
    if (index + 1 < m_shelves.size() && m_shelves[index + 1].cells.empty()) {
      m_shelves[index].height += m_shelves[index + 1].height;
      m_shelves.erase(m_shelves.begin() + static_cast<ptrdiff_t>(index) + 1);
    }
    if (index > 0 && m_shelves[index - 1].cells.empty()) {
      m_shelves[index - 1].height += m_shelves[index].height;
      m_shelves.erase(m_shelves.begin() + static_cast<ptrdiff_t>(index));
      --index;
    }
    if (index + 1 == m_shelves.size()) {
      m_nextY = m_shelves[index].y;
      m_shelves.pop_back();
    }
  }

  std::vector<Shelf> m_shelves; // top to bottom without gaps
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
