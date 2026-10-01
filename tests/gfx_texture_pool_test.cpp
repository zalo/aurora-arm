#include "gfx/texture.hpp"
#include "gfx/texture_pool.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace aurora::gfx {
namespace {
constexpr uint64_t Rgba8Bytes = 4;

uint64_t layer_bytes(uint32_t width, uint32_t height, uint32_t mips = 1) {
  uint64_t total = 0;
  for (uint32_t mip = 0; mip < mips; ++mip) {
    total += static_cast<uint64_t>(std::max(width >> mip, 1u)) * std::max(height >> mip, 1u) * Rgba8Bytes;
  }
  return total;
}

GXTexObj_ make_texture_object(GXTexWrapMode wrapS, GXTexWrapMode wrapT, GXTexFilter minFilter = GX_LINEAR,
                              GXTexFilter magFilter = GX_LINEAR, GXAnisotropy aniso = GX_ANISO_1) {
  // mode0 layout of GXInitTexObj: wrap_s[0:1], wrap_t[2:3], mag[4], min[5:7], aniso[19:20]
  constexpr u32 kGxToHwFilter[] = {0, 4, 1, 2, 5, 6};
  GXTexObj_ obj{};
  obj.mode0 = static_cast<u32>(wrapS) | (static_cast<u32>(wrapT) << 2) | ((magFilter == GX_LINEAR ? 1u : 0u) << 4) |
              (kGxToHwFilter[minFilter] << 5) | (static_cast<u32>(aniso) << 19);
  return obj;
}

TEST(TexturePoolTest, FirstSlabStartsWithRoomForNeighboursBySize) {
  EXPECT_EQ(texture_pool::next_slab_layers(16, 16, 0, layer_bytes(16, 16)), 8u);
  EXPECT_EQ(texture_pool::next_slab_layers(64, 64, 0, layer_bytes(64, 64)), 8u);
  EXPECT_EQ(texture_pool::next_slab_layers(128, 128, 0, layer_bytes(128, 128)), 4u);
  EXPECT_EQ(texture_pool::next_slab_layers(256, 256, 0, layer_bytes(256, 256)), 2u);
  EXPECT_EQ(texture_pool::next_slab_layers(512, 512, 0, layer_bytes(512, 512)), 1u);
}

TEST(TexturePoolTest, SlabsDoubleUpToTheLayerCap) {
  const uint64_t bytes = layer_bytes(16, 16);
  uint32_t layers = texture_pool::next_slab_layers(16, 16, 0, bytes);
  EXPECT_EQ(layers, 8u);
  layers = texture_pool::next_slab_layers(16, 16, layers, bytes);
  EXPECT_EQ(layers, 16u);
  layers = texture_pool::next_slab_layers(16, 16, layers, bytes);
  EXPECT_EQ(layers, 32u);
  layers = texture_pool::next_slab_layers(16, 16, layers, bytes);
  EXPECT_EQ(layers, texture_pool::MaxSlabLayers);
}

TEST(TexturePoolTest, SlabsStayUnderTheByteBudget) {
  // 256x256 RGBA8 is 256 KiB per layer: 8 layers fill the 2 MiB budget.
  const uint64_t bytes = layer_bytes(256, 256);
  EXPECT_EQ(texture_pool::next_slab_layers(256, 256, 4, bytes), 8u);
  EXPECT_EQ(texture_pool::next_slab_layers(256, 256, 8, bytes), 8u);
  EXPECT_EQ(texture_pool::next_slab_layers(256, 256, 16, bytes), 8u);
  // A layer larger than the budget still gets a slab of one.
  EXPECT_EQ(texture_pool::next_slab_layers(1024, 1024, 0, layer_bytes(1024, 1024)), 1u);
  EXPECT_EQ(texture_pool::next_slab_layers(1024, 1024, 1, layer_bytes(1024, 1024)), 1u);
  // Mip chains count toward the budget.
  EXPECT_EQ(texture_pool::next_slab_layers(128, 128, 32, layer_bytes(128, 128, 8)), 24u);
}

TEST(TexturePoolTest, ClassSeparatesWrapModesFiltersAndLodClamps) {
  const auto clamp = texture_class(make_texture_object(GX_CLAMP, GX_CLAMP));
  const auto repeat = texture_class(make_texture_object(GX_REPEAT, GX_REPEAT));
  const auto mixed = texture_class(make_texture_object(GX_CLAMP, GX_REPEAT));
  const auto nearest = texture_class(make_texture_object(GX_CLAMP, GX_CLAMP, GX_NEAR, GX_NEAR));
  const auto aniso = texture_class(make_texture_object(GX_CLAMP, GX_CLAMP, GX_LINEAR, GX_LINEAR, GX_ANISO_4));
  EXPECT_NE(clamp.sampler, repeat.sampler);
  EXPECT_NE(clamp.sampler, mixed.sampler);
  EXPECT_NE(clamp.sampler, nearest.sampler);
  EXPECT_NE(clamp.sampler, aniso.sampler);

  // LOD clamps reach the sampler of mipmapped and of non-clamp textures.
  auto lodClamped = make_texture_object(GX_REPEAT, GX_REPEAT);
  lodClamped.mode1 = 0x2000; // max LOD 2.0
  EXPECT_NE(texture_class(lodClamped).sampler, repeat.sampler);
}

TEST(TexturePoolTest, ClassIgnoresUniformOnlyState) {
  const auto base = make_texture_object(GX_CLAMP, GX_CLAMP);
  auto biased = base;
  biased.mode0 |= 0x20u << 9; // LOD bias +1.0 travels in the uniform record
  auto edgeLod = base;
  edgeLod.mode0 |= 1u << 8; // edge LOD, not sampler state
  EXPECT_EQ(texture_class(biased).sampler, texture_class(base).sampler);
  EXPECT_EQ(texture_class(edgeLod).sampler, texture_class(base).sampler);
}

TEST(TexturePoolTest, OnlySingleMipClampTexturesFormAtlasClasses) {
  const auto clamp = make_texture_object(GX_CLAMP, GX_CLAMP);
  EXPECT_TRUE(texture_class(clamp).atlas);
  EXPECT_FALSE(texture_class(make_texture_object(GX_REPEAT, GX_REPEAT)).atlas);
  EXPECT_FALSE(texture_class(make_texture_object(GX_CLAMP, GX_MIRROR)).atlas);

  auto mipmapped = make_texture_object(GX_CLAMP, GX_CLAMP, GX_LIN_MIP_LIN);
  mipmapped.flags |= 1;     // has mips
  mipmapped.mode1 = 0x2000; // max LOD 2.0: three levels
  EXPECT_FALSE(texture_class(mipmapped).atlas);

  // One level: LOD clamps do not reach the sampler, filters and anisotropy do.
  auto lodClamped = clamp;
  lodClamped.mode1 = 0x2000;
  EXPECT_EQ(texture_class(lodClamped).sampler, texture_class(clamp).sampler);
  EXPECT_NE(texture_class(make_texture_object(GX_CLAMP, GX_CLAMP, GX_NEAR)).sampler, texture_class(clamp).sampler);
  EXPECT_NE(texture_class(make_texture_object(GX_CLAMP, GX_CLAMP, GX_LIN_MIP_LIN)).sampler,
            texture_class(clamp).sampler);
  EXPECT_NE(texture_class(make_texture_object(GX_CLAMP, GX_CLAMP, GX_LINEAR, GX_LINEAR, GX_ANISO_4)).sampler,
            texture_class(clamp).sampler);
  // An atlas class never collides with the plain class of a REPEAT texture.
  EXPECT_NE(texture_class(make_texture_object(GX_REPEAT, GX_REPEAT)).sampler, texture_class(clamp).sampler);
}

TEST(TexturePoolTest, AtlasSlabsDoubleToTheirCap) {
  EXPECT_EQ(texture_pool::next_atlas_slab_layers(0), 1u);
  EXPECT_EQ(texture_pool::next_atlas_slab_layers(1), 2u);
  EXPECT_EQ(texture_pool::next_atlas_slab_layers(2), 4u);
  EXPECT_EQ(texture_pool::next_atlas_slab_layers(4), 8u);
  EXPECT_EQ(texture_pool::next_atlas_slab_layers(8), texture_pool::MaxAtlasSlabLayers);
}

TEST(TexturePoolTest, SmallLimitsKeepSlabsSmall) {
  const auto& limits = texture_pool::SmallSlabLimits;
  EXPECT_EQ(texture_pool::next_atlas_slab_layers(0, limits), 1u);
  EXPECT_EQ(texture_pool::next_atlas_slab_layers(1, limits), limits.maxAtlasLayers);
  EXPECT_EQ(texture_pool::next_atlas_slab_layers(limits.maxAtlasLayers, limits), limits.maxAtlasLayers);
  // 16x16 RGBA8 layers never go past the layer cap, 128x128 ones (64 KiB) fill the byte budget at 8.
  EXPECT_EQ(texture_pool::next_slab_layers(16, 16, 8, layer_bytes(16, 16), limits), limits.maxLayers);
  EXPECT_EQ(texture_pool::next_slab_layers(128, 128, 4, layer_bytes(128, 128), limits), 8u);
  EXPECT_EQ(texture_pool::next_slab_layers(256, 256, 0, layer_bytes(256, 256), limits), 2u);
  EXPECT_EQ(texture_pool::next_slab_layers(256, 256, 2, layer_bytes(256, 256), limits), 2u);
}

TEST(TexturePoolTest, ShelfPackerPlacesCellsWithGutters) {
  using texture_pool::AtlasGutter;
  using texture_pool::AtlasSize;
  texture_pool::ShelfPacker packer;
  uint32_t x = 0;
  uint32_t y = 0;
  ASSERT_TRUE(packer.place(100, 50, x, y));
  EXPECT_EQ(x, AtlasGutter);
  EXPECT_EQ(y, AtlasGutter);
  // Same shelf, one cell (100 + 2 gutter texels) to the right.
  ASSERT_TRUE(packer.place(100, 50, x, y));
  EXPECT_EQ(x, 100 + 2 * AtlasGutter + AtlasGutter);
  EXPECT_EQ(y, AtlasGutter);
  // Taller than the shelf: a new shelf opens below the first (50 + 2 gutter).
  ASSERT_TRUE(packer.place(60, 60, x, y));
  EXPECT_EQ(x, AtlasGutter);
  EXPECT_EQ(y, 50 + 2 * AtlasGutter + AtlasGutter);
  // Much shorter than both shelves: does not waste them, opens a third.
  ASSERT_TRUE(packer.place(8, 8, x, y));
  EXPECT_EQ(y, 50 + 2 * AtlasGutter + 60 + 2 * AtlasGutter + AtlasGutter);
  EXPECT_EQ(packer.live(), 4u);

  // The gutter is part of the cell: the largest texture is AtlasSize - 2.
  texture_pool::ShelfPacker full;
  EXPECT_FALSE(full.place(AtlasSize - 1, 8, x, y));
  ASSERT_TRUE(full.place(AtlasSize - 2 * AtlasGutter, AtlasSize - 2 * AtlasGutter, x, y));
  EXPECT_FALSE(full.place(1, 1, x, y));
}

TEST(TexturePoolTest, ShelfPackerReusesReleasedCells) {
  using texture_pool::AtlasGutter;
  texture_pool::ShelfPacker packer;
  uint32_t x = 0;
  uint32_t y = 0;
  std::array<std::array<uint32_t, 2>, 4> cells{};
  for (auto& cell : cells) {
    ASSERT_TRUE(packer.place(500, 500, cell[0], cell[1]));
  }
  EXPECT_FALSE(packer.place(500, 500, x, y));
  // A released cell takes the next texture that fits it, in place.
  packer.release(cells[1][0], cells[1][1]);
  EXPECT_EQ(packer.live(), 3u);
  ASSERT_TRUE(packer.place(500, 500, x, y));
  EXPECT_EQ(x, cells[1][0]);
  EXPECT_EQ(y, cells[1][1]);
  // A narrower texture leaves the rest of the cell free for another one.
  packer.release(cells[2][0], cells[2][1]);
  ASSERT_TRUE(packer.place(200, 480, x, y));
  EXPECT_EQ(x, cells[2][0]);
  EXPECT_EQ(y, cells[2][1]);
  ASSERT_TRUE(packer.place(200, 480, x, y));
  EXPECT_EQ(x, cells[2][0] + 200 + 2 * AtlasGutter);
  EXPECT_EQ(y, cells[2][1]);
  EXPECT_EQ(packer.live(), 5u);
}

TEST(TexturePoolTest, ShelfPackerGivesBackEmptiedShelves) {
  using texture_pool::AtlasGutter;
  texture_pool::ShelfPacker packer;
  uint32_t x = 0;
  uint32_t y = 0;
  std::array<std::array<uint32_t, 2>, 3> shelves{};
  for (auto& shelf : shelves) {
    ASSERT_TRUE(packer.place(100, 298, shelf[0], shelf[1]));
    ASSERT_TRUE(packer.place(900, 298, x, y));
  }
  const auto release_shelf = [&](size_t i) {
    packer.release(shelves[i][0], shelves[i][1]);
    packer.release(shelves[i][0] + 100 + 2 * AtlasGutter, shelves[i][1]);
  };
  EXPECT_EQ(packer.used_rows(), 900u);
  // An emptied shelf in the middle is cut to the height of the next texture; the rest stays free.
  release_shelf(1);
  EXPECT_EQ(packer.used_rows(), 900u);
  ASSERT_TRUE(packer.place(100, 98, x, y));
  EXPECT_EQ(y, shelves[1][1]);
  ASSERT_TRUE(packer.place(100, 198, x, y));
  EXPECT_EQ(y, shelves[1][1] + 100);
  // Emptied shelves at the bottom are unshelved, together with emptied rows above them.
  packer.release(x, y);
  release_shelf(2);
  EXPECT_EQ(packer.used_rows(), 400u);
  ASSERT_TRUE(packer.place(1000, 600, x, y));
  EXPECT_EQ(y, 400 + AtlasGutter);
  // The whole layer comes back once nothing is placed.
  packer.release(x, y);
  packer.release(AtlasGutter, shelves[1][1]);
  release_shelf(0);
  EXPECT_EQ(packer.live(), 0u);
  EXPECT_EQ(packer.used_rows(), 0u);
  ASSERT_TRUE(packer.place(1022, 1022, x, y));
}

TEST(TexturePoolTest, GutterReplicatesEdgeTexels) {
  // 2x2 RGBA8 image with one distinct texel per corner.
  const std::array<uint8_t, 16> image{
      0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, //
      0x30, 0x31, 0x32, 0x33, 0x40, 0x41, 0x42, 0x43, //
  };
  std::vector<uint8_t> padded;
  const uint32_t pitch = texture_pool::pad_with_gutter(image.data(), 8, 2, 2, 4, padded);
  ASSERT_EQ(pitch, 16u);
  ASSERT_EQ(padded.size(), 64u);
  const auto texel = [&](uint32_t x, uint32_t y) { return padded[y * pitch + x * 4]; };
  // Interior
  EXPECT_EQ(texel(1, 1), 0x10);
  EXPECT_EQ(texel(2, 1), 0x20);
  EXPECT_EQ(texel(1, 2), 0x30);
  EXPECT_EQ(texel(2, 2), 0x40);
  // Corners and edges replicate the nearest texel
  EXPECT_EQ(texel(0, 0), 0x10);
  EXPECT_EQ(texel(3, 0), 0x20);
  EXPECT_EQ(texel(0, 3), 0x30);
  EXPECT_EQ(texel(3, 3), 0x40);
  EXPECT_EQ(texel(1, 0), 0x10);
  EXPECT_EQ(texel(0, 2), 0x30);
  EXPECT_EQ(texel(3, 1), 0x20);
  EXPECT_EQ(texel(2, 3), 0x40);
  // Whole texels are copied
  EXPECT_EQ(padded[0 * pitch + 0 * 4 + 3], 0x13);
  EXPECT_EQ(padded[3 * pitch + 3 * 4 + 3], 0x43);
}
} // namespace
} // namespace aurora::gfx
