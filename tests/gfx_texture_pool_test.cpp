#include "gfx/texture.hpp"
#include "gfx/texture_pool.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

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

  auto lodClamped = make_texture_object(GX_CLAMP, GX_CLAMP);
  lodClamped.mode1 = 0x2000; // max LOD 2.0
  EXPECT_NE(texture_class(lodClamped).sampler, clamp.sampler);
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
} // namespace
} // namespace aurora::gfx
