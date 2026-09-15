// Instanced GX_POINTS sprites with CPU vertex decoding
//
// A point draw keeps one decoded record per point and renders that many instances of a shared
// six-index quad; consecutive point draws with unchanged state merge by adding instances. Lines
// and triangles keep their existing paths, as does the storage-buffer mode.

#include "gx_test_common.hpp"
#include "gx/pipeline.hpp"
#include "gx/vertex_loader.hpp"

#include <array>
#include <cstring>
#include <vector>

namespace aurora::gfx {
extern gx::DrawData g_testLastDraw;
extern uint32_t g_testDrawCount;
extern bool g_testMergeDraws;
extern std::vector<uint8_t> g_testVertexStream;
extern std::vector<uint8_t> g_testIndexStream;
} // namespace aurora::gfx

namespace {
constexpr std::array<u16, 6> kQuadIndices{0, 1, 3, 3, 2, 0};

// Loader for the draws below: one direct GX_F32 XYZ position
const aurora::gx::VertexLoader& loader_for(u8 lineMode) {
  aurora::gx::ShaderConfig config{};
  config.cpuVertexDecode = true;
  config.lineMode = lineMode;
  config.attrs[GX_VA_POS] = aurora::gx::AttrConfig{.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32, .le = false};
  config.vtxStride = 12;
  return aurora::gx::vertex_loader(config);
}

std::array<float, 3> expected_position(u32 i) { return {float(i), float(i) + 0.5f, -float(i)}; }

std::array<float, 3> record_position(const aurora::gx::VertexLoader& loader, aurora::gfx::Range range, u32 index) {
  std::array<float, 3> pos{};
  const size_t offset = range.offset + index * loader.layout.stride + loader.posOffset;
  std::memcpy(pos.data(), aurora::gfx::g_testVertexStream.data() + offset, sizeof(pos));
  return pos;
}

std::vector<u16> pushed_indices(aurora::gfx::Range range) {
  std::vector<u16> indices(range.size / sizeof(u16));
  std::memcpy(indices.data(), aurora::gfx::g_testIndexStream.data() + range.offset, range.size);
  return indices;
}

class GXPointSpriteTest : public GXFifoTest {
protected:
  void SetUp() override {
    GXFifoTest::SetUp();
    aurora::g_config.cpuVertexDecode = true;
    aurora::gfx::g_testMergeDraws = true;
    aurora::gfx::g_testVertexStream.clear();
    aurora::gfx::g_testIndexStream.clear();
    // Direct GX_F32 positions, encoded then decoded through the command processor
    GXClearVtxDesc();
    GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
    decode_fifo(flush_and_capture());
  }

  void TearDown() override {
    aurora::gfx::g_testMergeDraws = false;
    aurora::g_config.cpuVertexDecode = false;
    GXFifoTest::TearDown();
  }

  // Encodes one primitive whose vertices are expected_position(first) .. expected_position(first + count - 1)
  void encode_primitive(GXPrimitive prim, u16 count, u32 first = 0) {
    GXBegin(prim, GX_VTXFMT0, count);
    for (u32 i = first; i < first + count; ++i) {
      const auto pos = expected_position(i);
      GXPosition3f32(pos[0], pos[1], pos[2]);
    }
    GXEnd();
  }

  // Decodes the encoded draws; returns the last draw command pushed
  const aurora::gx::DrawData& decode_draws() {
    aurora::gfx::g_testDrawCount = 0;
    decode_fifo(capture_fifo());
    return aurora::gfx::g_testLastDraw;
  }
};
} // namespace

TEST_F(GXPointSpriteTest, PointsDrawOneInstancePerRecordOverSharedQuad) {
  encode_primitive(GX_POINTS, 3);
  const auto& draw = decode_draws();

  EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
  EXPECT_EQ(draw.vtxCount, 3u);
  EXPECT_EQ(draw.instanceCount, 3u);
  EXPECT_EQ(draw.indexCount, 6u);
  EXPECT_EQ(pushed_indices(draw.idxRange), std::vector<u16>(kQuadIndices.begin(), kQuadIndices.end()));

  const auto& loader = loader_for(3);
  EXPECT_EQ(loader.lineEndOffset, UINT16_MAX); // points carry no line end
  ASSERT_EQ(draw.vertRange.size, 3u * loader.layout.stride);
  for (u32 i = 0; i < 3; ++i) {
    EXPECT_EQ(record_position(loader, draw.vertRange, i), expected_position(i)) << "point " << i;
  }
}

TEST_F(GXPointSpriteTest, AdjacentPointDrawsMergeByAddingInstances) {
  encode_primitive(GX_POINTS, 2);
  encode_primitive(GX_POINTS, 3, 2);
  const auto& draw = decode_draws();

  EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
  EXPECT_EQ(draw.vtxCount, 5u);
  EXPECT_EQ(draw.instanceCount, 5u);
  EXPECT_EQ(draw.indexCount, 6u);
  // The merge added records and instances, not indices
  EXPECT_EQ(aurora::gfx::g_testIndexStream.size(), kQuadIndices.size() * sizeof(u16));
  EXPECT_EQ(pushed_indices(draw.idxRange), std::vector<u16>(kQuadIndices.begin(), kQuadIndices.end()));

  const auto& loader = loader_for(3);
  ASSERT_EQ(draw.vertRange.size, 5u * loader.layout.stride);
  for (u32 i = 0; i < 5; ++i) {
    EXPECT_EQ(record_position(loader, draw.vertRange, i), expected_position(i)) << "point " << i;
  }
}

TEST_F(GXPointSpriteTest, PointDrawsDoNotMergeAcrossStateChanges) {
  encode_primitive(GX_POINTS, 2);
  GXSetPointSize(8, GX_TO_ZERO); // uniform state changes between the draws
  encode_primitive(GX_POINTS, 3, 2);
  const auto& draw = decode_draws();

  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
  EXPECT_EQ(draw.vtxCount, 3u);
  EXPECT_EQ(draw.instanceCount, 3u);
  EXPECT_EQ(draw.indexCount, 6u);
}

TEST_F(GXPointSpriteTest, LinesAndTrianglesKeepTheirPaths) {
  // Lines are still expanded on the CPU: four records per segment, dense indices, one instance
  encode_primitive(GX_LINES, 4);
  const auto& lines = decode_draws();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
  EXPECT_EQ(lines.vtxCount, 8u);
  EXPECT_EQ(lines.instanceCount, 1u);
  EXPECT_EQ(lines.indexCount, 12u);
  EXPECT_EQ(lines.vertRange.size, 8u * loader_for(1).layout.stride);

  // Triangles: one record per vertex, no index buffer, one instance
  encode_primitive(GX_TRIANGLES, 3);
  const auto& triangles = decode_draws();
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
  EXPECT_EQ(triangles.vtxCount, 3u);
  EXPECT_EQ(triangles.instanceCount, 1u);
  EXPECT_EQ(triangles.indexCount, 0u);
  EXPECT_EQ(triangles.vertRange.size, 3u * loader_for(0).layout.stride);
}

TEST_F(GXPointSpriteTest, StorageBufferModeIsUnchanged) {
  aurora::g_config.cpuVertexDecode = false;
  encode_primitive(GX_POINTS, 3);
  encode_primitive(GX_POINTS, 2, 3);
  const auto& draw = decode_draws();

  // Raw GX vertices, one instance per point over the quad, and no merging of point draws
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
  EXPECT_EQ(draw.vtxCount, 2u);
  EXPECT_EQ(draw.instanceCount, 2u);
  EXPECT_EQ(draw.indexCount, 6u);
  EXPECT_EQ(draw.vertRange.size, 2u * 12u);
  EXPECT_EQ(pushed_indices(draw.idxRange), std::vector<u16>(kQuadIndices.begin(), kQuadIndices.end()));
}
