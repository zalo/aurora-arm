// Resident display-list geometry tests
//
// With AuroraConfig::residentDisplayLists set, GXCallDisplayList references lists instead of
// copying them into the FIFO. The command processor decodes each list once into an arena, later
// identical calls hit, a rewritten list decodes again, and releasing memory through
// GXInvalidateResidentGeometry removes exactly the entries depending on it, including entries that
// depend on the range only through a vertex array base.

#include "gx_test_common.hpp"

#include "dolphin/gx/GXAurora.h"
#include "gx/pipeline.hpp"
#include "gx/resident_geometry.hpp"
#include "gx/vertex_loader.hpp"

#include <algorithm>
#include <array>
#include <random>
#include <set>
#include <vector>

namespace aurora::gfx {
extern gx::DrawData g_testLastDraw;
extern uint32_t g_testDrawCount;
} // namespace aurora::gfx

using namespace aurora::gx;

namespace {

// Big-endian GX_S16 XYZ positions and GX_RGBA8 colors of one quad, indexed by GX_INDEX8.
struct Arrays {
  u8 positions[4 * 6] = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 0, 0, 0
      0x00, 0x64, 0x00, 0x00, 0x00, 0x00, // 100, 0, 0
      0x00, 0x64, 0x00, 0x64, 0x00, 0x00, // 100, 100, 0
      0xFF, 0x9C, 0x00, 0x64, 0x00, 0x00, // -100, 100, 0
  };
  u8 colors[4 * 4] = {
      0xFF, 0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x80,
  };
};

u8 op(GXPrimitive prim, GXVtxFmt fmt) { return static_cast<u8>(prim) | static_cast<u8>(fmt); }

// One GX_QUADS draw of (position index, color index) pairs, padded to 32 bytes with NOPs like the
// SDK's GXEndDisplayList.
std::vector<u8> quad_list(std::initializer_list<u8> vertices = {0, 0, 1, 1, 2, 2, 3, 3}) {
  std::vector<u8> dl{op(GX_QUADS, GX_VTXFMT0), 0, 4};
  dl.insert(dl.end(), vertices);
  dl.resize(32, GX_NOP);
  return dl;
}

bool contains(const std::vector<u8>& haystack, const std::vector<u8>& needle) {
  return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end()) != haystack.end();
}

bool has_aurora_cmd(const std::vector<u8>& bytes, u16 cmd) {
  return contains(bytes, {GX_AURORA, static_cast<u8>(cmd >> 8), static_cast<u8>(cmd & 0xFF)});
}

ShaderConfig quad_config() {
  ShaderConfig config{};
  config.cpuVertexDecode = true;
  config.vtxStride = 2;
  config.attrs[GX_VA_POS] = {
      .attrType = GX_INDEX8, .cnt = 3, .compType = GX_S16, .offset = 0, .stride = 6, .le = false};
  config.attrs[GX_VA_CLR0] = {
      .attrType = GX_INDEX8, .cnt = 4, .compType = GX_RGBA8, .offset = 1, .stride = 4, .le = false};
  return config;
}

std::array<AttrArray, MaxVtxAttr> quad_arrays(const Arrays& arrays) {
  std::array<AttrArray, MaxVtxAttr> out{};
  out[GX_VA_POS] = {.data = arrays.positions, .size = sizeof(arrays.positions), .stride = 6, .le = false};
  out[GX_VA_CLR0] = {.data = arrays.colors, .size = sizeof(arrays.colors), .stride = 4, .le = false};
  return out;
}

class GXResidentTest : public GXFifoTest {
protected:
  void SetUp() override {
    GXFifoTest::SetUp();
    resident::shutdown();
    aurora::g_config.cpuVertexDecode = true;
    aurora::g_config.residentDisplayLists = true;
    aurora::g_config.residentGeometryBudget = 0;
    aurora::gfx::g_testDrawCount = 0;
    aurora::gfx::g_testLastDraw = {};
  }

  void TearDown() override {
    resident::shutdown();
    aurora::g_config.cpuVertexDecode = false;
    aurora::g_config.residentDisplayLists = false;
    aurora::g_config.residentGeometryBudget = 0;
    GXFifoTest::TearDown();
  }

  void set_format() {
    GXClearVtxDesc();
    GXSetVtxDesc(GX_VA_POS, GX_INDEX8);
    GXSetVtxDesc(GX_VA_CLR0, GX_INDEX8);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_S16, 0);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
  }

  void set_arrays(const Arrays& arrays) {
    GXSetArray(GX_VA_POS, arrays.positions, sizeof(arrays.positions), 6, false);
    GXSetArray(GX_VA_CLR0, arrays.colors, sizeof(arrays.colors), 4, false);
  }

  // Encodes a call and runs everything written so far through the command processor.
  void call(const std::vector<u8>& dl) {
    GXCallDisplayList(dl.data(), static_cast<u32>(dl.size()));
    decode_fifo(flush_and_capture());
  }

  void release(const void* base, size_t size) {
    GXInvalidateResidentGeometry(base, static_cast<u32>(size));
    decode_fifo(flush_and_capture());
  }
};

} // namespace

TEST_F(GXResidentTest, CallsListsByReference) {
  set_format();
  const auto dl = quad_list();

  GXCallDisplayList(dl.data(), static_cast<u32>(dl.size()));
  auto bytes = flush_and_capture();
  EXPECT_TRUE(has_aurora_cmd(bytes, GX_AURORA_CALL_DL));
  EXPECT_FALSE(contains(bytes, dl));

  // Without the option the list is copied into the FIFO as before
  aurora::g_config.residentDisplayLists = false;
  GXCallDisplayList(dl.data(), static_cast<u32>(dl.size()));
  bytes = flush_and_capture();
  EXPECT_FALSE(has_aurora_cmd(bytes, GX_AURORA_CALL_DL));
  EXPECT_TRUE(contains(bytes, dl));
  EXPECT_FALSE(contains(bytes, {GX_AURORA}));

  // Releases are no-ops without the option
  GXInvalidateResidentGeometry(dl.data(), static_cast<u32>(dl.size()));
  EXPECT_TRUE(flush_and_capture().empty());
  aurora::g_config.residentDisplayLists = true;
  GXInvalidateResidentGeometry(dl.data(), static_cast<u32>(dl.size()));
  EXPECT_TRUE(has_aurora_cmd(flush_and_capture(), GX_AURORA_INVALIDATE_RESIDENT));
}

TEST_F(GXResidentTest, DecodesOnceThenHits) {
  const Arrays arrays;
  set_format();
  set_arrays(arrays);
  const auto dl = quad_list();
  call(dl);

  const auto& stats = resident::stats();
  EXPECT_EQ(stats.calls, 1u);
  EXPECT_EQ(stats.misses, 1u);
  EXPECT_EQ(stats.hits, 0u);
  EXPECT_EQ(stats.fallbacks, 0u);
  EXPECT_EQ(resident::entry_count(), 1u);
  EXPECT_TRUE(resident::index_consistent());
  ASSERT_EQ(aurora::gfx::g_testDrawCount, 1u);
  const auto& draw = aurora::gfx::g_testLastDraw;
  EXPECT_EQ(draw.residentArena, 1u);
  EXPECT_EQ(draw.vtxCount, 4u);
  EXPECT_EQ(draw.indexCount, 6u);
  EXPECT_EQ(draw.instanceCount, 1u);

  // The arena holds exactly what the vertex loader produces for the list's vertices
  PipelineConfig config{};
  populate_pipeline_config(config, GX_TRIANGLES, GX_VTXFMT0);
  const auto& loader = vertex_loader(config.shaderConfig);
  std::vector<u8> expected(4 * loader.layout.stride);
  decode_vertices(loader, dl.data() + 3, 4, expected.data(), gxState().arrays, gxState().currentPnMtx);
  auto uploads = resident::take_uploads();
  ASSERT_EQ(uploads.size(), 1u);
  EXPECT_EQ(uploads[0].arena, 0u);
  EXPECT_EQ(uploads[0].offset, 0u);
  EXPECT_EQ(uploads[0].data, expected);
  EXPECT_EQ(draw.vertRange.offset, 0u);
  EXPECT_EQ(draw.vertRange.size, expected.size());
  EXPECT_EQ(resident::used_bytes(), expected.size());

  // The same call again draws without decoding or uploading anything
  call(dl);
  EXPECT_EQ(stats.calls, 2u);
  EXPECT_EQ(stats.hits, 1u);
  EXPECT_EQ(stats.misses, 1u);
  EXPECT_EQ(resident::entry_count(), 1u);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
  EXPECT_EQ(aurora::gfx::g_testLastDraw.residentArena, 1u);
  EXPECT_EQ(aurora::gfx::g_testLastDraw.indexCount, 6u);
  EXPECT_TRUE(resident::take_uploads().empty());
  EXPECT_EQ(resident::used_bytes(), expected.size());
}

TEST_F(GXResidentTest, RewrittenListDecodesAgain) {
  const Arrays arrays;
  set_format();
  set_arrays(arrays);
  auto dl = quad_list();
  call(dl);
  call(dl);
  const auto& stats = resident::stats();
  EXPECT_EQ(stats.misses, 1u);
  EXPECT_EQ(stats.hits, 1u);
  const size_t bytes = resident::used_bytes();
  (void)resident::take_uploads();

  // Same pointer and length, different contents: the head hash changes and the list is decoded again
  dl[4] = 2;
  call(dl);
  EXPECT_EQ(stats.misses, 2u);
  EXPECT_EQ(stats.hits, 1u);
  EXPECT_EQ(resident::entry_count(), 1u);
  EXPECT_TRUE(resident::index_consistent());
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 3u);
  // Arenas are append-only: the new records follow the stale ones
  const auto uploads = resident::take_uploads();
  ASSERT_EQ(uploads.size(), 1u);
  EXPECT_EQ(uploads[0].offset, bytes);
  EXPECT_EQ(resident::used_bytes(), 2 * bytes);

  // A different vertex format for the same bytes is a different entry
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_S16, 4);
  call(dl);
  EXPECT_EQ(stats.misses, 3u);
  EXPECT_EQ(resident::entry_count(), 2u);
  EXPECT_TRUE(resident::index_consistent());
}

TEST_F(GXResidentTest, InvalidateRemovesDependentEntries) {
  const Arrays a;
  const Arrays b;
  set_format();
  const auto dlA = quad_list();
  const auto dlB = quad_list({3, 3, 2, 2, 1, 1, 0, 0});
  // Both lists index the same color array; positions come from their own arrays
  set_arrays(a);
  call(dlA);
  GXSetArray(GX_VA_POS, b.positions, sizeof(b.positions), 6, false);
  call(dlB);
  const auto& stats = resident::stats();
  ASSERT_EQ(resident::entry_count(), 2u);
  EXPECT_TRUE(resident::index_consistent());

  // Memory nobody depends on
  const u8 unrelated[64]{};
  release(unrelated, sizeof(unrelated));
  EXPECT_EQ(resident::entry_count(), 2u);
  EXPECT_EQ(stats.invalidated, 0u);

  // Releasing B's position array removes B, which depends on it only through the array base
  release(b.positions, sizeof(b.positions));
  EXPECT_EQ(resident::entry_count(), 1u);
  EXPECT_EQ(stats.invalidated, 1u);
  EXPECT_TRUE(resident::index_consistent());
  GXSetArray(GX_VA_POS, a.positions, sizeof(a.positions), 6, false);
  call(dlA);
  EXPECT_EQ(stats.hits, 1u);
  GXSetArray(GX_VA_POS, b.positions, sizeof(b.positions), 6, false);
  call(dlB);
  EXPECT_EQ(stats.misses, 3u);
  EXPECT_EQ(resident::entry_count(), 2u);

  // The shared color array takes both
  release(a.colors, sizeof(a.colors));
  EXPECT_EQ(resident::entry_count(), 0u);
  EXPECT_EQ(stats.invalidated, 3u);
  EXPECT_TRUE(resident::index_consistent());

  // Ranges are half-open: the bytes after a list leave it alone, its first byte removes it
  call(dlB);
  EXPECT_EQ(resident::entry_count(), 1u);
  release(dlB.data() + dlB.size(), 64);
  EXPECT_EQ(resident::entry_count(), 1u);
  release(dlB.data(), 1);
  EXPECT_EQ(resident::entry_count(), 0u);
  EXPECT_TRUE(resident::index_consistent());
}

TEST_F(GXResidentTest, UnsupportedListsRunInline) {
  const Arrays arrays;
  set_format();
  set_arrays(arrays);
  // A register write in front of the draw: processed exactly like a copied list
  std::vector<u8> dl{0x61, 0x41, 0x00, 0x00, 0x01};
  const auto quad = quad_list();
  dl.insert(dl.end(), quad.begin(), quad.end());
  call(dl);
  const auto& stats = resident::stats();
  EXPECT_EQ(stats.fallbacks, 1u);
  EXPECT_EQ(stats.misses, 0u);
  EXPECT_EQ(resident::entry_count(), 0u);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 1u);
  EXPECT_EQ(aurora::gfx::g_testLastDraw.residentArena, 0u);
  EXPECT_TRUE(gxState().bpRegValid.test(0x41));

  // Lines are never resident
  std::vector<u8> lines{op(GX_LINES, GX_VTXFMT0), 0, 2, 0, 0, 1, 1};
  call(lines);
  EXPECT_EQ(stats.fallbacks, 2u);
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 2u);
  EXPECT_EQ(aurora::gfx::g_testLastDraw.residentArena, 0u);
}

TEST_F(GXResidentTest, BudgetResetsAtFrameBoundary) {
  const Arrays arrays;
  set_format();
  set_arrays(arrays);
  const auto dlA = quad_list();
  const auto dlB = quad_list({3, 3, 2, 2, 1, 1, 0, 0});
  call(dlA);
  const size_t bytes = resident::used_bytes();
  ASSERT_NE(bytes, 0u);
  aurora::g_config.residentGeometryBudget = static_cast<uint32_t>(bytes);

  // The second list does not fit: it runs inline and the cache asks for a reset
  call(dlB);
  const auto& stats = resident::stats();
  EXPECT_EQ(stats.fallbacks, 1u);
  EXPECT_EQ(stats.resets, 0u);
  EXPECT_EQ(resident::entry_count(), 1u);
  EXPECT_EQ(aurora::gfx::g_testLastDraw.residentArena, 0u);
  call(dlA);
  EXPECT_EQ(stats.hits, 1u);

  // The frame boundary empties the cache; decoding starts over at the beginning of the arena
  aurora::gx::fifo::end_frame();
  EXPECT_EQ(stats.resets, 1u);
  EXPECT_EQ(resident::entry_count(), 0u);
  EXPECT_EQ(resident::used_bytes(), 0u);
  EXPECT_TRUE(resident::index_consistent());
  (void)resident::take_uploads();
  call(dlB);
  EXPECT_EQ(stats.misses, 2u);
  EXPECT_EQ(resident::entry_count(), 1u);
  const auto uploads = resident::take_uploads();
  ASSERT_EQ(uploads.size(), 1u);
  EXPECT_EQ(uploads[0].offset, 0u);
  EXPECT_EQ(uploads[0].data.size(), bytes);
}

TEST(GXResidentDecode, BuildsAbsoluteIndicesPerArena) {
  resident::shutdown();
  const Arrays arrays;
  const auto config = quad_config();
  const auto& loader = vertex_loader(config);
  const auto attrArrays = quad_arrays(arrays);
  const auto quad = quad_list();
  std::vector<u8> strip{op(GX_TRIANGLESTRIP, GX_VTXFMT0), 0, 4, 0, 0, 1, 1, 2, 2, 3, 3};
  std::vector<u8> mixed{op(GX_TRIANGLES, GX_VTXFMT0), 0, 3, 0, 0, 1, 1, 2, 2,
                        op(GX_TRIANGLES, GX_VTXFMT1), 0, 3, 0, 0, 1, 1, 2, 2};
  std::vector<u8> overrun{op(GX_TRIANGLES, GX_VTXFMT0), 0, 30, 0, 0};

  resident::Entry first;
  ASSERT_TRUE(resident::decode_display_list(quad.data(), static_cast<u32>(quad.size()), GX_VTXFMT0, loader, attrArrays,
                                            0, first));
  EXPECT_EQ(first.arena, 0u);
  EXPECT_EQ(first.firstVertex, 0u);
  EXPECT_EQ(first.vertexCount, 4u);
  EXPECT_EQ(first.indices, (std::vector<u32>{0, 1, 2, 2, 3, 0}));
  EXPECT_EQ(first.arrays[GX_VA_POS], static_cast<const void*>(arrays.positions));
  EXPECT_EQ(first.arrays[GX_VA_CLR0], static_cast<const void*>(arrays.colors));
  EXPECT_EQ(first.arrays[GX_VA_NRM], nullptr);

  resident::Entry second;
  ASSERT_TRUE(resident::decode_display_list(strip.data(), static_cast<u32>(strip.size()), GX_VTXFMT0, loader,
                                            attrArrays, 0, second));
  EXPECT_EQ(second.arena, 0u);
  EXPECT_EQ(second.firstVertex, 4u);
  EXPECT_EQ(second.indices, (std::vector<u32>{4, 5, 6, 6, 5, 7}));
  EXPECT_EQ(resident::used_bytes(), 8 * loader.layout.stride);
  EXPECT_EQ(resident::arena_stride(0), loader.layout.stride);

  // Both lists decoded into one contiguous upload; its bytes are the loader's output
  auto uploads = resident::take_uploads();
  ASSERT_EQ(uploads.size(), 1u);
  std::vector<u8> expected(8 * loader.layout.stride);
  decode_vertices(loader, quad.data() + 3, 4, expected.data(), attrArrays, 0);
  decode_vertices(loader, strip.data() + 3, 4, expected.data() + 4 * loader.layout.stride, attrArrays, 0);
  EXPECT_EQ(uploads[0].data, expected);

  // Refused lists leave the arena untouched
  resident::Entry refused;
  EXPECT_FALSE(resident::decode_display_list(mixed.data(), static_cast<u32>(mixed.size()), GX_VTXFMT0, loader,
                                             attrArrays, 0, refused));
  EXPECT_FALSE(resident::decode_display_list(overrun.data(), static_cast<u32>(overrun.size()), GX_VTXFMT0, loader,
                                             attrArrays, 0, refused));
  EXPECT_EQ(resident::used_bytes(), 8 * loader.layout.stride);
  EXPECT_TRUE(resident::take_uploads().empty());
  resident::shutdown();
}

// The pointer index against a brute-force model over random inserts, erases and range releases.
TEST(GXResidentIndex, MatchesBruteForceModel) {
  resident::shutdown();
  static u8 memory[64 * 32];
  std::mt19937 rng{7};
  const auto pointer = [&] { return memory + (rng() % 64) * 32; };
  struct Model {
    std::set<const u8*> pointers;
  };
  std::vector<std::pair<u64, Model>> model;
  u64 nextKey = 1;

  for (int round = 0; round < 2000; ++round) {
    const auto action = rng() % 4;
    if (action <= 1 || model.empty()) {
      resident::Entry entry;
      entry.list = pointer();
      const auto arrayCount = rng() % 4;
      for (u32 i = 0; i < arrayCount; ++i) {
        entry.arrays[GX_VA_POS + rng() % 12] = pointer(); // may repeat an attribute or a pointer
      }
      Model m;
      m.pointers.insert(static_cast<const u8*>(entry.list));
      for (const void* p : entry.arrays) {
        if (p != nullptr) {
          m.pointers.insert(static_cast<const u8*>(p));
        }
      }
      const u64 key = nextKey++;
      resident::insert(key, std::move(entry));
      model.emplace_back(key, std::move(m));
    } else if (action == 2) {
      const auto victim = model.begin() + rng() % model.size();
      resident::erase(victim->first);
      model.erase(victim);
    } else {
      const u8* base = pointer();
      const size_t size = 1 + rng() % 200;
      size_t expected = 0;
      std::erase_if(model, [&](const auto& item) {
        const bool hit = std::any_of(item.second.pointers.begin(), item.second.pointers.end(),
                                     [&](const u8* p) { return p >= base && p < base + size; });
        expected += hit;
        return hit;
      });
      EXPECT_EQ(resident::invalidate(base, size), expected);
    }
    ASSERT_EQ(resident::entry_count(), model.size());
    ASSERT_TRUE(resident::index_consistent());
    for (const auto& [key, m] : model) {
      ASSERT_NE(resident::find(key), nullptr);
    }
  }
  resident::shutdown();
}
