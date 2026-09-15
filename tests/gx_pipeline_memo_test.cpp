// Pipeline-state memo tests
//
// Pattern: drive the command processor with real GX API calls (encode, decode,
// draw) and check that the pipeline config and pipeline key a draw ends up with
// are identical to what populate_pipeline_config() and pipeline_ref() produce
// for the decoded state, whether the memo hit or missed. Then check that every
// hashed input misses when it changes and that shutdown forgets the memo.

#include "gx_test_common.hpp"
#include "gfx/recording.hpp"
#include "gx/pipeline.hpp"

#include <array>
#include <functional>
#include <random>

namespace fifo = aurora::gx::fifo;

namespace aurora::gfx {
extern gx::DrawData g_testLastDraw;
extern uint32_t g_testDrawCount;
} // namespace aurora::gfx

namespace {
// One knob per pipeline-relevant piece of GX state.
struct Material {
  GXTevMode tevMode = GX_MODULATE;
  u8 numTevStages = 1;
  GXTevSwapSel texSwap = GX_TEV_SWAP0;
  GXTevColorChan swap1Red = GX_CH_RED;
  u8 numIndStages = 0;
  bool lighting = false;
  GXTexGenSrc texGenSrc = GX_TG_TEX0;
  GXCompare alphaComp0 = GX_ALWAYS;
  u8 alphaRef0 = 0;
  GXFogType fogType = GX_FOG_NONE;
  GXBlendMode blendMode = GX_BM_BLEND;
  GXBlendFactor blendSrc = GX_BL_SRCALPHA;
  GXBlendFactor blendDst = GX_BL_INVSRCALPHA;
  GXLogicOp logicOp = GX_LO_CLEAR;
  GXCompare depthFunc = GX_LEQUAL;
  bool depthCompare = true;
  bool depthUpdate = true;
  GXCullMode cullMode = GX_CULL_BACK;
  bool dstAlpha = false;
  u8 dstAlphaValue = 0;
  bool colorUpdate = true;
  bool alphaUpdate = true;
  f32 frontOffset = 0.f;
  GXAttrType nrmDesc = GX_NONE;
  u8 nrmStride = 12;
  GXCompType posType = GX_F32;
  u8 posFrac = 0;
};

alignas(32) const std::array<f32, 12> sNormals{};

void apply(const Material& m) {
  GXClearVtxDesc();
  GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
  GXSetVtxDesc(GX_VA_NRM, m.nrmDesc);
  GXSetVtxDesc(GX_VA_TEX0, GX_DIRECT);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, m.posType, m.posFrac);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_NRM, GX_NRM_XYZ, GX_F32, 0);
  GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_TEX0, GX_TEX_ST, GX_F32, 0);
  if (m.nrmDesc == GX_INDEX8) {
    GXSetArray(GX_VA_NRM, sNormals.data(), sizeof(sNormals), m.nrmStride, false);
  }
  GXSetNumChans(1);
  GXSetChanCtrl(GX_COLOR0A0, m.lighting, GX_SRC_REG, GX_SRC_REG, m.lighting ? GX_LIGHT0 : 0, GX_DF_CLAMP, GX_AF_NONE);
  GXSetNumTexGens(1);
  GXSetTexCoordGen2(GX_TEXCOORD0, GX_TG_MTX2x4, m.texGenSrc, GX_IDENTITY, GX_FALSE, GX_PTIDENTITY);
  GXSetNumTevStages(m.numTevStages);
  for (u8 i = 0; i < m.numTevStages; ++i) {
    const auto stage = static_cast<GXTevStageID>(GX_TEVSTAGE0 + i);
    GXSetTevOrder(stage, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
    GXSetTevOp(stage, m.tevMode);
    GXSetTevSwapMode(stage, GX_TEV_SWAP0, m.texSwap);
  }
  GXSetTevSwapModeTable(GX_TEV_SWAP1, m.swap1Red, GX_CH_GREEN, GX_CH_BLUE, GX_CH_ALPHA);
  GXSetNumIndStages(m.numIndStages);
  if (m.numIndStages > 0) {
    GXSetIndTexOrder(GX_INDTEXSTAGE0, GX_TEXCOORD0, GX_TEXMAP1);
  }
  GXSetAlphaCompare(m.alphaComp0, m.alphaRef0, GX_AOP_AND, GX_ALWAYS, 0);
  GXSetFog(m.fogType, 0.f, 100.f, 0.1f, 1000.f, GXColor{0, 0, 0, 0});
  GXSetBlendMode(m.blendMode, m.blendSrc, m.blendDst, m.logicOp);
  GXSetZMode(m.depthCompare, m.depthFunc, m.depthUpdate);
  GXSetCullMode(m.cullMode);
  GXSetDstAlpha(m.dstAlpha, m.dstAlphaValue);
  GXSetColorUpdate(m.colorUpdate);
  GXSetAlphaUpdate(m.alphaUpdate);
  GX2SetPolygonOffset(m.frontOffset, 0.f, 0.f, 0.f, 0.f);
}

// Sends the material, draws one triangle with it and waits for the processor.
void draw(const Material& m) {
  apply(m);
  GXBegin(GX_TRIANGLES, GX_VTXFMT0, 3);
  for (int i = 0; i < 3; ++i) {
    if (m.posType == GX_F32) {
      GXPosition3f32(static_cast<f32>(i), 0.f, 0.f);
    } else {
      GXPosition3s16(static_cast<s16>(i), 0, 0);
    }
    if (m.nrmDesc == GX_DIRECT) {
      GXNormal3f32(0.f, 0.f, 1.f);
    } else if (m.nrmDesc == GX_INDEX8) {
      GXNormal1x8(0);
    }
    GXTexCoord2f32(0.f, 0.f);
  }
  GXEnd();
  fifo::drain();
}

// The uncached path: rebuild the config from the decoded state.
aurora::gx::PipelineConfig uncached_config() {
  aurora::gx::PipelineConfig config{};
  aurora::gx::populate_pipeline_config(config, GX_TRIANGLES, GX_VTXFMT0);
  return config;
}

aurora::gfx::PipelineRef last_pipeline() { return aurora::gfx::g_testLastDraw.pipeline; }

// The most recent draw must carry exactly the config and pipeline key the
// uncached path derives from the same state.
void expect_matches_uncached() {
  const auto expected = uncached_config();
  const auto& cached = fifo::testing::cached_pipeline_config();
  EXPECT_TRUE(cached.shaderConfig == expected.shaderConfig);
  EXPECT_EQ(std::memcmp(&cached, &expected, sizeof(expected)), 0);
  EXPECT_EQ(last_pipeline(), aurora::gfx::pipeline_ref(expected));
}

class GXPipelineMemoTest : public GXFifoTest {
protected:
  void SetUp() override {
    GXFifoTest::SetUp();
    fifo::init();
    fifo::begin_frame();
    aurora::gfx::g_testDrawCount = 0;
    m_baseMisses = fifo::testing::pipeline_memo_misses();
  }

  void TearDown() override {
    fifo::end_frame();
    GXFifoTest::TearDown();
  }

  // Memo misses since the test started
  uint32_t misses() const { return fifo::testing::pipeline_memo_misses() - m_baseMisses; }

private:
  uint32_t m_baseMisses = 0;
};
} // namespace

TEST_F(GXPipelineMemoTest, AlternatingMaterialsHitMemo) {
  Material a{};
  Material b{};
  b.tevMode = GX_REPLACE;
  b.blendMode = GX_BM_NONE;

  draw(a);
  EXPECT_EQ(misses(), 1u);
  expect_matches_uncached();
  const auto refA = last_pipeline();

  draw(b);
  EXPECT_EQ(misses(), 2u);
  expect_matches_uncached();
  const auto refB = last_pipeline();
  EXPECT_NE(refA, refB);

  // Every switch back re-sends a material the memo already resolved.
  for (int i = 0; i < 4; ++i) {
    draw(a);
    EXPECT_EQ(misses(), 2u);
    expect_matches_uncached();
    EXPECT_EQ(last_pipeline(), refA);

    draw(b);
    EXPECT_EQ(misses(), 2u);
    expect_matches_uncached();
    EXPECT_EQ(last_pipeline(), refB);
  }
  EXPECT_EQ(aurora::gfx::g_testDrawCount, 10u);
}

TEST_F(GXPipelineMemoTest, EveryHashedInputMisses) {
  struct Mutation {
    const char* name;
    std::function<void(Material&)> apply;
  };
  const std::array mutations{
      Mutation{"tev op", [](Material& m) { m.tevMode = GX_REPLACE; }},
      Mutation{"tev stage count", [](Material& m) { m.numTevStages = 2; }},
      Mutation{"tev swap select", [](Material& m) { m.texSwap = GX_TEV_SWAP1; }},
      Mutation{"tev swap table", [](Material& m) { m.swap1Red = GX_CH_ALPHA; }},
      Mutation{"indirect stages", [](Material& m) { m.numIndStages = 1; }},
      Mutation{"lighting", [](Material& m) { m.lighting = true; }},
      Mutation{"texgen source", [](Material& m) { m.texGenSrc = GX_TG_TEX1; }},
      Mutation{"alpha compare",
               [](Material& m) {
                 m.alphaComp0 = GX_GREATER;
                 m.alphaRef0 = 128;
               }},
      Mutation{"fog type", [](Material& m) { m.fogType = GX_FOG_LIN; }},
      Mutation{"blend mode", [](Material& m) { m.blendMode = GX_BM_NONE; }},
      Mutation{"blend factor", [](Material& m) { m.blendSrc = GX_BL_ONE; }},
      Mutation{"logic op", [](Material& m) { m.logicOp = GX_LO_OR; }},
      Mutation{"depth func", [](Material& m) { m.depthFunc = GX_ALWAYS; }},
      Mutation{"depth compare", [](Material& m) { m.depthCompare = false; }},
      Mutation{"depth update", [](Material& m) { m.depthUpdate = false; }},
      Mutation{"cull mode", [](Material& m) { m.cullMode = GX_CULL_NONE; }},
      Mutation{"dst alpha",
               [](Material& m) {
                 m.dstAlpha = true;
                 m.dstAlphaValue = 0x80;
               }},
      Mutation{"color update", [](Material& m) { m.colorUpdate = false; }},
      Mutation{"alpha update", [](Material& m) { m.alphaUpdate = false; }},
      Mutation{"polygon offset", [](Material& m) { m.frontOffset = 1.f; }},
      Mutation{"vertex descriptor direct", [](Material& m) { m.nrmDesc = GX_DIRECT; }},
      Mutation{"vertex descriptor indexed", [](Material& m) { m.nrmDesc = GX_INDEX8; }},
      Mutation{"array stride",
               [](Material& m) {
                 m.nrmDesc = GX_INDEX8;
                 m.nrmStride = 24;
               }},
      Mutation{"vertex component type",
               [](Material& m) {
                 m.posType = GX_S16;
                 m.posFrac = 4;
               }},
      Mutation{"vertex fraction", [](Material& m) { m.posFrac = 2; }},
  };

  const Material base{};
  draw(base);
  EXPECT_EQ(misses(), 1u);
  expect_matches_uncached();
  const auto refBase = last_pipeline();

  for (const auto& mutation : mutations) {
    SCOPED_TRACE(mutation.name);
    Material m = base;
    mutation.apply(m);
    const auto before = misses();
    draw(m);
    EXPECT_EQ(misses(), before + 1);
    expect_matches_uncached();
    EXPECT_NE(last_pipeline(), refBase);

    // Returning to the base material is a hit.
    draw(base);
    EXPECT_EQ(misses(), before + 1);
    expect_matches_uncached();
    EXPECT_EQ(last_pipeline(), refBase);
  }
}

TEST_F(GXPipelineMemoTest, RandomMaterialsMatchUncachedPath) {
  std::mt19937 rng{0x6d656d6f};
  const auto pick = [&](int n) { return static_cast<int>(rng() % n); };
  std::array<Material, 8> pool{};
  for (auto& m : pool) {
    m.tevMode = std::array{GX_MODULATE, GX_DECAL, GX_BLEND, GX_REPLACE, GX_PASSCLR}[pick(5)];
    m.numTevStages = static_cast<u8>(1 + pick(3));
    m.numIndStages = static_cast<u8>(pick(2));
    m.lighting = pick(2) != 0;
    m.alphaComp0 = std::array{GX_ALWAYS, GX_GREATER, GX_LESS}[pick(3)];
    m.alphaRef0 = static_cast<u8>(pick(256));
    m.fogType = std::array{GX_FOG_NONE, GX_FOG_LIN, GX_FOG_EXP}[pick(3)];
    m.blendMode = std::array{GX_BM_NONE, GX_BM_BLEND, GX_BM_LOGIC}[pick(3)];
    m.blendSrc = std::array{GX_BL_ONE, GX_BL_SRCALPHA, GX_BL_DSTALPHA}[pick(3)];
    m.blendDst = std::array{GX_BL_ZERO, GX_BL_INVSRCALPHA, GX_BL_ONE}[pick(3)];
    m.depthFunc = std::array{GX_LEQUAL, GX_ALWAYS, GX_LESS}[pick(3)];
    m.depthCompare = pick(2) != 0;
    m.depthUpdate = pick(2) != 0;
    m.cullMode = std::array{GX_CULL_NONE, GX_CULL_BACK, GX_CULL_FRONT}[pick(3)];
    m.dstAlpha = pick(2) != 0;
    m.dstAlphaValue = static_cast<u8>(pick(256));
    m.colorUpdate = pick(2) != 0;
    m.alphaUpdate = pick(2) != 0;
    m.frontOffset = static_cast<f32>(pick(3));
    m.nrmDesc = std::array{GX_NONE, GX_DIRECT, GX_INDEX8}[pick(3)];
    m.nrmStride = static_cast<u8>(12 * (1 + pick(2)));
    m.posType = std::array{GX_F32, GX_S16}[pick(2)];
    m.posFrac = static_cast<u8>(pick(8));
  }

  constexpr uint32_t DrawCount = 96;
  for (uint32_t i = 0; i < DrawCount; ++i) {
    SCOPED_TRACE(i);
    draw(pool[pick(pool.size())]);
    expect_matches_uncached();
  }
  EXPECT_EQ(aurora::gfx::g_testDrawCount, DrawCount);
  // At most one miss per distinct material; the rest were hits.
  EXPECT_LE(misses(), pool.size());
  EXPECT_LT(misses(), DrawCount);
}

TEST_F(GXPipelineMemoTest, ShutdownForgetsMemo) {
  const Material a{};
  draw(a);
  EXPECT_EQ(misses(), 1u);
  const auto refA = last_pipeline();

  // The pipeline cache is emptied on shutdown, so a re-initialized processor must
  // request the pipeline again even though no register changed.
  fifo::end_frame();
  fifo::shutdown();
  fifo::init();
  fifo::begin_frame();

  draw(a);
  EXPECT_EQ(misses(), 2u);
  expect_matches_uncached();
  EXPECT_EQ(last_pipeline(), refA);
}
