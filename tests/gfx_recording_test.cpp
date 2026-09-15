#include <gtest/gtest.h>

#include "gfx/clear.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/pipeline_cache.hpp"
#include "gfx/recording.hpp"
#include "gfx/texture.hpp"
#include "gx/gx.hpp"
#include "gx/pipeline.hpp"
#include "internal.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <memory>

namespace aurora::gfx {
namespace {

constexpr auto ColorFormat = wgpu::TextureFormat::RGBA8Unorm;
constexpr auto DepthFormat = wgpu::TextureFormat::Depth24Plus;

class GfxRecordingTest : public ::testing::Test {
protected:
  void SetUp() override {
    webgpu::g_graphicsConfig.surfaceConfiguration.format = ColorFormat;
    webgpu::g_graphicsConfig.depthFormat = DepthFormat;
    webgpu::g_graphicsConfig.msaaSamples = 1;
    webgpu::g_frameBuffer.size = {640, 480, 1};
    webgpu::g_frameBuffer.format = ColorFormat;
    webgpu::g_depthBuffer.size = {640, 480, 1};
    webgpu::g_depthBuffer.format = DepthFormat;
    g_config.disableRenderPassFusion = false;
    gx::g_gxState.clearColor = clear_value();
    gx::g_gxState.colorUpdate = true;
    gx::g_gxState.alphaUpdate = true;
    gx::g_gxState.depthCompare = true;
    gx::g_gxState.depthUpdate = true;
    detail::testing::suppress_pipeline_creation(true);
    detail::testing::suppress_render_worker(true);
    detail::begin_recording(frame, 0);
  }

  void TearDown() override {
    if (recordingActive) {
      if (is_offscreen()) {
        end_offscreen();
      }
      finish();
      detail::end_recording();
    }
    detail::shutdown_recording();
  }

  void seed(uint32_t width, uint32_t height) {
    detail::testing::seed_offscreen_cache(width, height, ColorFormat, DepthFormat);
  }

  void copy_current_offscreen() {
    const auto& pass = frame.renderPasses.back();
    const auto& size = pass.colorAttachments[SceneColorAttachmentIndex].size;
    auto target = std::make_shared<TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{}, size,
                                               ColorFormat, 1, GX_TF_RGBA8);
    resolve_pass_into(std::move(target), {0, 0, static_cast<int32_t>(size.width), static_cast<int32_t>(size.height)},
                      false, false, false, {}, 1.f);
  }

  size_t count_efb_passes() const {
    return static_cast<size_t>(
        std::ranges::count_if(frame.renderPasses, [](const auto& pass) { return pass.label.starts_with("EFB"); }));
  }

  // --- Pass fusion helpers: a game rendering small render-to-texture targets (shadow maps) at the origin ---

  static constexpr ClipRect SmallRect{0, 0, 256, 256};
  static constexpr Viewport SmallViewport{0.f, 0.f, 256.f, 256.f, 0.f, 1.f};
  static constexpr uint8_t WriteColor = 1;
  static constexpr uint8_t WriteAlpha = 2;
  static constexpr uint8_t WriteDepth = 4;

  static Vec4<float> clear_value() { return {0.f, 0.f, 0.f, 1.f}; }

  static TextureHandle make_target(uint32_t width, uint32_t height, GXTexFmt format = GX_TF_I8) {
    return std::make_shared<TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{},
                                        wgpu::Extent3D{width, height, 1}, ColorFormat, 1, format);
  }

  // A GX draw with the given update masks enabled (GXSetColorUpdate, GXSetAlphaUpdate, GXSetZMode).
  static void gx_draw(bool color = true, bool alpha = false, bool depth = false) {
    gx::g_gxState.colorUpdate = color;
    gx::g_gxState.alphaUpdate = alpha;
    gx::g_gxState.depthCompare = depth;
    gx::g_gxState.depthUpdate = depth;
    push_draw_command(gx::DrawData{});
  }

  // A small render-to-texture segment: viewport and scissor at the origin, one color-only draw.
  static void small_segment(const Viewport& viewport = SmallViewport, const ClipRect& scissor = SmallRect) {
    set_viewport(viewport);
    set_scissor(scissor);
    gx_draw();
  }

  // GXCopyTex with clear. GX copy clears honor the update masks; a shadow-map copy clears color only.
  static void copy(TextureHandle target, const ClipRect& rect = SmallRect, bool clearColor = true,
                   bool clearAlpha = false, bool clearDepth = false, Vec4<float> clearValue = clear_value()) {
    resolve_pass_into(std::move(target), rect, clearColor, clearAlpha, clearDepth, clearValue, gx::clear_depth_value(),
                      GX_TF_I8);
  }

  // Records a small segment and its copy. With fusion the pass stays open for the segment that follows.
  TextureHandle begin_fused() {
    small_segment();
    auto target = make_target(256, 256);
    copy(target);
    EXPECT_EQ(frame.renderPasses.size(), 1u);
    EXPECT_FALSE(frame.renderPasses[0].sealed);
    return target;
  }

  static size_t count_commands(const detail::RenderPass& pass, detail::CommandType type) {
    return static_cast<size_t>(
        std::ranges::count_if(pass.commands, [type](const auto& cmd) { return cmd.type == type; }));
  }

  static const detail::Command* last_command(const detail::RenderPass& pass, detail::CommandType type) {
    const auto it = std::ranges::find_if(pass.commands.rbegin(), pass.commands.rend(),
                                         [type](const auto& cmd) { return cmd.type == type; });
    return it == pass.commands.rend() ? nullptr : &*it;
  }

  static bool all_viewports_at(const detail::RenderPass& pass, float left) {
    return std::ranges::all_of(pass.commands, [left](const auto& cmd) {
      return cmd.type != detail::CommandType::SetViewport || cmd.data.setViewport.left == left;
    });
  }

  static bool all_scissors_at(const detail::RenderPass& pass, int32_t x) {
    return std::ranges::all_of(pass.commands, [x](const auto& cmd) {
      return cmd.type != detail::CommandType::SetScissor || cmd.data.setScissor.x == x;
    });
  }

  detail::FramePacket frame;
  bool recordingActive = true;
};

TEST_F(GfxRecordingTest, CreateRestoreReturnsToEfb) {
  seed(320, 180);
  begin_offscreen(320, 180);
  ASSERT_TRUE(is_offscreen());

  end_offscreen();

  EXPECT_FALSE(is_offscreen());
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_TRUE(frame.renderPasses[0].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, EfbPassUsesDiscoveredSceneLayout) {
  ASSERT_FALSE(frame.renderPasses.empty());
  const auto discovered = scene_render_target_layout();
  const auto targetLayout = frame.renderPasses.front().target_layout();

  EXPECT_EQ(targetLayout.key, discovered.key);
  EXPECT_EQ(targetLayout.colorAttachmentCount, discovered.colorAttachmentCount);
  EXPECT_EQ(targetLayout.colorAttachments[SceneColorAttachmentIndex].semantic, ColorAttachmentSemantic::SceneColor);
}

TEST_F(GfxRecordingTest, CopiedOffscreenPassIsRetainedOnRestore) {
  seed(320, 180);
  begin_offscreen(320, 180);
  copy_current_offscreen();

  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_FALSE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[0].has_consumer());
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
}

TEST_F(GfxRecordingTest, ReplacementRetainsCopiedPassesAndDiscardsContinuations) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  copy_current_offscreen();
  begin_offscreen(160, 90);
  copy_current_offscreen();

  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 5u);
  EXPECT_TRUE(frame.renderPasses[0].has_consumer());
  EXPECT_FALSE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
  EXPECT_TRUE(frame.renderPasses[2].has_consumer());
  EXPECT_FALSE(frame.renderPasses[2].discardable);
  EXPECT_TRUE(frame.renderPasses[3].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, RepeatedUncopiedCreatesDiscardEarlierPasses) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  begin_offscreen(160, 90);
  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_TRUE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, PublicCreatePassRejectsExistingOffscreenPass) {
  seed(320, 180);
  seed(160, 90);
  ASSERT_TRUE(create_pass(320, 180));

  EXPECT_FALSE(create_pass(160, 90));

  ResolvedTargets ignored;
  EXPECT_TRUE(resolve_pass({.color = false, .depth = false}, ignored));
}

TEST_F(GfxRecordingTest, FinalizedPassesAreSealedOrDeliberatelyDiscarded) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  copy_current_offscreen();
  begin_offscreen(160, 90);
  end_offscreen();
  finish();

  ASSERT_FALSE(frame.renderPasses.empty());
  for (const auto& pass : frame.renderPasses) {
    EXPECT_TRUE(pass.sealed);
    if (!pass.has_consumer() && pass.label.starts_with("Offscreen")) {
      EXPECT_TRUE(pass.discardable);
    }
  }
  detail::end_recording();
  recordingActive = false;
}

// --- Pass fusion ---

TEST_F(GfxRecordingTest, AdjacentSmallCopiesFuseIntoOnePass) {
  const auto first = begin_fused();
  small_segment();
  const auto second = make_target(256, 256);
  copy(second);

  ASSERT_EQ(frame.renderPasses.size(), 2u);
  const auto& fused = frame.renderPasses[0];
  EXPECT_TRUE(fused.sealed);
  EXPECT_EQ(fused.resolveTarget, first);
  EXPECT_EQ(fused.resolveRect, SmallRect);
  ASSERT_EQ(fused.extraResolves.size(), 1u);
  EXPECT_EQ(fused.extraResolves[0].target, second);
  EXPECT_EQ(fused.extraResolves[0].rect, (ClipRect{256, 0, 256, 256}));
  EXPECT_EQ(fused.extraResolves[0].format, GX_TF_I8);
  EXPECT_EQ(fused.extraResolves[0].uniformRange.size, 16u);
  EXPECT_EQ(fused.writeMask, WriteColor);
  // The second segment was recorded right of the first copy rectangle.
  EXPECT_EQ(count_commands(fused, detail::CommandType::Draw), 2u);
  EXPECT_EQ(last_command(fused, detail::CommandType::SetViewport)->data.setViewport.left, 256.f);
  EXPECT_EQ(last_command(fused, detail::CommandType::SetScissor)->data.setScissor.x, 256);

  // The main scene continues in a pass that opens with the color-only clear the second copy requested.
  const auto& main = frame.renderPasses[1];
  EXPECT_FALSE(main.sealed);
  EXPECT_TRUE(main.label.starts_with("EFB"));
  EXPECT_EQ(main.leadingClearMask, WriteColor);
  EXPECT_FALSE(main.colorAttachments[SceneColorAttachmentIndex].clear);
  EXPECT_FALSE(main.clearDepth);
  ASSERT_FALSE(main.commands.empty());
  EXPECT_EQ(main.commands[0].type, detail::CommandType::Draw);
  EXPECT_TRUE(all_viewports_at(main, 0.f));

  finish();
  EXPECT_EQ(frame.renderPasses.size(), 2u);
}

TEST_F(GfxRecordingTest, FusionContinuesAfterLeadingClearDraw) {
  // A pass that opens with a color-only clear draw counts as cleared at pass start.
  begin_fused();
  small_segment();
  copy(make_target(256, 256));
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  ASSERT_EQ(frame.renderPasses[1].leadingClearMask, WriteColor);

  small_segment();
  const auto third = make_target(256, 256);
  copy(third);
  EXPECT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_FALSE(frame.renderPasses[1].sealed);
  EXPECT_EQ(frame.renderPasses[1].resolveTarget, third);

  small_segment();
  copy(make_target(256, 256));
  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_EQ(frame.renderPasses[1].extraResolves.size(), 1u);
}

TEST_F(GfxRecordingTest, FullClearCopyFusesWithLoadOpClear) {
  small_segment();
  copy(make_target(256, 256), SmallRect, true, true, false);
  EXPECT_EQ(frame.renderPasses.size(), 1u);
  EXPECT_FALSE(frame.renderPasses[0].sealed);
}

TEST_F(GfxRecordingTest, ViewportThatDoesNotFitSplitsFusion) {
  const auto first = begin_fused();
  const Viewport full{0.f, 0.f, 640.f, 480.f, 0.f, 1.f};
  set_viewport(full);

  ASSERT_EQ(frame.renderPasses.size(), 2u);
  const auto& fused = frame.renderPasses[0];
  EXPECT_TRUE(fused.sealed);
  EXPECT_EQ(fused.resolveTarget, first);
  EXPECT_TRUE(fused.extraResolves.empty());
  EXPECT_TRUE(all_viewports_at(fused, 0.f));
  EXPECT_EQ(count_commands(fused, detail::CommandType::Draw), 1u);

  // The shifted segment became the pass the copy would have started, with its coordinates restored.
  const auto& split = frame.renderPasses[1];
  EXPECT_FALSE(split.sealed);
  EXPECT_EQ(split.leadingClearMask, WriteColor);
  ASSERT_GE(split.commands.size(), 4u);
  EXPECT_EQ(split.commands[0].type, detail::CommandType::Draw);
  EXPECT_EQ(split.commands[1].type, detail::CommandType::SetViewport);
  EXPECT_EQ(split.commands[1].data.setViewport, SmallViewport);
  EXPECT_EQ(split.commands[2].type, detail::CommandType::SetScissor);
  EXPECT_EQ(split.commands[2].data.setScissor, SmallRect);
  EXPECT_EQ(split.commands[3].data.setViewport, full);
  EXPECT_TRUE(all_scissors_at(split, 0));
  EXPECT_EQ(split.writeMask, 0u);

  gx_draw();
  EXPECT_EQ(split.writeMask, WriteColor);
}

TEST_F(GfxRecordingTest, NegativeScissorSplitsFusion) {
  begin_fused();
  set_scissor({-8, 0, 256, 256});
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_EQ(last_command(frame.renderPasses[1], detail::CommandType::SetScissor)->data.setScissor.x, -8);
}

TEST_F(GfxRecordingTest, ClearDrawSplitsFusion) {
  begin_fused();
  push_draw_command(clear::DrawData{});
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_EQ(count_commands(frame.renderPasses[1], detail::CommandType::Draw), 2u);
  EXPECT_EQ(frame.renderPasses[1].writeMask, 0u);
}

TEST_F(GfxRecordingTest, SamplingTheFirstCopySplitsFusion) {
  const auto first = begin_fused();
  on_copy_texture_sampled(make_target(64, 64));
  EXPECT_EQ(frame.renderPasses.size(), 1u);

  on_copy_texture_sampled(first);
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_FALSE(frame.renderPasses[1].sealed);
}

TEST_F(GfxRecordingTest, PaletteConversionSplitsFusion) {
  begin_fused();
  queue_palette_conv({});
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].paletteConvs.empty());
  EXPECT_EQ(frame.renderPasses[1].paletteConvs.size(), 1u);
}

TEST_F(GfxRecordingTest, FinishSplitsFusion) {
  begin_fused();
  finish();
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_FALSE(frame.renderPasses[0].captureDepthSnapshot);
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_TRUE(frame.renderPasses[1].captureDepthSnapshot);
  EXPECT_EQ(frame.renderPasses[1].leadingClearMask, WriteColor);
  detail::end_recording();
  recordingActive = false;
}

TEST_F(GfxRecordingTest, PublicResolvePassSplitsFusion) {
  begin_fused();
  ResolvedTargets ignored;
  ASSERT_TRUE(resolve_pass({.color = false, .depth = false}, ignored));
  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_FALSE(frame.renderPasses[1].discardable);
  EXPECT_EQ(frame.renderPasses[1].leadingClearMask, WriteColor);
  EXPECT_FALSE(frame.renderPasses[2].sealed);
}

TEST_F(GfxRecordingTest, OffscreenPassSplitsFusion) {
  seed(320, 180);
  begin_fused();
  begin_offscreen(320, 180);
  // The split pass has no consumer yet, so it is suspended while the offscreen pass records.
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_TRUE(frame.renderPasses[1].label.starts_with("Offscreen"));

  end_offscreen();
  ASSERT_EQ(frame.renderPasses.size(), 3u);
  const auto& restored = frame.renderPasses[2];
  EXPECT_TRUE(restored.label.starts_with("EFB"));
  EXPECT_EQ(restored.leadingClearMask, WriteColor);
  EXPECT_TRUE(all_viewports_at(restored, 0.f));
}

TEST_F(GfxRecordingTest, SecondCopyMustClearEverythingWritten) {
  const auto first = begin_fused();
  gx_draw(true, false, true); // the shifted segment writes depth; a color-only copy clear cannot hide it
  const auto second = make_target(256, 256);
  copy(second);

  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_EQ(frame.renderPasses[0].resolveTarget, first);
  EXPECT_TRUE(frame.renderPasses[0].extraResolves.empty());
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_EQ(frame.renderPasses[1].resolveTarget, second);
  EXPECT_EQ(frame.renderPasses[1].writeMask, WriteColor | WriteDepth);
  EXPECT_TRUE(all_viewports_at(frame.renderPasses[1], 0.f));
  EXPECT_FALSE(frame.renderPasses[2].sealed);
}

TEST_F(GfxRecordingTest, SecondCopyMustFitBesideTheFirst) {
  begin_fused();
  small_segment();
  copy(make_target(400, 256), {0, 0, 400, 256}); // 256 + 400 > 640: the shifted rectangle does not fit
  // Fusion refused the second copy, and the split pass was resolved on its own instead.
  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_TRUE(frame.renderPasses[0].extraResolves.empty());
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_EQ(frame.renderPasses[1].resolveRect, (ClipRect{0, 0, 400, 256}));
  EXPECT_TRUE(all_viewports_at(frame.renderPasses[1], 0.f));
}

TEST_F(GfxRecordingTest, FusionRequiresColorOnlyFirstSegment) {
  {
    set_viewport(SmallViewport);
    set_scissor(SmallRect);
    gx_draw(true, false, true); // writes depth
    copy(make_target(256, 256));
    EXPECT_EQ(frame.renderPasses.size(), 2u);
    EXPECT_TRUE(frame.renderPasses[0].sealed);
  }
  {
    gx_draw(true, true, false); // writes alpha
    copy(make_target(256, 256));
    EXPECT_EQ(frame.renderPasses.size(), 3u);
  }
  {
    gx_draw(); // color only, but the copy clears nothing
    copy(make_target(256, 256), SmallRect, false, false, false);
    EXPECT_EQ(frame.renderPasses.size(), 4u);
  }
}

TEST_F(GfxRecordingTest, DepthWriteRequiresDepthCompare) {
  gx::g_gxState.colorUpdate = true;
  gx::g_gxState.alphaUpdate = false;
  gx::g_gxState.depthCompare = false;
  gx::g_gxState.depthUpdate = true;
  set_viewport(SmallViewport);
  set_scissor(SmallRect);
  push_draw_command(gx::DrawData{});
  EXPECT_EQ(frame.renderPasses[0].writeMask, WriteColor);
  copy(make_target(256, 256));
  EXPECT_EQ(frame.renderPasses.size(), 1u);
}

TEST_F(GfxRecordingTest, ClearDrawsDoNotFeedTheWriteMask) {
  push_draw_command(clear::DrawData{});
  EXPECT_EQ(frame.renderPasses[0].writeMask, 0u);
  gx_draw(true, true, true);
  EXPECT_EQ(frame.renderPasses[0].writeMask, WriteColor | WriteAlpha | WriteDepth);
}

TEST_F(GfxRecordingTest, CopyClearMustMatchPassStartClear) {
  small_segment();
  copy(make_target(256, 256), SmallRect, true, false, false, {1.f, 0.f, 0.f, 1.f});
  EXPECT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);

  // The continuation opens with a red clear draw; a copy clearing to another color does not fuse either.
  small_segment();
  copy(make_target(256, 256));
  EXPECT_EQ(frame.renderPasses.size(), 3u);
}

TEST_F(GfxRecordingTest, FirstSegmentMustStayLeftOfTheSecondRegion) {
  small_segment({0.f, 0.f, 512.f, 256.f, 0.f, 1.f}, {0, 0, 512, 256});
  copy(make_target(256, 256));
  EXPECT_EQ(frame.renderPasses.size(), 2u);
}

TEST_F(GfxRecordingTest, FusionNeedsRoomForTheSecondCopy) {
  small_segment({0.f, 0.f, 400.f, 256.f, 0.f, 1.f}, {0, 0, 400, 256});
  copy(make_target(400, 256), {0, 0, 400, 256});
  EXPECT_EQ(frame.renderPasses.size(), 2u);
}

TEST_F(GfxRecordingTest, FusionCanBeDisabledByConfig) {
  g_config.disableRenderPassFusion = true;
  small_segment();
  copy(make_target(256, 256));
  EXPECT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
}

} // namespace
} // namespace aurora::gfx
