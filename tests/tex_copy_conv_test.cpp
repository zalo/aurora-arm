#include <gtest/gtest.h>

#include "gfx/tex_copy_conv.hpp"

#include <string_view>

namespace aurora::gfx::tex_copy_conv {
namespace {

constexpr std::string_view SingleTargetShader = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    let i = intensity(c.rgb);
    return vec4f(i, i, i, 1.0);
}
)";

TEST(TexCopyConvTest, DualFragmentSourceRewritesSingleTargetShader) {
  const auto dual = dual_fragment_source(SingleTargetShader);
  ASSERT_FALSE(dual.empty());

  // The single-target entry point became a function of the UV, sampling through its parameter.
  const auto conv = dual.find("fn conv(uv: vec2f) -> vec4f {");
  ASSERT_NE(conv, std::string::npos);
  EXPECT_EQ(dual.find("@location(0) vec4f"), std::string::npos);
  const auto epilogue = dual.find("struct DualOutput");
  ASSERT_NE(epilogue, std::string::npos);
  EXPECT_EQ(dual.substr(conv, epilogue - conv).find("in.uv"), std::string::npos);
  EXPECT_NE(dual.find("textureSample(src, src_samp, uv)"), std::string::npos);

  // The two-target entry point converts each attachment from its own UV transform.
  EXPECT_NE(dual.find("@fragment fn fs_main(in: VertexOutput) -> DualOutput {"), std::string::npos);
  EXPECT_NE(dual.find("out.a = conv(in.uv);"), std::string::npos);
  EXPECT_NE(dual.find("out.b = conv(in.uv2);"), std::string::npos);
}

TEST(TexCopyConvTest, DualFragmentSourceRejectsOtherEntryPoints) {
  EXPECT_TRUE(dual_fragment_source(R"(
@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    return vec4f(0.0);
}
)")
                  .empty());
}

} // namespace
} // namespace aurora::gfx::tex_copy_conv
