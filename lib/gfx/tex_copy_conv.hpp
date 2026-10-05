#pragma once

#include "types.hpp"

#include <dolphin/gx/GXEnum.h>

#include <string>
#include <string_view>
#include <array>

namespace aurora::gfx::tex_copy_conv {

enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct alignas(16) Uniforms {
  Vec2<float> offset;
  Vec2<float> scale{1.f, 1.f};
  uint32_t opaqueAlpha = 0;
  std::array<uint32_t, 3> _pad{};
};
static_assert(sizeof(Uniforms) == 32);

// Uniforms of the two-target conversion (run_dual): one transform per target.
struct alignas(16) DualUniforms {
  Vec2<float> offset;
  Vec2<float> scale{1.f, 1.f};
  Vec2<float> offset2;
  Vec2<float> scale2{1.f, 1.f};
  uint32_t opaqueAlpha = 0;
  std::array<uint32_t, 3> _pad{};
};
static_assert(sizeof(DualUniforms) == 48);

struct ConvRequest {
  GXTexFmt fmt;
  GXPixelFmt srcFmt;
  wgpu::TextureView srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // Uniforms
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
};

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
// Two same-format, unscaled conversions from one source in a single render pass with two color targets.
// `dualUniformRange` holds a DualUniforms; `req.dst` receives the first, `dst2` the second.
bool dual_supported(GXTexFmt fmt);
void run_dual(const wgpu::CommandEncoder& cmd, const ConvRequest& req, const TextureHandle& dst2,
              Range dualUniformRange);
// Rewrites a single-target conversion fragment shader into the `conv(uv)` function the two-target entry point calls
// once per attachment. Empty when the shader does not have the expected entry point. Exposed for tests.
std::string dual_fragment_source(std::string_view fragShader);

bool snapshot_depth_supported() noexcept;
void snapshot_depth(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& srcDepth, uint32_t msaaSamples,
                    const wgpu::TextureView& dst);

} // namespace aurora::gfx::tex_copy_conv
