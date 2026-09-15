#pragma once

#include "types.hpp"

#include <dolphin/gx/GXEnum.h>

#include <string>
#include <string_view>

namespace aurora::gfx::tex_copy_conv {

enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct ConvRequest {
  GXTexFmt fmt;
  wgpu::TextureView srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // UV transform uniform (offset + scale)
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
};

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
// Two same-format, unscaled conversions from one source in a single render pass with two color targets.
// `dualUniformRange` holds both UV transforms (32 bytes: offset, scale, offset2, scale2); `req.dst` receives the
// first, `dst2` the second.
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
