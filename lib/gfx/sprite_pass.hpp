#pragma once
// Half-resolution sprite pass (AuroraConfig::halfResolutionSpritePoints).
//
// Particle systems draw tens of thousands of blended point sprites per frame; at the EFB resolution their
// fill cost dominates the GPU on mobile parts (a particle-heavy stage: ~25k sprites, ~8 ms of fill at
// 640x480 on a Mali-G52 handheld). When the previous frame drew at least `halfResolutionSpritePoints` points, runs of
// eligible point draws (SRCALPHA blends onto INVSRCALPHA or ONE, or alpha-tested opaque sprites, color
// writes on, no destination alpha) are recorded into a half-size color target with a downsampled copy of
// the scene depth, accumulating premultiplied color and coverage (ShaderConfig::spriteAccumulate), and
// composited back onto the scene with one full-screen draw. Depth-writing sprites only update the half-size
// depth, so geometry drawn after them is not occluded by them: an approximation, hence the opt-in threshold.
// Measured on the device above: mean frame 26.6 -> 23.9 ms, p95 50 -> 36 ms on that stage; scenes below the
// threshold are untouched.
#include "../webgpu/gpu.hpp"

#include <aurora/gfx.hpp>

namespace aurora::gfx::sprite_pass {
bool enabled() noexcept;
uint32_t threshold() noexcept;
// Half-size color/depth targets sized to the current EFB (created on demand; render worker or encode time).
const webgpu::TextureWithSampler& color_target();
const webgpu::TextureWithSampler& depth_target();
// Encoder tasks: prepare (clear the color target, downsample the scene depth) and composite.
EncoderTaskId prep_task();
EncoderTaskId composite_task();
// The scene color view the composite draws onto (set per frame op by the encoder).
void set_scene_view(wgpu::TextureView view);
void shutdown();
} // namespace aurora::gfx::sprite_pass
