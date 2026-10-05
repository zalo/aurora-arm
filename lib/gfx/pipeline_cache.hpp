#pragma once

#include "types.hpp"

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

struct CompiledPipeline {
  wgpu::RenderPipeline main;
  wgpu::RenderPipeline prepass;

  [[nodiscard]] uint32_t pipeline_count() const {
    return static_cast<uint32_t>(static_cast<bool>(main)) + static_cast<uint32_t>(static_cast<bool>(prepass));
  }
};

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();
void rebuild_pipeline_cache();

PipelineRef find_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const rmlui::PipelineConfig& config);

bool get_pipeline(PipelineRef ref, CompiledPipeline& pipeline);
// Bounded cache (AURORA_PIPELINE_CACHE_MAX, RAM-scaled by default). Changes whenever pipelines are retired:
// whoever memoizes PipelineRefs outside this cache drops them then and asks find_pipeline again, which is what
// keeps a pipeline in use alive (or recompiles one that was released).
uint64_t pipeline_cache_generation() noexcept;
// Render worker, once per frame: drops the pipelines that stayed retired for a whole sweep period.
void release_retired_pipelines();
size_t live_pipeline_count() noexcept;
uint64_t evicted_pipeline_count() noexcept;
// Renderer time accounting (AuroraConfig::renderStats): time and count of blocking pipeline waits.
uint64_t pipeline_wait_ns() noexcept;
uint64_t pipeline_wait_count() noexcept;

namespace detail::testing {
// Queues new pipelines instead of creating them, for host tests that record frames without a GPU device.
void suppress_pipeline_creation(bool suppress) noexcept;
} // namespace detail::testing

} // namespace aurora::gfx
