#pragma once

#include "types.hpp"

#include <functional>

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

using NewPipelineCallback = std::function<wgpu::RenderPipeline()>;

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();

template <typename Config>
PipelineRef find_pipeline(ShaderType type, const Config& config, NewPipelineCallback&& cb);

bool get_pipeline(PipelineRef ref, wgpu::RenderPipeline& pipeline);
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
