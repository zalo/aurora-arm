#pragma once

#include "../internal.hpp"

#include <cstdint>

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::gx::fifo {

struct ProcessResult {
  uint32_t bytesProcessed;
  bool drawDone;
};

// Process GX FIFO commands until the next draw done event or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;
// Forgets every resolved pipeline (draw cache and pipeline-state memo). Called on
// shutdown, before the pipeline cache is emptied.
void reset_pipeline_memo() noexcept;

namespace testing {
// Number of dirty-pipeline events that rebuilt the pipeline config (memo misses).
uint32_t pipeline_memo_misses() noexcept;
// Pipeline config resolved for the most recent draw.
const PipelineConfig& cached_pipeline_config() noexcept;
} // namespace testing

} // namespace aurora::gx::fifo
