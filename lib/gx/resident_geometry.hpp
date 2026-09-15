#pragma once

#include "gx.hpp"
#include "vertex_loader.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// Resident display-list geometry (AuroraConfig::residentDisplayLists, requires cpuVertexDecode).
//
// Titles built on static model formats draw the same display lists, indexing the same vertex arrays,
// every frame. With CPU vertex decoding each call would decode and upload the list again. Instead,
// GXCallDisplayList sends a reference (GX_AURORA_CALL_DL) and the command processor decodes each
// distinct (list, length, attribute configuration, array bases, current PN matrix) once into a
// GPU-resident vertex arena; later calls only append 32-bit indices into the arena. Measured on a
// Mali-G52 handheld (Miyoo Flip) running Melee's Onett stage: 447 list calls per frame, all resident
// after warm-up (~2 MB), FIFO translation worker 24.4 -> 16.4 ms/frame and per-frame vertex uploads
// ~3.8 MB -> ~0.3 MB.
//
// Validation: every call rehashes the first and last 64 bytes of the list plus its length, so a list
// rewritten in place is decoded again. Memory that held lists or arrays must be released through
// GXInvalidateResidentGeometry() (GX_AURORA_INVALIDATE_RESIDENT) before it is reused; entries are
// indexed by every pointer they depend on, so a release is a range query over that index rather than
// a scan of all entries.
//
// Arenas are append-only. Uploads travel with the frame op recorded while they were decoded
// (gfx::ArenaUpload) and are copied into the arena buffers through a staging buffer before that op
// is encoded, so an arena is never written while the GPU may be reading the same bytes. When the
// decoded bytes would exceed the budget the cache stops accepting lists (they are processed inline)
// and is emptied at the next frame boundary.
//
// Everything here runs on the FIFO processing thread (or the game thread while the processor is
// idle); the GPU side lives in pipeline.cpp and runs on the render thread.
namespace aurora::gx::resident {

// Decoded bytes kept resident when AuroraConfig::residentGeometryBudget is zero.
constexpr size_t DefaultBudget = 64 * 1024 * 1024;

struct Entry {
  u32 arena = 0;       // arena index (one arena per decoded record stride)
  u32 firstVertex = 0; // first record of this list within the arena
  u32 vertexCount = 0;
  std::vector<u32> indices; // triangle list of absolute arena vertex indices
  const void* list = nullptr;
  u32 listBytes = 0;
  u64 listHash = 0;                             // list_hash() when decoded
  std::array<const void*, MaxVtxAttr> arrays{}; // bases of the indexed arrays the records came from
};

struct Stats {
  u64 calls = 0;
  u64 hits = 0;
  u64 misses = 0;
  u64 fallbacks = 0;   // lists processed inline
  u64 invalidated = 0; // entries removed by invalidate()
  u64 resets = 0;
};

// Cheap content check: XXH3 of the first and last 64 bytes of the list and its length.
u64 list_hash(const u8* list, u32 size) noexcept;

// Identity of a decoded list: the list, the attribute configuration it decodes with, the arrays
// its indexed attributes read and the PN matrix baked into records without a matrix index.
u64 entry_key(const void* list, u32 size, const ShaderConfig& config, const std::array<AttrArray, MaxVtxAttr>& arrays,
              u32 currentPnMtx) noexcept;

Entry* find(u64 key) noexcept;
Entry& insert(u64 key, Entry entry);
void erase(u64 key) noexcept;
// Removes every entry depending on memory in [base, base + size); returns how many.
size_t invalidate(const void* base, size_t size) noexcept;

// Decodes a display list made only of triangle-class draws of vertex format `fmt` (GX_NOPs allowed,
// GX_AURORA_DRAW_INDEXED accepted) with `loader` into the arena for its record stride and fills `out`
// (list, listBytes and listHash are left to the caller). Returns false, leaving the cache untouched,
// when the list holds anything else, is malformed, or does not fit the budget.
bool decode_display_list(const u8* list, u32 size, GXVtxFmt fmt, const VertexLoader& loader,
                         const std::array<AttrArray, MaxVtxAttr>& arrays, u32 currentPnMtx, Entry& out);

u32 arena_stride(u32 arena) noexcept;
size_t used_bytes() noexcept;
size_t entry_count() noexcept;
Stats& stats() noexcept;

// Bytes decoded since the previous call, for the frame op about to be recorded.
std::vector<gfx::ArenaUpload> take_uploads();
// Frame boundary (processor idle): performs a reset requested by an over-budget decode.
void end_frame() noexcept;
// Every index entry names an existing entry that depends on its pointer, and each dependency is
// indexed exactly once.
bool index_consistent() noexcept;
void shutdown() noexcept;

// Render thread (pipeline.cpp).
void encode_uploads(wgpu::CommandEncoder& encoder, const std::vector<gfx::ArenaUpload>& uploads);
// Buffer of an arena (null before its first upload) and its size in bytes.
const wgpu::Buffer& arena_buffer(u32 arena, uint64_t& size) noexcept;
void release_buffers() noexcept;

} // namespace aurora::gx::resident
