#pragma once

#include "gx.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// CPU vertex decoding (AuroraConfig::cpuVertexDecode).
//
// GX vertex streams are converted on the CPU into interleaved float records that the vertex shader
// consumes through conventional @location inputs, instead of fetching and converting them from
// storage buffers per vertex. The decoded record layout is a function of the shader configuration:
//
//   location  input            contents
//   0         v_matrices vec3u one matrix index per byte (PNMTXIDX, TEX0..7MTXIDX), quad corner in
//                              bits 24-31 of .z for lines/points
//   1         v_pos      vec3f
//   2         v_nrm      vec3f
//   3, 4      v_clr0/1   vec4f
//   5-12      v_tex0-7   vec2f
//   13, 14    v_binrm, v_tangent vec3f (GX_NRM_NBT / GX_NRM_NBT3)
//   15        v_line_end vec4f end position and end PN matrix index (GX_LINES / GX_LINESTRIP)
//
// A VertexLoader is built once per distinct attribute configuration (Dolphin VertexLoaderManager
// style): it holds the decoded layout and one monomorphic conversion routine per attribute with all
// format decisions hoisted out of the per-vertex loop. Every byte of the output record is written
// exactly once, so the destination needs no clearing and can be frame storage that is only ever
// written. Measured on a Mali-G52 handheld (Miyoo Flip) running Melee, this took the FIFO
// translation worker from 28.6 to 24.4 ms/frame on the Onett stage.
namespace aurora::gx {

constexpr u32 MaxDecodedVertexAttrs = 16;
// Size of a record with every location present.
constexpr u32 MaxDecodedVertexStride = 12 + 12 + 12 + 16 * 2 + 8 * 8 + 12 + 12 + 16;

struct DecodedVertexLayout {
  std::array<wgpu::VertexAttribute, MaxDecodedVertexAttrs> attributes{};
  u32 count = 0;
  u32 stride = 0;
};

// Whether shader location `location` is part of the decoded record for `config`.
bool decoded_vertex_has_location(const ShaderConfig& config, u32 location) noexcept;
// Byte size of the attribute at shader location `location`.
u32 decoded_vertex_attr_size(u32 location) noexcept;
DecodedVertexLayout decoded_vertex_layout(const ShaderConfig& config) noexcept;

struct VertexLoaderStep {
  using Fn = void (*)(const VertexLoaderStep& step, const u8* raw, size_t count, size_t vtxStride, u8* out,
                      size_t outStride, const u8* array, size_t arraySize);
  Fn fn = nullptr;
  u8 attr = 0;       // GXAttr (selects the indexed array)
  u8 indexBytes = 0; // 0 = GX_DIRECT, 1 = GX_INDEX8, 2 = GX_INDEX16
  u8 components = 0; // components in the destination attribute (unwritten ones read as zero)
  u8 srcOffset = 0;  // within the raw GX vertex
  u16 dstOffset = 0; // within the decoded record
  u32 stride = 0;    // indexed array stride
  u32 within = 0;    // byte offset inside the array record (NBT slices), or bytes to clear
  float scale = 1.f; // 1 / 2^frac
};

struct VertexLoader {
  DecodedVertexLayout layout{};
  std::vector<VertexLoaderStep> steps;
  u8 vtxStride = 0; // raw GX vertex size
  u8 lineMode = 0;  // ShaderConfig::lineMode
  bool hasMatrices = false;
  u16 matrixOffset = 0;           // location 0
  u16 posOffset = 0;              // location 1
  u16 lineEndOffset = UINT16_MAX; // location 15, UINT16_MAX when absent
};

// Returns the loader for `config`, building it on first use. Loaders are cached by the hash of the
// attribute configuration; the cache is not synchronized and belongs to the FIFO processing thread.
const VertexLoader& vertex_loader(const ShaderConfig& config) noexcept;

// Decodes `count` GX vertices of `loader.vtxStride` bytes each from `raw`, writing exactly
// count * loader.layout.stride bytes to `out`. Indexed attributes read from `arrays`; indices
// outside an array decode as zero. `currentPnMtx` fills the PN matrix index when the vertex does
// not carry one.
void decode_vertices(const VertexLoader& loader, const u8* raw, size_t count, u8* out,
                     const std::array<AttrArray, MaxVtxAttr>& arrays, u32 currentPnMtx) noexcept;

// Number of quads produced by `vtxCount` vertices of a GX_LINES (1), GX_LINESTRIP (2) or GX_POINTS
// (3) draw.
u32 line_instance_count(u8 lineMode, u32 vtxCount) noexcept;

// Expands `instances` line segments or points from records decoded with `loader` into four quad
// corners each, writing exactly instances * 4 * loader.layout.stride bytes to `out`. Every corner
// carries the start position and PN matrix index; corners 2 and 3 take the remaining attributes of
// the end vertex; the end position and PN matrix index travel in v_line_end and the corner index in
// bits 24-31 of v_matrices.z, matching the instanced expansion of the storage-buffer vertex shader.
// Records are assembled in local memory and streamed to `out`, which is never read back: on a
// write-combined mapped destination a single 4-byte read per record cost ~20 ms/frame.
void expand_line_vertices(const VertexLoader& loader, const u8* decoded, size_t instances, u8* out) noexcept;

} // namespace aurora::gx
