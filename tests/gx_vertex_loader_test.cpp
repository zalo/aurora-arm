// CPU vertex decoder tests
//
// The specialized vertex loaders (lib/gx/vertex_loader.cpp) are checked against a straightforward
// per-vertex reference decoder on randomized attribute configurations, vertex data and arrays,
// including indices outside the arrays. Output must be byte-identical, and every output byte must
// be written (the loaders never clear their destination).

#include "gx/vertex_loader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <random>
#include <vector>

namespace aurora::gx {
namespace {
// Reference decoded vertex: one field per shader location.
struct Vertex {
  u32 matrices[3]{};
  float pos[3]{};
  float nrm[3]{};
  float clr[2][4]{};
  float tex[8][2]{};
  float binrm[3]{};
  float tangent[3]{};
  float lineEnd[4]{};
};

u32 numeric_comp_size(u32 compType) {
  return compType == GX_F32 ? 4 : (compType == GX_U16 || compType == GX_S16) ? 2 : 1;
}

u32 color_size(u32 compType) {
  return (compType == GX_RGB565 || compType == GX_RGBA4) ? 2 : (compType == GX_RGB8 || compType == GX_RGBA6) ? 3 : 4;
}

u32 read_integer(const u8* p, u32 bytes, bool le) {
  u32 v = 0;
  for (u32 i = 0; i < bytes; ++i) {
    v |= u32(p[i]) << ((le ? i : bytes - i - 1) * 8);
  }
  return v;
}

// Source bytes of one attribute slice, following the storage-buffer shader's addressing
// (attr_address / attr_load_nbt_slice): indexed records outside the array read as zero.
const u8* reference_address(const AttrConfig& m, const AttrArray& array, const u8* vertex, u32 slice, u32 bytes) {
  static constexpr u8 Zero[64]{};
  const u32 within = slice * 3 * numeric_comp_size(m.compType);
  if (m.attrType == GX_DIRECT) {
    return vertex + m.offset + within;
  }
  const u32 indexBytes = m.attrType == GX_INDEX8 ? 1 : 2;
  const u32 index = read_integer(vertex + m.offset + (m.nbt3 ? slice * indexBytes : 0), indexBytes, false);
  const u64 offset = u64(index) * m.stride + within;
  if (array.data == nullptr || offset + bytes > array.size) {
    return Zero;
  }
  return static_cast<const u8*>(array.data) + offset;
}

Vertex reference_decode(const ShaderConfig& c, const u8* vertex, const std::array<AttrArray, MaxVtxAttr>& arrays,
                        u32 currentPnMtx) {
  Vertex out{};
  out.matrices[0] = currentPnMtx;
  for (u32 ai = GX_VA_PNMTXIDX; ai <= GX_VA_TEX7; ++ai) {
    const auto attr = static_cast<GXAttr>(ai);
    const auto& m = c.attrs[ai];
    if (m.attrType == GX_NONE) {
      continue;
    }
    const bool le = m.attrType != GX_DIRECT && m.le;
    if (attr <= GX_VA_TEX7MTXIDX) {
      u32 v = *reference_address(m, arrays[ai], vertex, 0, 1);
      if (attr == GX_VA_PNMTXIDX) {
        v /= 3;
      }
      out.matrices[ai / 4] &= ~(255u << ((ai % 4) * 8));
      out.matrices[ai / 4] |= v << ((ai % 4) * 8);
      continue;
    }
    if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
      float* dst = out.clr[attr - GX_VA_CLR0];
      const u32 bytes = color_size(m.compType);
      const u8* p = reference_address(m, arrays[ai], vertex, 0, bytes);
      const u32 v = read_integer(p, bytes, le);
      if (m.compType == GX_RGB565) {
        dst[0] = float((v >> 11) & 31) / 31.f;
        dst[1] = float((v >> 5) & 63) / 63.f;
        dst[2] = float(v & 31) / 31.f;
        dst[3] = 1.f;
      } else if (m.compType == GX_RGBA4 || m.compType == GX_RGBA6) {
        const u32 bits = m.compType == GX_RGBA4 ? 4 : 6;
        const u32 mask = (1u << bits) - 1;
        for (u32 j = 0; j < 4; ++j) {
          dst[j] = float((v >> ((3 - j) * bits)) & mask) / float(mask);
        }
      } else {
        for (u32 j = 0; j < 3; ++j) {
          dst[j] = float(p[j]) / 255.f;
        }
        dst[3] = m.compType == GX_RGBA8 ? float(p[3]) / 255.f : 1.f;
      }
      continue;
    }
    const u32 compSize = numeric_comp_size(m.compType);
    const u32 count = attr == GX_VA_NRM ? 3 : m.cnt;
    const u32 slices = attr == GX_VA_NRM && m.cnt == 9 ? 3 : 1;
    for (u32 slice = 0; slice < slices; ++slice) {
      float* dst = attr == GX_VA_POS   ? out.pos
                   : attr == GX_VA_NRM ? (slice == 0   ? out.nrm
                                          : slice == 1 ? out.binrm
                                                       : out.tangent)
                                       : out.tex[attr - GX_VA_TEX0];
      const u8* p = reference_address(m, arrays[ai], vertex, slice, count * compSize);
      for (u32 j = 0; j < count; ++j) {
        const u32 v = read_integer(p + j * compSize, compSize, le);
        if (m.compType == GX_F32) {
          dst[j] = std::bit_cast<float>(v);
        } else {
          const s32 n = m.compType == GX_S8 ? s8(v) : m.compType == GX_S16 ? s16(v) : s32(v);
          dst[j] = float(n) / float(1u << m.frac);
        }
      }
    }
  }
  return out;
}

// Interleaves reference vertices with the decoded layout.
std::vector<u8> reference_pack(const ShaderConfig& c, const std::vector<Vertex>& vertices) {
  const auto layout = decoded_vertex_layout(c);
  std::vector<u8> out(vertices.size() * layout.stride);
  for (size_t i = 0; i < vertices.size(); ++i) {
    const auto& v = vertices[i];
    for (u32 a = 0; a < layout.count; ++a) {
      const auto& attr = layout.attributes[a];
      const u32 location = attr.shaderLocation;
      const void* src = location == 0    ? static_cast<const void*>(v.matrices)
                        : location == 1  ? v.pos
                        : location == 2  ? v.nrm
                        : location < 5   ? v.clr[location - 3]
                        : location < 13  ? v.tex[location - 5]
                        : location == 13 ? v.binrm
                        : location == 14 ? v.tangent
                                         : v.lineEnd;
      std::memcpy(out.data() + i * layout.stride + attr.offset, src, decoded_vertex_attr_size(location));
    }
  }
  return out;
}

// Reference quad expansion, mirroring the instanced line/point vertex shader: every corner takes
// the start position and PN matrix index, corners 2 and 3 the end vertex's other attributes.
std::vector<Vertex> reference_expand(u8 lineMode, const std::vector<Vertex>& decoded) {
  const size_t instances = line_instance_count(lineMode, static_cast<u32>(decoded.size()));
  std::vector<Vertex> out(instances * 4);
  for (size_t i = 0; i < instances; ++i) {
    const auto& a = decoded[lineMode == 1 ? i * 2 : i];
    const auto& b = decoded[lineMode == 3 ? i : lineMode == 1 ? i * 2 + 1 : i + 1];
    for (u32 corner = 0; corner < 4; ++corner) {
      auto& v = out[i * 4 + corner];
      v = corner >= 2 ? b : a;
      v.matrices[2] |= corner << 24;
      std::memcpy(v.pos, a.pos, sizeof(v.pos));
      v.matrices[0] = (v.matrices[0] & ~255u) | (a.matrices[0] & 255u);
      std::memcpy(v.lineEnd, b.pos, sizeof(b.pos));
      v.lineEnd[3] = float(b.matrices[0] & 255u);
    }
  }
  return out;
}

struct Scenario {
  ShaderConfig config{};
  std::vector<std::vector<u8>> arrayData;
  std::array<AttrArray, MaxVtxAttr> arrays{};
  std::vector<u8> raw;
  u32 count = 0;
  u32 currentPnMtx = 0;
};

// Random attribute configuration with vertex offsets assigned like populate_pipeline_config.
Scenario make_scenario(std::mt19937& rng, u8 lineMode) {
  const auto chance = [&](u32 n) { return rng() % n == 0; };
  const auto pick = [&](auto... values) {
    const std::array options{static_cast<u32>(values)...};
    return options[rng() % options.size()];
  };
  Scenario s;
  s.config.lineMode = lineMode;
  s.currentPnMtx = rng() % 10;
  s.arrayData.resize(MaxVtxAttr);
  u8 offset = 0;
  for (u32 ai = GX_VA_PNMTXIDX; ai <= GX_VA_TEX7; ++ai) {
    const auto attr = static_cast<GXAttr>(ai);
    auto& m = s.config.attrs[ai];
    const bool present = attr == GX_VA_POS || (attr == GX_VA_PNMTXIDX && chance(2)) ||
                         (attr <= GX_VA_TEX7MTXIDX && chance(4)) || (attr == GX_VA_NRM && chance(2)) ||
                         ((attr == GX_VA_CLR0 || attr == GX_VA_CLR1) && chance(2)) || (attr >= GX_VA_TEX0 && chance(3));
    if (!present) {
      continue;
    }
    // Matrix indices are direct in practice; indexed forms are exercised rarely.
    m.attrType = attr <= GX_VA_TEX7MTXIDX ? (chance(8) ? pick(GX_INDEX8, GX_INDEX16) : u32(GX_DIRECT))
                                          : pick(GX_DIRECT, GX_INDEX8, GX_INDEX16);
    u32 recordBytes = 0;
    if (attr <= GX_VA_TEX7MTXIDX) {
      m.cnt = 1;
      m.compType = GX_U8;
      recordBytes = 1;
    } else if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
      m.cnt = 1;
      m.compType = pick(GX_RGB565, GX_RGB8, GX_RGBX8, GX_RGBA4, GX_RGBA6, GX_RGBA8);
      recordBytes = color_size(m.compType);
    } else {
      m.cnt = attr == GX_VA_POS ? pick(2, 3) : attr == GX_VA_NRM ? pick(3, 9) : pick(1, 2);
      m.compType = attr == GX_VA_NRM ? pick(GX_S8, GX_S16, GX_F32) : pick(GX_U8, GX_S8, GX_U16, GX_S16, GX_F32);
      m.frac = m.compType == GX_F32 ? 0 : rng() % 16;
      m.nbt3 = attr == GX_VA_NRM && m.cnt == 9 && chance(2);
      recordBytes = numeric_comp_size(m.compType) * m.cnt;
    }
    m.offset = offset;
    if (m.attrType == GX_DIRECT) {
      offset += recordBytes;
      continue;
    }
    const u32 indexBytes = m.attrType == GX_INDEX8 ? 1 : 2;
    offset += indexBytes * (m.nbt3 ? 3 : 1);
    m.le = chance(2);
    m.stride = static_cast<u8>(recordBytes + rng() % 9);
    auto& data = s.arrayData[ai];
    if (!chance(16)) { // occasionally leave the array unbound
      data.resize(static_cast<size_t>(m.stride) * (1 + rng() % 300) + rng() % 7);
      for (auto& byte : data) {
        byte = static_cast<u8>(rng());
      }
      s.arrays[ai] = {.data = data.data(), .size = static_cast<u32>(data.size()), .stride = m.stride, .le = m.le};
    }
  }
  s.config.vtxStride = offset;
  s.count = 1 + rng() % 64;
  s.raw.resize(static_cast<size_t>(s.count) * offset);
  for (auto& byte : s.raw) {
    byte = static_cast<u8>(rng());
  }
  // Indices mostly land inside the array, sometimes beyond it.
  for (u32 ai = GX_VA_PNMTXIDX; ai <= GX_VA_TEX7; ++ai) {
    const auto& m = s.config.attrs[ai];
    if (m.attrType != GX_INDEX8 && m.attrType != GX_INDEX16) {
      continue;
    }
    const u32 indexBytes = m.attrType == GX_INDEX8 ? 1 : 2;
    const u32 records = static_cast<u32>(s.arrayData[ai].size() / std::max<u32>(m.stride, 1));
    for (u32 i = 0; i < s.count; ++i) {
      for (u32 slice = 0; slice < (m.nbt3 ? 3u : 1u); ++slice) {
        u32 index = chance(8) ? rng() % 65536 : rng() % (records + 8);
        if (indexBytes == 1) {
          index &= 255;
        }
        u8* p = s.raw.data() + i * offset + m.offset + slice * indexBytes;
        if (indexBytes == 2) {
          *p++ = static_cast<u8>(index >> 8);
        }
        *p = static_cast<u8>(index);
      }
    }
  }
  return s;
}

std::vector<Vertex> reference_decode_all(const Scenario& s) {
  std::vector<Vertex> vertices(s.count);
  for (u32 i = 0; i < s.count; ++i) {
    vertices[i] = reference_decode(s.config, s.raw.data() + i * s.config.vtxStride, s.arrays, s.currentPnMtx);
  }
  return vertices;
}

// Decodes into storage prefilled with `fill` so that unwritten bytes would show.
std::vector<u8> decode_with_loader(const Scenario& s, u8 fill) {
  const auto& loader = vertex_loader(s.config);
  std::vector<u8> out(static_cast<size_t>(s.count) * loader.layout.stride, fill);
  decode_vertices(loader, s.raw.data(), s.count, out.data(), s.arrays, s.currentPnMtx);
  return out;
}
} // namespace

TEST(GXVertexLoader, LayoutFollowsAttributeConfig) {
  ShaderConfig c{};
  c.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  auto layout = decoded_vertex_layout(c);
  EXPECT_EQ(layout.count, 1u);
  EXPECT_EQ(layout.stride, 12u);
  EXPECT_EQ(layout.attributes[0].shaderLocation, 1u);
  EXPECT_EQ(layout.attributes[0].format, wgpu::VertexFormat::Float32x3);

  c.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8};
  c.attrs[GX_VA_CLR1] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_RGBA8};
  layout = decoded_vertex_layout(c);
  EXPECT_EQ(layout.count, 3u);
  EXPECT_EQ(layout.stride, 12u + 12u + 16u);
  EXPECT_EQ(layout.attributes[0].shaderLocation, 0u);
  EXPECT_EQ(layout.attributes[2].shaderLocation, 4u);
  EXPECT_EQ(layout.attributes[2].offset, 24u);

  // Lines carry the matrix word and the line end; points only the matrix word
  c = {};
  c.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  c.lineMode = 1;
  EXPECT_TRUE(decoded_vertex_has_location(c, 0));
  EXPECT_TRUE(decoded_vertex_has_location(c, 15));
  c.lineMode = 3;
  EXPECT_TRUE(decoded_vertex_has_location(c, 0));
  EXPECT_FALSE(decoded_vertex_has_location(c, 15));

  // Every location present
  for (u32 ai = GX_VA_PNMTXIDX; ai <= GX_VA_TEX7; ++ai) {
    c.attrs[ai] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8};
  }
  c.attrs[GX_VA_NRM].cnt = 9;
  c.lineMode = 2;
  layout = decoded_vertex_layout(c);
  EXPECT_EQ(layout.count, MaxDecodedVertexAttrs);
  EXPECT_EQ(layout.stride, MaxDecodedVertexStride);
}

TEST(GXVertexLoader, DecodesKnownValues) {
  std::array<AttrArray, MaxVtxAttr> arrays{};
  ShaderConfig c{};
  c.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 0};
  c.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_S16, .offset = 1, .frac = 4};
  c.vtxStride = 7;
  const auto& loader = vertex_loader(c);
  EXPECT_EQ(loader.layout.stride, 24u);
  // Big-endian signed 16.4 fixed point: -32, 24, 256 -> -2, 1.5, 16; PN matrix index 9 -> 3
  const u8 direct[] = {9, 0xff, 0xe0, 0, 24, 1, 0};
  std::array<u8, 24> out{};
  decode_vertices(loader, direct, 1, out.data(), arrays, 0);
  u32 matrices[3];
  float pos[3];
  std::memcpy(matrices, out.data(), 12);
  std::memcpy(pos, out.data() + 12, 12);
  EXPECT_EQ(matrices[0], 3u);
  EXPECT_EQ(matrices[1], 0u);
  EXPECT_EQ(matrices[2], 0u);
  EXPECT_EQ(pos[0], -2.f);
  EXPECT_EQ(pos[1], 1.5f);
  EXPECT_EQ(pos[2], 16.f);

  // Big-endian 16-bit index into a little-endian float XY array; index 2 is out of range
  c = {};
  c.attrs[GX_VA_POS] = {.attrType = GX_INDEX16, .cnt = 2, .compType = GX_F32, .offset = 0, .stride = 8, .le = true};
  c.vtxStride = 2;
  const float array[] = {0.f, 0.f, 3.25f, -7.5f};
  arrays[GX_VA_POS] = {.data = array, .size = sizeof(array), .stride = 8, .le = true};
  const u8 indexed[] = {0, 1, 0, 2};
  const auto& indexedLoader = vertex_loader(c);
  EXPECT_EQ(indexedLoader.layout.stride, 12u);
  decode_vertices(indexedLoader, indexed, 2, out.data(), arrays, 0);
  std::memcpy(pos, out.data(), 12);
  EXPECT_EQ(pos[0], 3.25f);
  EXPECT_EQ(pos[1], -7.5f);
  EXPECT_EQ(pos[2], 0.f);
  std::memcpy(pos, out.data() + 12, 12);
  EXPECT_EQ(pos[0], 0.f);
  EXPECT_EQ(pos[1], 0.f);

  // RGB565 red with GX_NRM_NBT3: three independent indices select slices of nine-component records
  c = {};
  c.attrs[GX_VA_CLR0] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_RGB565, .offset = 0};
  c.attrs[GX_VA_NRM] = {.attrType = GX_INDEX8,
                        .cnt = 9,
                        .compType = GX_S8,
                        .offset = 2,
                        .stride = 9,
                        .frac = 6,
                        .le = true,
                        .nbt3 = true};
  c.vtxStride = 5;
  const u8 nbt[] = {64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64};
  arrays = {};
  arrays[GX_VA_NRM] = {.data = nbt, .size = sizeof(nbt), .stride = 9, .le = true};
  const u8 vertex[] = {0xf8, 0, 0, 1, 2};
  const auto& nbtLoader = vertex_loader(c);
  EXPECT_EQ(nbtLoader.layout.stride, 12u + 12u + 16u + 12u + 12u);
  std::array<u8, 64> nbtOut{};
  decode_vertices(nbtLoader, vertex, 1, nbtOut.data(), arrays, 0);
  float values[16];
  std::memcpy(values, nbtOut.data(), sizeof(values));
  EXPECT_EQ(values[3], 1.f); // nrm.x
  EXPECT_EQ(values[6], 1.f); // clr0.r
  EXPECT_EQ(values[7], 0.f);
  EXPECT_EQ(values[8], 0.f);
  EXPECT_EQ(values[9], 1.f);  // clr0.a
  EXPECT_EQ(values[11], 1.f); // binrm.y
  EXPECT_EQ(values[15], 1.f); // tangent.z
}

TEST(GXVertexLoader, CachesLoadersPerConfiguration) {
  ShaderConfig c{};
  c.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_S16, .frac = 4};
  c.vtxStride = 6;
  const auto* first = &vertex_loader(c);
  ShaderConfig same = c;
  same.tevStageCount = 5; // not part of the vertex format
  EXPECT_EQ(first, &vertex_loader(same));
  ShaderConfig lines = c;
  lines.lineMode = 1;
  EXPECT_NE(first, &vertex_loader(lines));
  EXPECT_EQ(first, &vertex_loader(c));
}

TEST(GXVertexLoader, MatchesReferenceDecoder) {
  std::mt19937 rng{0x6A7A6C64};
  for (u32 iteration = 0; iteration < 2000; ++iteration) {
    const auto s = make_scenario(rng, 0);
    const auto expected = reference_pack(s.config, reference_decode_all(s));
    const auto decoded = decode_with_loader(s, 0xCD);
    ASSERT_EQ(decoded.size(), expected.size()) << "iteration " << iteration;
    ASSERT_EQ(decoded, expected) << "iteration " << iteration;
    // Every byte is written: a differently prefilled destination decodes identically
    ASSERT_EQ(decode_with_loader(s, 0x00), expected) << "iteration " << iteration;
  }
}

TEST(GXVertexLoader, ExpandsLinesAndPointsLikeReference) {
  std::mt19937 rng{0x4C696E65};
  for (u32 iteration = 0; iteration < 600; ++iteration) {
    const u8 lineMode = 1 + iteration % 3;
    const auto s = make_scenario(rng, lineMode);
    const auto& loader = vertex_loader(s.config);
    ASSERT_TRUE(loader.hasMatrices);
    ASSERT_EQ(loader.lineEndOffset != UINT16_MAX, lineMode != 3);
    const auto vertices = reference_decode_all(s);
    const auto expected = reference_pack(s.config, reference_expand(lineMode, vertices));
    const auto decoded = decode_with_loader(s, 0xCD);
    ASSERT_EQ(decoded, reference_pack(s.config, vertices)) << "iteration " << iteration;
    const u32 instances = line_instance_count(lineMode, s.count);
    std::vector<u8> expanded(static_cast<size_t>(instances) * 4 * loader.layout.stride, 0xCD);
    expand_line_vertices(loader, decoded.data(), instances, expanded.data());
    ASSERT_EQ(expanded, expected) << "iteration " << iteration << " lineMode " << lineMode;
  }
}

TEST(GXVertexLoader, LineInstanceCounts) {
  EXPECT_EQ(line_instance_count(1, 7), 3u);
  EXPECT_EQ(line_instance_count(2, 7), 6u);
  EXPECT_EQ(line_instance_count(2, 0), 0u);
  EXPECT_EQ(line_instance_count(3, 7), 7u);
}
} // namespace aurora::gx
