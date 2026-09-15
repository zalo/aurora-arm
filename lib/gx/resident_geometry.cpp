#include "resident_geometry.hpp"

#include "../gfx/hash.hpp"
#include "aurora/dl.hpp"

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <utility>

namespace aurora::gx::resident {
namespace {
struct Arena {
  u32 stride = 0;
  size_t used = 0; // bytes decoded so far
  std::vector<gfx::ArenaUpload> pending;
};

struct Cache {
  std::vector<Arena> arenas;
  absl::flat_hash_map<u64, Entry> entries;
  // Every pointer an entry depends on (list, array bases) -> entry key. Releasing a memory region
  // is a range query here; scanning all entries x 26 array pointers per release was 43% of the FIFO
  // worker in a scene whose effect archives churn constantly.
  std::multimap<uintptr_t, u64> pointerIndex;
  bool resetRequested = false;
  Stats stats;
};

Cache& cache() noexcept {
  static Cache instance;
  return instance;
}

size_t budget() noexcept {
  return g_config.residentGeometryBudget != 0 ? g_config.residentGeometryBudget : DefaultBudget;
}

// Calls `fn` once per distinct non-null pointer the entry depends on.
template <typename F>
void for_each_dependency(const Entry& entry, F&& fn) {
  std::array<const void*, MaxVtxAttr + 1> seen{};
  size_t count = 0;
  const auto visit = [&](const void* p) {
    if (p == nullptr || std::find(seen.begin(), seen.begin() + count, p) != seen.begin() + count) {
      return;
    }
    seen[count++] = p;
    fn(reinterpret_cast<uintptr_t>(p));
  };
  visit(entry.list);
  for (const void* p : entry.arrays) {
    visit(p);
  }
}

u32 arena_for_stride(u32 stride) {
  auto& arenas = cache().arenas;
  for (u32 i = 0; i < arenas.size(); ++i) {
    if (arenas[i].stride == stride) {
      return i;
    }
  }
  arenas.push_back(Arena{.stride = stride});
  return static_cast<u32>(arenas.size() - 1);
}

void reset() noexcept {
  auto& c = cache();
  c.entries.clear();
  c.pointerIndex.clear();
  for (auto& arena : c.arenas) {
    arena.used = 0; // pending uploads keep their own offsets and still reach their frame ops
  }
  c.resetRequested = false;
  ++c.stats.resets;
}
} // namespace

u64 list_hash(const u8* list, u32 size) noexcept {
  const u32 head = std::min<u32>(size, 64);
  const u32 tail = std::min<u32>(size - head, 64);
  Hasher hasher{size};
  hasher.update(list, head);
  hasher.update(list + size - tail, tail);
  return hasher.digest();
}

u64 entry_key(const void* list, u32 size, const ShaderConfig& config, const std::array<AttrArray, MaxVtxAttr>& arrays,
              u32 currentPnMtx) noexcept {
  Hasher hasher;
  hasher.update(reinterpret_cast<uintptr_t>(list));
  hasher.update(size);
  if (decoded_vertex_has_location(config, 0)) {
    hasher.update(currentPnMtx); // baked into the matrix word of records without a PN matrix index
  }
  hasher.update(config.vtxStride);
  static_assert(std::has_unique_object_representations_v<AttrConfig>);
  hasher.update(config.attrs);
  for (u32 i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const u8 type = config.attrs[i].attrType;
    if (type == GX_INDEX8 || type == GX_INDEX16) {
      hasher.update(reinterpret_cast<uintptr_t>(arrays[i].data));
      hasher.update(arrays[i].size);
    }
  }
  return hasher.digest();
}

Entry* find(u64 key) noexcept {
  auto& entries = cache().entries;
  const auto it = entries.find(key);
  return it != entries.end() ? &it->second : nullptr;
}

Entry& insert(u64 key, Entry entry) {
  auto& c = cache();
  erase(key);
  auto& stored = c.entries.emplace(key, std::move(entry)).first->second;
  for_each_dependency(stored, [&](uintptr_t p) { c.pointerIndex.emplace(p, key); });
  return stored;
}

void erase(u64 key) noexcept {
  auto& c = cache();
  const auto it = c.entries.find(key);
  if (it == c.entries.end()) {
    return;
  }
  for_each_dependency(it->second, [&](uintptr_t p) {
    const auto [begin, end] = c.pointerIndex.equal_range(p);
    for (auto index = begin; index != end; ++index) {
      if (index->second == key) {
        c.pointerIndex.erase(index);
        break;
      }
    }
  });
  c.entries.erase(it);
}

size_t invalidate(const void* base, size_t size) noexcept {
  if (size == 0) {
    return 0;
  }
  auto& c = cache();
  const auto lo = reinterpret_cast<uintptr_t>(base);
  const uintptr_t hi = lo + std::min<uintptr_t>(size, UINTPTR_MAX - lo);
  std::vector<u64> keys;
  for (auto it = c.pointerIndex.lower_bound(lo); it != c.pointerIndex.end() && it->first < hi; ++it) {
    keys.push_back(it->second);
  }
  size_t removed = 0;
  for (const u64 key : keys) {
    if (c.entries.contains(key)) {
      erase(key);
      ++removed;
    }
  }
  c.stats.invalidated += removed;
  return removed;
}

bool decode_display_list(const u8* list, u32 size, GXVtxFmt fmt, const VertexLoader& loader,
                         const std::array<AttrArray, MaxVtxAttr>& arrays, u32 currentPnMtx, Entry& out) {
  const u32 stride = loader.layout.stride;
  if (stride == 0 || loader.vtxStride == 0 || loader.lineMode != 0) {
    return false;
  }
  // Validate the whole list and count its vertices and indices before touching the arena.
  size_t vertexCount = 0;
  size_t indexCount = 0;
  {
    dl::Reader reader{list, size, loader.vtxStride};
    while (const auto cmd = reader.next()) {
      if (cmd->kind == dl::Command::Kind::Passthrough) {
        if (cmd->data[0] == GX_NOP) {
          continue;
        }
        return false;
      }
      const auto& draw = cmd->draw;
      if (draw.fmt != fmt) {
        return false;
      }
      if (cmd->kind == dl::Command::Kind::Draw) {
        if (!dl::expand_triangles(draw.prim, draw.vtxCount, [&](u16, u16, u16) { indexCount += 3; })) {
          return false;
        }
      } else {
        if (draw.prim != GX_TRIANGLES) {
          return false;
        }
        for (u32 i = 0; i < draw.indexCount; ++i) {
          if (draw.index(i) >= draw.vtxCount) {
            return false;
          }
        }
        indexCount += draw.indexCount;
      }
      vertexCount += draw.vtxCount;
    }
    if (reader.failed() || vertexCount == 0 || vertexCount > UINT32_MAX / stride) {
      return false;
    }
  }

  auto& c = cache();
  const size_t bytes = vertexCount * stride;
  if (used_bytes() + bytes > budget()) {
    // Over budget: process lists inline until the frame boundary empties the cache. A list that
    // alone exceeds the budget never fits and must not empty the cache every frame.
    if (used_bytes() != 0) {
      c.resetRequested = true;
    }
    return false;
  }
  const u32 arenaIndex = arena_for_stride(stride);
  auto& arena = c.arenas[arenaIndex];
  const size_t offset = arena.used;
  arena.used += bytes;
  if (arena.pending.empty() || arena.pending.back().offset + arena.pending.back().data.size() != offset) {
    arena.pending.push_back(gfx::ArenaUpload{.arena = arenaIndex, .offset = static_cast<uint32_t>(offset)});
  }
  auto& data = arena.pending.back().data;
  const size_t dataOffset = data.size();
  data.resize(dataOffset + bytes);
  u8* records = data.data() + dataOffset;

  out.arena = arenaIndex;
  out.firstVertex = static_cast<u32>(offset / stride);
  out.vertexCount = static_cast<u32>(vertexCount);
  out.indices.clear();
  out.indices.reserve(indexCount);
  out.arrays = {};
  for (const auto& step : loader.steps) {
    if (step.indexBytes != 0) {
      out.arrays[step.attr] = arrays[step.attr].data;
    }
  }
  u32 base = out.firstVertex;
  dl::Reader reader{list, size, loader.vtxStride};
  while (const auto cmd = reader.next()) {
    if (cmd->kind == dl::Command::Kind::Passthrough) {
      continue;
    }
    const auto& draw = cmd->draw;
    decode_vertices(loader, draw.vertices, draw.vtxCount, records, arrays, currentPnMtx);
    records += static_cast<size_t>(draw.vtxCount) * stride;
    if (cmd->kind == dl::Command::Kind::Draw) {
      dl::expand_triangles(draw.prim, draw.vtxCount, [&](u16 i0, u16 i1, u16 i2) {
        out.indices.insert(out.indices.end(), {base + i0, base + i1, base + i2});
      });
    } else {
      for (u32 i = 0; i < draw.indexCount; ++i) {
        out.indices.push_back(base + draw.index(i));
      }
    }
    base += draw.vtxCount;
  }
  return true;
}

u32 arena_stride(u32 arena) noexcept { return cache().arenas[arena].stride; }

size_t used_bytes() noexcept {
  size_t total = 0;
  for (const auto& arena : cache().arenas) {
    total += arena.used;
  }
  return total;
}

size_t entry_count() noexcept { return cache().entries.size(); }

Stats& stats() noexcept { return cache().stats; }

std::vector<gfx::ArenaUpload> take_uploads() {
  std::vector<gfx::ArenaUpload> uploads;
  for (auto& arena : cache().arenas) {
    for (auto& upload : arena.pending) {
      uploads.push_back(std::move(upload));
    }
    arena.pending.clear();
  }
  return uploads;
}

void end_frame() noexcept {
  if (cache().resetRequested) {
    reset();
  }
}

bool index_consistent() noexcept {
  auto& c = cache();
  size_t dependencies = 0;
  bool consistent = true;
  for (const auto& [key, entry] : c.entries) {
    for_each_dependency(entry, [&](uintptr_t p) {
      ++dependencies;
      const auto [begin, end] = c.pointerIndex.equal_range(p);
      consistent &= std::count_if(begin, end, [&](const auto& item) { return item.second == key; }) == 1;
    });
  }
  return consistent && dependencies == c.pointerIndex.size();
}

void shutdown() noexcept {
  auto& c = cache();
  c.entries.clear();
  c.pointerIndex.clear();
  c.arenas.clear();
  c.resetRequested = false;
  c.stats = {};
}

} // namespace aurora::gx::resident
