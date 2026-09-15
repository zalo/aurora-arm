#pragma once
// Sampled, nested CPU timings for the renderer's hot paths. Compiled in with AURORA_DEEP_TIMERS=1 (CMake
// option AURORA_DEEP_TIMERS); off, every Scope is an empty object. When compiled in, AURORA_DEEP_PROFILE=1
// enables it at run time and AURORA_DEEP_INTERVAL=N (default 120) selects every Nth frame to sample; each
// sampled frame prints one [deep-profile] line per (zone, tag) with call count, wall time, self time and an
// optional unit count, per lane (fifo, render). No GL query, fence or GPU wait is introduced.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <vector>

namespace aurora::gfx::profile {
inline bool enabled() {
#if AURORA_DEEP_TIMERS
  static const bool value = [] {
    const char* s = std::getenv("AURORA_DEEP_PROFILE");
    return s != nullptr && std::strcmp(s, "1") == 0;
  }();
  return value;
#else
  return false;
#endif
}
inline unsigned interval() {
  static const unsigned value = [] {
    const char* s = std::getenv("AURORA_DEEP_INTERVAL");
    return s != nullptr ? static_cast<unsigned>(std::max(1, std::atoi(s))) : 120u;
  }();
  return value;
}
inline uint64_t cpu_now() {
  timespec t{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return static_cast<uint64_t>(t.tv_sec) * 1000000000 + static_cast<uint64_t>(t.tv_nsec);
}
inline uint64_t wall_now() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline bool cpu_enabled() {
  static const bool value = std::getenv("AURORA_DEEP_CPU") != nullptr;
  return value;
}
struct Totals {
  uint64_t calls = 0, wall = 0, cpu = 0, selfWall = 0, selfCpu = 0, units = 0;
};
struct Key {
  const char* name;
  uint64_t tag;
  bool operator==(const Key&) const = default;
};
struct KeyHash {
  size_t operator()(const Key& key) const {
    return (reinterpret_cast<uintptr_t>(key.name) >> 3) ^ (key.tag * 0x9e3779b97f4a7c15ULL);
  }
};
struct Scope;
struct State {
  bool active = false;
  uint64_t frame = 0, tag = 0;
  const char* lane = "unknown";
  Scope* parent = nullptr;
  std::unordered_map<Key, Totals, KeyHash> totals;
  std::vector<std::string> patterns;
};
inline thread_local State state;
// Frame the FIFO processor is recording, published by begin_recording for the fifo lane.
inline std::atomic<uint64_t> fifoFrame{0};
inline bool active() { return enabled() && state.active; }
inline void begin(uint64_t frame, const char* lane) {
  if (!enabled()) {
    return;
  }
  state.active = frame % interval() == 0;
  state.frame = frame;
  state.lane = lane;
  state.tag = 0;
  state.parent = nullptr;
  state.totals.clear();
  state.patterns.clear();
  if (state.active) {
    state.totals.reserve(4096);
    state.patterns.reserve(512);
  }
}
struct Scope {
  bool on;
  const char* name;
  uint64_t tag, units, wall, cpu, childWall, childCpu;
  Scope* parent;
  __attribute__((always_inline)) Scope(const char* label, uint64_t key = UINT64_MAX, uint64_t count = 0) : on(active()) {
    if (!on) {
      return;
    }
    name = label;
    tag = key == UINT64_MAX ? state.tag : key;
    units = count;
    childWall = childCpu = 0;
    cpu = 0;
    parent = state.parent;
    state.parent = this;
    if (cpu_enabled()) {
      cpu = cpu_now();
    }
    wall = wall_now();
  }
  __attribute__((always_inline)) ~Scope() {
    if (on) {
      finish();
    }
  }
  __attribute__((noinline)) void finish() {
    const uint64_t elapsedWall = wall_now() - wall;
    const uint64_t elapsedCpu = cpu_enabled() ? cpu_now() - cpu : 0;
    state.parent = parent;
    if (parent != nullptr) {
      parent->childWall += elapsedWall;
      parent->childCpu += elapsedCpu;
    }
    auto& sum = state.totals[{name, tag}];
    ++sum.calls;
    sum.wall += elapsedWall;
    sum.cpu += elapsedCpu;
    sum.selfWall += elapsedWall > childWall ? elapsedWall - childWall : 0;
    sum.selfCpu += elapsedCpu > childCpu ? elapsedCpu - childCpu : 0;
    sum.units += units;
  }
};
inline void end() {
  if (!enabled() || !state.active) {
    return;
  }
  state.active = false;
  for (const auto& [key, s] : state.totals) {
    std::fprintf(stderr,
                 "[deep-profile] frame=%llu lane=%s zone=%s tag=%llu calls=%llu wall_us=%.3f cpu_us=%.3f "
                 "self_wall_us=%.3f self_cpu_us=%.3f units=%llu cpu_measured=%u\n",
                 static_cast<unsigned long long>(state.frame), state.lane, key.name,
                 static_cast<unsigned long long>(key.tag), static_cast<unsigned long long>(s.calls), s.wall / 1000.,
                 s.cpu / 1000., s.selfWall / 1000., s.selfCpu / 1000., static_cast<unsigned long long>(s.units),
                 cpu_enabled() ? 1u : 0u);
  }
  for (const auto& pattern : state.patterns) {
    std::fprintf(stderr, "[deep-pattern] frame=%llu lane=%s %s\n", static_cast<unsigned long long>(state.frame),
                 state.lane, pattern.c_str());
  }
}
} // namespace aurora::gfx::profile
