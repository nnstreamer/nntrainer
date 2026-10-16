#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "expert_lru.h"

using Ids = std::vector<int>;
using Key = causallm::ExpertLru::Key;
using Keys = std::vector<Key>;
static std::array<unsigned char, 128> storage{};

static void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

// These blocks are extracted verbatim from the two production assignment
// methods. Arithmetic/selection is checked by the author's whole-method test;
// this extension stresses the exact materialization block against real LRU.
static Ids materialize(const std::vector<Ids> &rows, bool candidate) {
  Ids result;
  auto *extra_top_k = &result;
  uint64_t seen = 0;
  bool compact_recency = true;
  for (const auto &row : rows) {
    std::vector<std::pair<float, int>> scored;
    for (int id : row)
      scored.emplace_back(0.0f, id);
    const unsigned extended = static_cast<unsigned>(row.size());
    if (candidate) {
#include "candidate_hint.inc"
    } else {
#include "reference_hint.inc"
    }
  }
  return result;
}

static Keys keys(const Ids &ids, const std::array<unsigned, 64> &map) {
  Keys result;
  for (auto it = ids.rbegin(); it != ids.rend(); ++it)
    result.push_back(&storage.at(map.at(static_cast<unsigned>(*it))));
  return result;
}

static Keys selected(unsigned n, unsigned offset) {
  Keys result;
  for (unsigned i = 0; i < n; ++i)
    result.push_back(&storage[(i + offset) % storage.size()]);
  return result;
}

static void equal(causallm::ExpertLru &a, causallm::ExpertLru &b) {
  check(a.capacity() == b.capacity(), "capacity changed");
  check(a.size() == b.size(), "residency count changed");
  check(a.order() == b.order(), "final key order changed");
  for (const auto &value : storage)
    check(a.resident(&value) == b.resident(&value), "resident key changed");
  for (unsigned n : {0u, 1u, 4u, 16u, 64u})
    check(a.room(selected(n, 43)) == b.room(selected(n, 43)),
          "held room changed");
}

static void trial(const Ids &full, const Ids &compact,
                  const std::array<unsigned, 64> &map, unsigned capacity,
                  double half, unsigned initial, unsigned rotate) {
  causallm::ExpertLru a, b;
  int layer;
  a.setLrfu(half);
  b.setLrfu(half);
  a.addLayer(&layer, capacity);
  b.addLayer(&layer, capacity);
  Keys loads_a, loads_b, evicts_a, evicts_b;
  auto load_a = [&](Key key) { loads_a.push_back(key); };
  auto load_b = [&](Key key) { loads_b.push_back(key); };
  auto evict_a = [&](Key key) { evicts_a.push_back(key); };
  auto evict_b = [&](Key key) { evicts_b.push_back(key); };
  const Keys seed = selected(std::min(initial, capacity), rotate);
  check(a.acquire(seed, load_a, evict_a) == b.acquire(seed, load_b, evict_b),
        "seed changed");
  // Unequal use counts and elapsed acquires exercise nontrivial LRFU state.
  if (!seed.empty()) {
    for (unsigned i = 0; i < 11; ++i) {
      Keys need{seed[(i * i + 3) % seed.size()]};
      a.acquire(need, load_a, evict_a);
      b.acquire(need, load_b, evict_b);
    }
  }
  const unsigned held = std::min(2u, capacity);
  a.hold(held);
  b.hold(held);
  a.refresh(keys(full, map));
  b.refresh(keys(compact, map));
  equal(a, b);
  a.unhold(held);
  b.unhold(held);
  for (unsigned step = 0; step < 10; ++step) {
    Keys pins;
    auto order = a.order();
    if (!order.empty() && step % 2)
      pins.push_back(order[step % order.size()]);
    const unsigned n =
      capacity ? std::min<unsigned>(1 + step % 3, capacity) : 0;
    check(a.makeRoom(n, pins, evict_a) == b.makeRoom(n, pins, evict_b),
          "makeRoom result changed");
    const Keys need = selected(n, 79 + step * 3);
    check(a.acquire(need, load_a, evict_a) == b.acquire(need, load_b, evict_b),
          "future misses changed");
    check(loads_a == loads_b, "future load order changed");
    check(evicts_a == evicts_b, "future eviction order changed");
    equal(a, b);
  }
}

int main() {
  try {
    std::mt19937 random(0x361733);
    unsigned trials = 0, streams = 0;
    for (unsigned mode = 0; mode < 12; ++mode) {
      const unsigned rows = mode < 2   ? mode
                            : mode < 6 ? 17
                            : mode < 9 ? 1024
                                       : 2048;
      std::vector<Ids> input;
      for (unsigned row = 0; row < rows; ++row) {
        Ids values;
        if (mode == 2) {
          values = {63, 0, 62, 1, 61, 2, 60, 3, 59};
        } else if (mode == 3) {
          values = {0, 1, 2, 3, 4, 5, 6, 7, int(row ? 8 : 63)};
        } else if (mode == 4) {
          // Full mask is reached only in the final rows.
          const unsigned from = row < 10 ? 0 : (row - 10) * 9;
          for (unsigned k = 0; k < 9; ++k)
            values.push_back(int((from + k) % 64));
        } else if (mode == 5) {
          // The exact prefix may never cover all experts.
          for (unsigned k = 0; k < 9; ++k)
            values.push_back(int((row + k) % 13));
        } else {
          Ids universe;
          for (int id = 0; id < 64; ++id)
            universe.push_back(id);
          std::shuffle(universe.begin(), universe.end(), random);
          values.assign(universe.begin(), universe.begin() + 9);
        }
        input.push_back(values);
      }
      const Ids full = materialize(input, false);
      const Ids compact = materialize(input, true);
      check(compact.size() <= 64, "unbounded compact output");
      ++streams;
      for (unsigned alias = 0; alias < 4; ++alias) {
        std::array<unsigned, 64> mapping;
        for (unsigned i = 0; i < 64; ++i)
          mapping[i] = alias == 0   ? i
                       : alias == 1 ? i % 4
                       : alias == 2 ? (i * 3 + 7) % 17
                                    : random() % 89;
        for (unsigned capacity : {0u, 1u, 4u, 16u, 64u})
          for (double half : {0.0, 1.0, 32.0, 1000.0})
            for (unsigned initial : {0u, 3u, 128u}) {
              trial(full, compact, mapping, capacity, half, initial,
                    (streams * 7 + alias * 5) % 128);
              ++trials;
            }
      }
    }
    std::printf("RECENCY_LRU %u streams %u state-transition cases PASS\n",
                streams, trials);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "RECENCY_LRU FAIL: %s\n", error.what());
    return 1;
  }
}
