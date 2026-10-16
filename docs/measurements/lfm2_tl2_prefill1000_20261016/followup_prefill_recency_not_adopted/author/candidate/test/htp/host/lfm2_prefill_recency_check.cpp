// SPDX-License-Identifier: Apache-2.0
// Actual assignment/guard/caller extracts and the unchanged real ExpertLru.
#include "expert_lru.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

// Only extracted application guards use these macros on the host. No ARM
// arithmetic is emulated; assignment code uses the compiler's actual libm/STL.
#ifndef RECENCY_CHECK_GENERIC
#ifndef __aarch64__
#define __aarch64__ 1
#endif
#define ENABLE_HEXKL 1
#endif

static void check(bool ok, const char *what) {
  if (!ok)
    throw std::runtime_error(what);
}
static unsigned numeric_cases, guard_cases, lru_cases, observer_cases;
using Assignments = std::vector<std::vector<std::pair<unsigned, float>>>;

namespace nntrainer {
enum class Tdatatype { FP32, FP16, TL2CX_WH, QS4CX_WH };
enum class Tformat { NCHW, NHWC };
struct ComputeOps {};
static ComputeOps htp, other;
static ComputeOps *get_htp_ops() { return &htp; }
struct Tensor {
  std::array<unsigned, 4> dim{1, 1, 1024, 2048};
  Tdatatype type = Tdatatype::FP32;
  Tformat format = Tformat::NCHW;
  bool contiguous = true;
  ComputeOps *ops = &htp;
  float *data = nullptr;
  unsigned batch() const { return dim[0]; }
  unsigned channel() const { return dim[1]; }
  unsigned height() const { return dim[2]; }
  unsigned width() const { return dim[3]; }
  Tdatatype getDataType() const { return type; }
  Tformat getFormat() const { return format; }
  bool getContiguous() const { return contiguous; }
  ComputeOps *getOps() const { return ops; }
  template <class T> T *getData() const { return reinterpret_cast<T *>(data); }
  void reshape(std::array<unsigned, 4> shape) {
    const auto count = [](const std::array<unsigned, 4> &d) {
      return size_t(d[0]) * d[1] * d[2] * d[3];
    };
    if (count(dim) != count(shape))
      throw std::invalid_argument("reshape changed element count");
    dim = shape;
  }
};
struct RunLayerContext {
  unsigned inputs = 1;
  std::vector<Tensor> weights = std::vector<Tensor>(128);
  unsigned getNumInputs() const { return inputs; }
  Tensor &getWeight(unsigned i) { return weights.at(i); }
};
namespace props {
struct Unit {
  unsigned value = 1536;
  unsigned get() const { return value; }
};
} // namespace props
} // namespace nntrainer

#include "constants.inc"
#include "trace.inc"
namespace baseline {
struct Lfm2MoELayer {
  unsigned num_experts = 64, topk = 4;
  bool softmax_router = false;
#include "baseline-decl.inc"
};
#include "baseline.inc"
} // namespace baseline
namespace candidate {
struct Lfm2MoELayer {
  unsigned num_experts = 64, topk = 4;
  bool softmax_router = false;
#include "candidate-decl.inc"
};
#include "candidate.inc"
#if defined(__aarch64__) && defined(ENABLE_HEXKL)
#include "helper.inc"
#endif
struct Gate {
  unsigned from = 0, num_experts = 64, topk = 4;
  bool experts_virtual = true, in_norm = true, router_norm = false;
  bool out_norm = false, out_add = false, softmax_router = false;
  bool gelu_act = false;
  std::tuple<nntrainer::props::Unit> moe_props;
  std::vector<unsigned> expert_gate_up_proj_indices, expert_down_proj_indices;
  Gate() {
    for (unsigned i = 0; i < 64; ++i) {
      expert_gate_up_proj_indices.push_back(2 * i);
      expert_down_proj_indices.push_back(2 * i + 1);
    }
  }
  bool run(nntrainer::Tensor input_, nntrainer::RunLayerContext &context) {
    auto input = input_;
    const unsigned total_tokens = input.batch() * input.height();
    const unsigned hidden_size = input.width();
    // This exact source statement precedes the actual caller gate. Keeping
    // the original [1,1,M,K] here would conceal the real dispatch bug.
#include "prepared-caller.inc"
    return compact_recency;
  }
};
} // namespace candidate

static bool enabled() {
#if defined(__aarch64__) && defined(ENABLE_HEXKL)
  const char *s = std::getenv("NNTR_LFM2_COMPACT_PREFILL_RECENCY");
  const char *t = std::getenv("NNTR_MOE_TRACE");
  return s && std::strcmp(s, "1") == 0 && (!t || !*t) && moeTrace() == nullptr;
#else
  return false;
#endif
}
static nntrainer::RunLayerContext native_context() {
  nntrainer::RunLayerContext c;
  for (unsigned e = 0; e < 64; ++e) {
    c.weights[2 * e].type = c.weights[2 * e + 1].type =
      nntrainer::Tdatatype::TL2CX_WH;
    c.weights[2 * e].dim = {1, 1, 2048, 3072};
    c.weights[2 * e + 1].dim = {1, 1, 1536, 2048};
  }
  return c;
}

static void guards() {
  auto c = native_context();
  nntrainer::Tensor rows;
  candidate::Gate g;
  for (unsigned m : {1024u, 2048u}) {
    rows.dim[2] = m;
    check(g.run(rows, c) == enabled(),
          "actual flattened caller never qualified");
    ++guard_cases;
  }
  rows.dim[2] = 1024;
  auto reject = [&](const candidate::Gate &gate, nntrainer::Tensor x,
                    nntrainer::RunLayerContext ctx) {
    auto copy = gate;
    try {
      check(!copy.run(x, ctx), "unsupported caller/metadata accepted");
    } catch (const std::invalid_argument &) {
      // Actual Tensor::reshape rejects a changed element count before the
      // candidate caller guard, e.g. unsupported input channel != 1.
    }
    ++guard_cases;
  };
  for (unsigned m : {0u, 1u, 512u, 1023u, 1025u, 2049u}) {
    auto x = rows;
    x.dim[2] = m;
    reject(g, x, c);
  }
  for (unsigned field : {0u, 1u, 3u}) {
    auto x = rows;
    x.dim[field] += 1;
    reject(g, x, c);
  }
  for (unsigned mode = 0; mode < 5; ++mode) {
    auto x = rows;
    if (mode == 0)
      x.type = nntrainer::Tdatatype::FP16;
    if (mode == 1)
      x.format = nntrainer::Tformat::NHWC;
    if (mode == 2)
      x.contiguous = false;
    if (mode == 3)
      x.ops = &nntrainer::other;
    if (mode == 4)
      x.ops = nullptr;
    reject(g, x, c);
  }
  for (unsigned which = 0; which < 128; ++which) {
    auto ctx = c;
    ctx.weights[which].type = nntrainer::Tdatatype::QS4CX_WH;
    reject(g, rows, ctx);
  }
  for (unsigned e : {0u, 127u}) {
    for (unsigned dim : {2u, 3u}) {
      auto ctx = c;
      ++ctx.weights[e].dim[dim];
      reject(g, rows, ctx);
    }
  }
  for (unsigned mode = 0; mode < 14; ++mode) {
    auto z = g;
    auto ctx = c;
    switch (mode) {
    case 0:
      z.from = 1;
      break;
    case 1:
      z.experts_virtual = false;
      break;
    case 2:
      z.in_norm = false;
      break;
    case 3:
      z.router_norm = true;
      break;
    case 4:
      z.out_norm = true;
      break;
    case 5:
      z.out_add = true;
      break;
    case 6:
      z.softmax_router = true;
      break;
    case 7:
      z.gelu_act = true;
      break;
    case 8:
      z.num_experts = 128;
      break;
    case 9:
      z.topk = 8;
      break;
    case 10:
      std::get<0>(z.moe_props).value = 704;
      break;
    case 11:
      z.expert_gate_up_proj_indices.pop_back();
      break;
    case 12:
      z.expert_down_proj_indices.pop_back();
      break;
    case 13:
      ctx.inputs = 2;
      break;
    }
    reject(z, rows, ctx);
  }
}

static uint32_t bits(float f) {
  uint32_t v;
  std::memcpy(&v, &f, sizeof(v));
  return v;
}
static float from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
static void equal_assignments(const Assignments &a, const Assignments &b) {
  check(a.size() == b.size(), "expert count changed");
  for (size_t e = 0; e < a.size(); ++e) {
    check(a[e].size() == b[e].size(), "assignment count changed");
    for (size_t i = 0; i < a[e].size(); ++i) {
      check(a[e][i].first == b[e][i].first, "token append order changed");
      check(bits(a[e][i].second) == bits(b[e][i].second),
            "routing FP32 bits changed");
    }
  }
}
static std::vector<int> first_ids(const std::vector<int> &s) {
  std::vector<int> out;
  for (int e : s)
    if (std::find(out.begin(), out.end(), e) == out.end())
      out.push_back(e);
  return out;
}

static void lru_compare(const std::vector<int> &full,
                        const std::vector<int> &small, unsigned seed) {
  std::array<unsigned, 96> keys{};
  std::mt19937 rng(seed);
  for (double half : {0.0, 8.0, 32.0}) {
    for (unsigned alias : {1u, 7u, 64u}) {
      causallm::ExpertLru a, b;
      a.addLayer(&a, 37);
      b.addLayer(&b, 37);
      a.setLrfu(half);
      b.setLrfu(half);
      auto step = [&](const std::vector<const void *> &need) {
        std::vector<const void *> ea, eb, la, lb;
        a.acquire(
          need, [&](const void *k) { la.push_back(k); },
          [&](const void *k) { ea.push_back(k); });
        b.acquire(
          need, [&](const void *k) { lb.push_back(k); },
          [&](const void *k) { eb.push_back(k); });
        check(ea == eb && la == lb && a.order() == b.order(),
              "actual future LRU/LRFU eviction/load order changed");
      };
      for (unsigned i = 0; i < 20; ++i) {
        std::vector<const void *> need;
        for (unsigned k = 0; k < 4; ++k) {
          const void *key = &keys[(rng() % 24) * 4 + k];
          need.push_back(key);
        }
        step(need);
      }
      auto map = [&](const std::vector<int> &s) {
        std::vector<const void *> out;
        for (auto it = s.rbegin(); it != s.rend(); ++it)
          out.push_back(&keys[static_cast<unsigned>(*it) % alias]);
        return out;
      };
      const auto ra = map(full), rb = map(small);
      const auto size = a.size();
      a.refresh(ra);
      b.refresh(rb);
      check(a.order() == b.order() && a.size() == size && b.size() == size &&
              a.capacity() == b.capacity(),
            "actual refresh order/size changed");
      for (unsigned i = 0; i < 30; ++i) {
        std::vector<const void *> need;
        for (unsigned k = 0; k < 4; ++k)
          need.push_back(&keys[(rng() % 24) * 4 + k]);
        step(need);
      }
      ++lru_cases;
    }
  }
}

static void numerical(unsigned m, unsigned e, unsigned mode, bool compact,
                      bool softmax = false, bool append = false,
                      bool no_extra = false) {
  std::vector<float> logits(size_t(m) * e), bias(e);
  std::mt19937 rng(19 + m + e + mode);
  for (unsigned j = 0; j < e; ++j)
    bias[j] = softmax ? 1.0f + j / 64.0f : int(rng() % 129 - 64) / 128.0f;
  for (unsigned i = 0; i < m; ++i) {
    for (unsigned j = 0; j < e; ++j) {
      float value = int(rng() % 2049 - 1024) / 64.0f;
      if (mode == 1)
        value = 0.0f; // fixed prefix, never sees all 64
      if (mode == 2)
        value = static_cast<float>((j + e - (9 * i) % e) % e) / 8.0f - 4.0f;
      if (mode == 3)
        value =
          i + 8 < m ? 0.0f : static_cast<float>((j + 9 * i) % e) / 8.0f - 4.0f;
      if (mode == 4) {
        static const uint32_t special[] = {
          0,           0x80000000u, 1,           0x80000001u,
          0x00800000u, 0x80800000u, 0x7f800000u, 0xff800000u,
          0x7fc12345u, 0xffc45678u, 0x7f7fffffu, 0xff7fffffu};
        value = from_bits(special[(i * 3 + j) % 12]);
      }
      logits[size_t(i) * e + j] = value;
    }
  }
  if (mode >= 1 && mode <= 3)
    std::fill(bias.begin(), bias.end(), 0.0f);
  if (mode == 5 && e > 63)
    bias[63] = 100.0f; // bit 63 must be a uint64_t shift
  if (mode == 6 && e > 2)
    bias[2] = from_bits(0x7fc43210u);
  nntrainer::Tensor x, b;
  x.data = logits.data();
  b.data = bias.data();
  baseline::Lfm2MoELayer ref;
  candidate::Lfm2MoELayer opt;
  ref.num_experts = opt.num_experts = e;
  ref.topk = opt.topk = std::min(4u, e);
  ref.softmax_router = opt.softmax_router = softmax;
  Assignments ar(e), ao(e);
  std::vector<int> full, reduced;
  if (append)
    full = reduced = {11, 63, 11};
  ref.buildExpertAssignments(x, b, m, ar, no_extra ? nullptr : &full);
  opt.buildExpertAssignments(x, b, m, ao, no_extra ? nullptr : &reduced,
                             compact);
  equal_assignments(ar, ao);
  const bool narrowed = compact && e == 64 && !append && !no_extra;
  check(reduced == (narrowed ? first_ids(full) : full),
        "hint first-occurrence/append semantics changed");
  if (narrowed) {
    check(reduced.size() <= 64 && reduced.capacity() == 64,
          "compact metadata bound changed");
    if (mode == 2 && m >= 64)
      check(reduced.size() == 64, "full seen-mask case did not cover all bits");
    if (m >= 1024 && mode <= 3)
      lru_compare(full, reduced, m + mode);
  }
  ++numeric_cases;
}

static void concurrent_observers() {
  std::array<unsigned, 64> keys{};
  std::vector<const void *> ascending, descending;
  for (auto &key : keys)
    ascending.push_back(&key);
  descending = ascending;
  std::reverse(descending.begin(), descending.end());
  for (bool compact : {false, true}) {
    causallm::ExpertLru lru;
    lru.addLayer(&lru, 64);
    lru.acquire(
      ascending, [](const void *) {}, [](const void *) {});
    std::vector<const void *> repeated;
    if (!compact)
      for (unsigned i = 0; i < 256; ++i)
        repeated.insert(repeated.end(), descending.begin(), descending.end());
    else
      repeated = descending;
    std::atomic<bool> begin{false}, stop{false}, invalid{false};
    std::atomic<unsigned> snapshots{0};
    std::thread observer([&] {
      while (!stop.load(std::memory_order_acquire)) {
        auto order = lru.order();
        if (order != ascending && order != descending)
          invalid.store(true);
        if (lru.size() != 64 || !lru.resident(&keys[0]))
          invalid.store(true);
        ++snapshots;
        begin.store(true, std::memory_order_release);
      }
    });
    while (!begin.load(std::memory_order_acquire))
      std::this_thread::yield();
    for (unsigned i = 0; i < 100; ++i) {
      lru.refresh(repeated);
      lru.refresh(ascending);
    }
    stop.store(true, std::memory_order_release);
    observer.join();
    check(!invalid.load() && snapshots.load() != 0,
          "observer saw intermediate refresh state");
    ++observer_cases;
  }
}

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "opened-trace") {
      check(moeTrace() != nullptr,
            "opened-trace test needs a writable configured trace");
#if !defined(_WIN32)
      check(unsetenv("NNTR_MOE_TRACE") == 0, "unsetenv failed");
#else
      check(_putenv_s("NNTR_MOE_TRACE", "") == 0, "trace env removal failed");
#endif
    }
    guards();
    for (unsigned m : {0u, 1u, 7u, 64u, 1024u, 2048u})
      for (unsigned mode = 0; mode < 7; ++mode)
        numerical(m, 64, mode, true);
    for (unsigned e : {32u, 65u, 128u})
      numerical(7, e, 0, true);
    numerical(64, 128, 4, false, true); // Gemma-style unchanged softmax
    numerical(1024, 64, 0, false);
    numerical(2048, 64, 2, true, false, true);
    numerical(1024, 64, 5, true, false, false, true);
    concurrent_observers();
    std::printf(
      "PASS numeric=%u guards=%u real_lru=%u observers=%u enabled=%d\n",
      numeric_cases, guard_cases, lru_cases, observer_cases, enabled());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  return 0;
}
