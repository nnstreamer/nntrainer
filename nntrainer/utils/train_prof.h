// SPDX-License-Identifier: Apache-2.0
/**
 * @file   train_prof.h
 * @brief  [PERF-TRAIN] env-gated (NNTR_TRAIN_PROF) per-layer training timer.
 *         Diagnostic only. Layer execution is sequential, so no locking.
 */
#ifndef __NNTR_TRAIN_PROF_H__
#define __NNTR_TRAIN_PROF_H__

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace nntrainer {
namespace train_prof {

enum Phase { FWD = 0, DERIV = 1, GRAD = 2, OTHER = 3 };

struct Acc {
  double ms[4] = {0, 0, 0, 0};
};

inline bool enabled() {
  static const bool e = std::getenv("NNTR_TRAIN_PROF") != nullptr;
  return e;
}

/** first layer index counted as a LoRA (backwarded) block */
inline int loraStart() {
  static const int s = [] {
    const char *e = std::getenv("NNTR_TRAIN_PROF");
    return e ? std::atoi(e) : 0;
  }();
  return s;
}

inline std::map<std::string, Acc> &table() {
  static std::map<std::string, Acc> t;
  return t;
}

/** data-fetch wait of the current training iteration (ms) */
inline double &lastFetchMs() {
  static double v = 0;
  return v;
}

inline unsigned &iters() {
  static unsigned n = 0;
  return n;
}

inline void add(const std::string &key, int phase, double ms) {
  table()[key].ms[phase] += ms;
}

/** "layer23_ffn_up" -> "L:ffn_up" (top/LoRA blocks) or "F:ffn_up" (frozen) */
inline std::string layerKey(const std::string &type, const std::string &name) {
  std::string stripped;
  int id = -1;
  size_t i = 0;
  if (name.rfind("layer", 0) == 0) {
    i = 5;
    id = 0;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9')
      id = id * 10 + (name[i++] - '0');
    if (i < name.size() && name[i] == '_')
      ++i;
  }
  stripped = name.substr(i);
  std::string grp = id < 0 ? "-" : (id >= loraStart() ? "L" : "F");
  return grp + ":" + stripped + "(" + type + ")";
}

using clk = std::chrono::steady_clock;

struct Scope {
  const char *key;
  int phase;
  clk::time_point t0;
  bool on;
  Scope(const char *k, int p) : key(k), phase(p), on(enabled()) {
    if (on)
      t0 = clk::now();
  }
  void stop() {
    if (on)
      add(key, phase,
          std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    on = false;
  }
  ~Scope() { stop(); }
};

inline void reset() {
  table().clear();
  iters() = 0;
}

inline void dump(double fwd_ms_avg, double bwd_ms_avg) {
  const double n = iters() ? (double)iters() : 1.0;
  std::printf("[PERF-TRAIN] ==== avg over %u iters (fwd %.1f / bwd %.1f ms)"
              " ====\n",
              iters(), fwd_ms_avg, bwd_ms_avg);
  double tot[4] = {0, 0, 0, 0};
  for (auto &kv : table()) {
    const Acc &a = kv.second;
    std::printf("[PERF-TRAIN] %-48s fwd %8.2f  deriv %8.2f  grad %8.2f  "
                "other %8.2f\n",
                kv.first.c_str(), a.ms[0] / n, a.ms[1] / n, a.ms[2] / n,
                a.ms[3] / n);
    if (kv.first.rfind("fc.", 0) != 0 && kv.first.rfind("opt.", 0) != 0)
      for (int p = 0; p < 4; ++p)
        tot[p] += a.ms[p];
  }
  std::printf("[PERF-TRAIN] %-48s fwd %8.2f  deriv %8.2f  grad %8.2f\n",
              "TOTAL(layer nodes)", tot[0] / n, tot[1] / n, tot[2] / n);
  std::fflush(stdout);
}

} // namespace train_prof
} // namespace nntrainer

#endif // __NNTR_TRAIN_PROF_H__
