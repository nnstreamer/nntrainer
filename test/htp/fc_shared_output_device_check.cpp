// SPDX-License-Identifier: Apache-2.0
/** @file fc_shared_output_device_check.cpp
 * @brief Public ComputeOps checks for row-major shared FC slice output.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include <algorithm>
#include <chrono>
#include <compute_ops.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <htp_backend.h>
#include <htp_prepared_fc.h>
#include <htp_wh_layout.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t guard_bits = 0x7fc12345u;

float guardValue() {
  float value;
  std::memcpy(&value, &guard_bits, sizeof(value));
  return value;
}

struct Shared {
  float *base;
  float *p;
  size_t count, prefix, total;
  explicit Shared(size_t n, size_t offset = 0) :
    base(nullptr),
    p(nullptr),
    count(n),
    prefix(32 + offset),
    total(prefix + count + 32) {
    auto &backend = nntrainer::HtpBackend::global();
    base = static_cast<float *>(backend.alloc_shared(total * sizeof(float)));
    if (!base || !backend.isSharedRange(base, total * sizeof(float)))
      throw std::runtime_error("shared allocation failed");
    p = base + prefix;
    std::fill(base, base + total, guardValue());
  }
  ~Shared() { nntrainer::HtpBackend::global().free_shared(base); }
  Shared(const Shared &) = delete;
  Shared &operator=(const Shared &) = delete;
  void guards() const {
    for (size_t i = 0; i < total; ++i) {
      if (i >= prefix && i < prefix + count)
        continue;
      uint32_t bits;
      std::memcpy(&bits, base + i, sizeof(bits));
      if (bits != guard_bits)
        throw std::runtime_error("shared guard changed at " +
                                 std::to_string(i));
    }
  }
};

struct Weight {
  unsigned K, N;
  std::vector<uint8_t> wh;
  std::vector<float> scales;
  Weight(unsigned k, unsigned n) :
    K(k), N(n), wh(nntrainer::whBytes(K, N)), scales(2u * N, 0.0f) {
    std::vector<int8_t> rm(size_t(K) * N);
    uint32_t seed = K * 3u + N * 13u + 17u;
    for (unsigned c = 0; c < N; ++c)
      scales[c] = (1u + c % 31u) * (1.0f / 1024.0f);
    for (unsigned r = 0; r < K; ++r)
      for (unsigned c = 0; c < N; ++c) {
        seed = seed * 1664525u + 1013904223u;
        int8_t value = static_cast<int8_t>((seed >> 28u) - 8);
        rm[size_t(r) * N + c] = value;
        scales[N + c] += value;
      }
    nntrainer::whPack(rm.data(), K, N, wh.data());
  }
  unsigned handles() const {
    return (N + nntrainer::preparedFcSliceCols(K) - 1) /
           nntrainer::preparedFcSliceCols(K);
  }
  void run(unsigned M, float *input, float *output) {
    nntrainer::get_htp_ops()->gemm_qs4cx_wh_accel_fp32(wh.data(), scales.data(),
                                                       input, output, M, N, K);
  }
};

// Backend registration caches keys by pointer, so every weight stays alive
// until all checks and benchmarks have finished.
std::vector<std::unique_ptr<Weight>> weights;
Weight &weight(unsigned K, unsigned N) {
  for (auto &w : weights)
    if (w->K == K && w->N == N)
      return *w;
  weights.push_back(std::make_unique<Weight>(K, N));
  return *weights.back();
}

void activation(float *p, unsigned M, unsigned K) {
  uint32_t seed = M * 17u + K + 29u;
  for (unsigned r = 0; r < M; ++r)
    for (unsigned c = 0; c < K; ++c) {
      seed = seed * 1664525u + 1013904223u;
      float x = (int((seed >> 9u) & 32767u) - 16384) * (1.0f / 2048.0f);
      if (r % 7u == 0)
        x = 0.0f;
      else if (r % 7u == 1)
        x = -0.375f;
      else if (c % 127u == 0)
        x *= 8.0f;
      p[size_t(r) * K + c] = x;
    }
}

uint64_t hash(const float *p, size_t n) {
  uint64_t h = UINT64_C(14695981039346656037);
  const auto *bytes = reinterpret_cast<const uint8_t *>(p);
  for (size_t i = 0; i < n * sizeof(float); ++i)
    h = (h ^ bytes[i]) * UINT64_C(1099511628211);
  return h;
}

void equal(const char *label, const float *want, const float *got, size_t n) {
  if (std::memcmp(want, got, n * sizeof(float)) == 0)
    return;
  for (size_t i = 0; i < n; ++i)
    if (std::memcmp(want + i, got + i, sizeof(float))) {
      std::fprintf(stderr, "%s differs at %zu: %.9g versus %.9g\n", label, i,
                   want[i], got[i]);
      break;
    }
  throw std::runtime_error(label);
}

void checkCase(Weight &w, unsigned M, size_t offset) {
  const size_t in_count = size_t(M) * w.K, out_count = size_t(M) * w.N;
  std::vector<float> plain_input(in_count), want(out_count), got(out_count);
  activation(plain_input.data(), M, w.K);
  Shared input(in_count, offset), output(out_count, offset);
  std::copy(plain_input.begin(), plain_input.end(), input.p);
  w.run(M, plain_input.data(), want.data()); // general nonshared path
  w.run(M, input.p, output.p);               // public candidate dispatch
  equal("shared output", want.data(), output.p, out_count);
  equal("input preservation", plain_input.data(), input.p, in_count);
  input.guards();
  output.guards();
  w.run(M, input.p, got.data()); // shared input, nonshared output fallback
  equal("nonshared output", want.data(), got.data(), out_count);
  std::printf("FC_CHECK M=%u K=%u N=%u handles=%u offset=%zu bitwise=PASS "
              "hash=%016llx\n",
              M, w.K, w.N, w.handles(), offset,
              static_cast<unsigned long long>(hash(output.p, out_count)));
}

void checkAlias(Weight &w, unsigned M, size_t input_offset,
                size_t output_offset) {
  const size_t ni = size_t(M) * w.K, no = size_t(M) * w.N;
  const size_t count = std::max(input_offset + ni, output_offset + no);
  std::vector<float> plain(count, 0.0f);
  activation(plain.data() + input_offset, M, w.K);
  Shared shared(count);
  std::copy(plain.begin(), plain.end(), shared.p);
  // Preserve the existing staged path's behavior even for overlap across
  // FC row chunks. This is a dispatch-regression check, not a new promise
  // that arbitrary overlap is a mathematically valid in-place GEMM.
  w.run(M, plain.data() + input_offset, plain.data() + output_offset);
  w.run(M, shared.p + input_offset, shared.p + output_offset);
  equal("alias fallback", plain.data(), shared.p, count);
  shared.guards();
  std::printf("FC_ALIAS M=%u K=%u N=%u input_offset=%zu output_offset=%zu "
              "bitwise=PASS\n",
              M, w.K, w.N, input_offset, output_offset);
}

void check() {
  for (unsigned N : {32u, 2080u, 4160u, 11776u}) {
    auto &w = weight(2048u, N);
    for (unsigned M : {1u, 2u, 7u, 63u, 64u, 65u}) {
      checkCase(w, M, 0);
      if (M == 7 || M == 65)
        checkCase(w, M, 1); // unequal slices and unaligned shared subranges
    }
  }
  checkCase(weight(2048, 11776), 1024, 0);
  checkCase(weight(2048, 3072), 1024, 0);  // concatenated QKV-sized FC only
  checkCase(weight(11776, 2048), 1024, 0); // six handles, four row chunks
  auto &alias_w = weight(2048, 2080);
  for (auto offsets : {std::pair<size_t, size_t>{0, 0},
                       {0, 32},
                       {32, 0},
                       {0, size_t(7) * 2048}})
    checkAlias(alias_w, 7, offsets.first, offsets.second);
  // The first output begins at a later input chunk. Candidate must reject
  // the whole call even though the first chunk's local ranges are apart.
  checkAlias(weight(11776, 2048), 641, 0, size_t(320) * 11776);
  std::puts("ALL_FC_SHARED_OUTPUT_CHECKS_PASSED");
}

void profile(const std::string &mode) {
  constexpr unsigned M = 1024;
  auto &w = weight(2048, 11776);
  const size_t ni = size_t(M) * w.K, no = size_t(M) * w.N;
  Shared input(ni), output(no);
  activation(input.p, M, w.K);
  std::vector<float> plain(no);
  float *result = mode == "plain" ? plain.data() : output.p;
  for (unsigned i = 0; i < 3; ++i)
    w.run(M, input.p, result);
  output.guards();
  std::printf("FC_PROFILE mode=%s public_api=gemm_qs4cx_wh_accel_fp32 M=%u "
              "K=%u N=%u handles=%u calls=3 hash=%016llx\n",
              mode.c_str(), M, w.K, w.N, w.handles(),
              static_cast<unsigned long long>(hash(result, no)));
}

void benchmark(unsigned repeats, const std::string &mode) {
  for (auto shape : {std::pair<unsigned, unsigned>{2048, 11776},
                     {2048, 3072},
                     {11776, 2048}}) {
    constexpr unsigned M = 1024;
    auto &w = weight(shape.first, shape.second);
    const size_t ni = size_t(M) * w.K, no = size_t(M) * w.N;
    Shared input(ni), output(no);
    std::vector<float> plain(no);
    activation(input.p, M, w.K);
    // Weight registration, output allocation and warmup stay outside timers.
    w.run(M, input.p, plain.data());
    w.run(M, input.p, output.p);
    equal("benchmark initial", plain.data(), output.p, no);
    std::vector<double> times[2];
    for (unsigned r = 0; r < repeats + 3; ++r) {
      for (unsigned order = 0; order < 2; ++order) {
        const unsigned shared = (r + order) & 1u;
        if ((mode == "shared" && !shared) || (mode == "plain" && shared))
          continue;
        float *out = shared ? output.p : plain.data();
        const auto start = std::chrono::steady_clock::now();
        w.run(M, input.p, out);
        const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        if (r >= 3)
          times[shared].push_back(us);
      }
    }
    equal("benchmark final", plain.data(), output.p, no);
    output.guards();
    for (unsigned shared = 0; shared < 2; ++shared) {
      auto &t = times[shared];
      if (t.empty())
        continue;
      std::sort(t.begin(), t.end());
      const double median = (t[(t.size() - 1) / 2] + t[t.size() / 2]) * 0.5;
      std::printf("FC_TIMING M=%u K=%u N=%u handles=%u mode=%s repeats=%zu "
                  "median_us=%.3f min_us=%.3f max_us=%.3f hash=%016llx\n",
                  M, w.K, w.N, w.handles(), shared ? "shared" : "plain",
                  t.size(), median, t.front(), t.back(),
                  static_cast<unsigned long long>(hash(output.p, no)));
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    nntrainer::init_backend();
    if (!nntrainer::HtpBackend::global().enabled())
      throw std::runtime_error("HTP backend unavailable");
    const std::string action = argc > 1 ? argv[1] : "check";
    const std::string mode = argc > 2 ? argv[2] : "both";
    if (action == "check")
      check();
    else if (action == "profile") {
      if (mode != "shared" && mode != "plain")
        throw std::runtime_error("profile needs shared or plain");
      profile(mode);
    } else if (action == "bench") {
      if (mode != "both" && mode != "shared" && mode != "plain")
        throw std::runtime_error("bench needs both, shared or plain");
      const unsigned repeats = argc > 3 ? unsigned(std::stoul(argv[3])) : 16u;
      if (!repeats || repeats > 1000)
        throw std::runtime_error("repeats must be 1..1000");
      benchmark(repeats, mode);
    } else
      throw std::runtime_error("usage: check | profile shared|plain | "
                               "bench both|shared|plain [repeats]");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  return 0;
}
