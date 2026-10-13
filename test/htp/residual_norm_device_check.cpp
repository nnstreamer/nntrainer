// SPDX-License-Identifier: Apache-2.0
/** @file residual_norm_device_check.cpp
 * @brief Compare fused residual RMSNorm with two independent DSP operations.
 */
#include <algorithm>
#include <chrono>
#include <compute_ops.h>
#include <cstdio>
#include <cstring>
#include <htp_backend.h>
#include <stdexcept>
#include <vector>

struct Shared {
  float *p;
  explicit Shared(size_t n) :
    p(static_cast<float *>(
      nntrainer::HtpBackend::global().alloc_shared(n * sizeof(float)))) {
    if (!p)
      throw std::runtime_error("shared allocation failed");
    std::fill(p, p + n, 0.f);
  }
  ~Shared() { nntrainer::HtpBackend::global().free_shared(p); }
};

static void equal(const char *what, const float *a, const float *b, size_t n) {
  if (std::memcmp(a, b, n * sizeof(float))) {
    for (size_t i = 0; i < n; ++i)
      if (std::memcmp(a + i, b + i, sizeof(float))) {
        std::fprintf(stderr, "%s differs at %zu: %.9g vs %.9g\n", what, i, a[i],
                     b[i]);
        break;
      }
    throw std::runtime_error(what);
  }
}

int main() {
  try {
    nntrainer::init_backend();
    auto *ops = nntrainer::get_htp_ops();
    using Math = nntrainer::ComputeOps::InferenceMath;
    unsigned cases = 0;
    for (unsigned N : {64u, 2048u, 2816u, 4096u, 8192u})
      for (unsigned M : {1u, 2u, 7u, 65u, 1024u, 2048u}) {
        // Cover the bounded scratch limit and its allocation-free fallback
        // without allocating full-context tensors for unrelated large widths.
        if (N >= 4096u && M > 65u)
          continue;
        const size_t count = size_t(M) * N;
        Shared x(count), residual(count), sum(count), norm(count);
        Shared esum(count), enorm(count), gamma(N);
        for (unsigned i = 0; i < N; ++i)
          gamma.p[i] = 0.5f + (i % 31) * 0.046875f;
        for (bool with_gamma : {false, true})
          for (unsigned alias = 0; alias < 7; ++alias) {
            unsigned seed = 23 + M + N;
            auto random = [&]() {
              seed = seed * 1103515245u + 12345u;
              return (float((seed >> 8) & 65535) / 65535.f - 0.5f) * 4.f;
            };
            for (size_t i = 0; i < count; ++i) {
              x.p[i] = random() * (1.f + i % 7);
              residual.p[i] = i % 11 ? random() : -x.p[i];
            }
            const float *g = with_gamma ? gamma.p : nullptr;
            if (!ops->inference_math_fp32(Math::Add, M, N, N, 1e-5f, 1.f, x.p,
                                          residual.p, nullptr, esum.p) ||
                !ops->inference_math_fp32(Math::RmsNorm, M, N, N, 1e-5f, 1.f,
                                          esum.p, nullptr, g, enorm.p))
              throw std::runtime_error("reference rejected");
            float *sum_outputs[] = {sum.p, x.p, residual.p, sum.p,
                                    sum.p, x.p, residual.p};
            float *norm_outputs[] = {norm.p,     norm.p,     norm.p, x.p,
                                     residual.p, residual.p, x.p};
            float *s = sum_outputs[alias];
            float *y = norm_outputs[alias];
            if (!ops->add_rmsnorm_fp32(M, N, 1e-5f, x.p, residual.p, g, s, y))
              throw std::runtime_error("fused operation rejected");
            equal("sum", s, esum.p, count);
            equal("norm", y, enorm.p, count);
            ++cases;
          }
        std::printf(
          "PASS M=%u N=%u: gamma/no gamma and seven aliases bit-exact\n", M, N);
      }
    std::printf("ALL_RESIDUAL_NORM_CHECKS_PASSED cases=%u\n", cases);
    {
      constexpr unsigned M = 1024, N = 2048, repeats = 40;
      constexpr size_t count = size_t(M) * N;
      Shared x(count), residual(count), sum(count), norm(count), gamma(N);
      std::fill(x.p, x.p + count, 0.25f);
      std::fill(residual.p, residual.p + count, -0.125f);
      std::fill(gamma.p, gamma.p + N, 1.f);
      std::vector<double> times;
      for (unsigned i = 0; i < repeats + 5; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (!ops->add_rmsnorm_fp32(M, N, 1e-5f, x.p, residual.p, gamma.p, sum.p,
                                   norm.p))
          throw std::runtime_error("benchmark operation rejected");
        const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        if (i >= 5)
          times.push_back(us);
      }
      std::sort(times.begin(), times.end());
      std::printf("RESIDUAL_NORM_TIMING M=%u N=%u repeats=%u median_us=%.3f\n",
                  M, N, repeats,
                  (times[repeats / 2 - 1] + times[repeats / 2]) / 2);
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  return 0;
}
