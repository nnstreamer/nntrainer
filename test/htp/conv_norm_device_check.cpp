// SPDX-License-Identifier: Apache-2.0
/** @file conv_norm_device_check.cpp
 * @brief Public API parity for prefill RMSNorm and Conv input packing fusion.
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
#include <htp_wh_layout.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned K = 2048, C = 2048, N = 2048;
constexpr float eps = 1e-5f;
constexpr uint32_t guard = 0x7fc12345u;

struct Shared {
  float *base, *p;
  size_t size;
  explicit Shared(size_t count) : size(count) {
    base = static_cast<float *>(
      nntrainer::HtpBackend::global().alloc_shared((count + 64) * 4u));
    if (!base)
      throw std::runtime_error("shared allocation failed");
    p = base + 32;
    for (size_t i = 0; i < count + 64; ++i)
      std::memcpy(base + i, &guard, sizeof(guard));
  }
  ~Shared() { nntrainer::HtpBackend::global().free_shared(base); }
  Shared(const Shared &) = delete;
  Shared &operator=(const Shared &) = delete;
  void guards() const {
    for (size_t i = 0; i < 32; ++i) {
      uint32_t a, b;
      std::memcpy(&a, base + i, sizeof(a));
      std::memcpy(&b, p + size + i, sizeof(b));
      if (a != guard || b != guard)
        throw std::runtime_error("shared guard overwritten");
    }
  }
};

void equal(const char *what, const float *got, const float *want, size_t n) {
  if (!std::memcmp(got, want, n * sizeof(float)))
    return;
  for (size_t i = 0; i < n; ++i)
    if (std::memcmp(got + i, want + i, sizeof(float))) {
      std::fprintf(stderr, "%s differs at %zu: %.9g versus %.9g\n", what, i,
                   got[i], want[i]);
      break;
    }
  throw std::runtime_error(what);
}

struct Weight {
  std::vector<uint8_t> bytes;
  std::vector<float> scales;
  Weight(unsigned rows, unsigned cols) :
    bytes(nntrainer::whBytes(rows, cols)), scales(2u * cols, 0.0f) {
    std::vector<int8_t> matrix(size_t(rows) * cols);
    uint32_t seed = rows * 7u + cols * 13u;
    for (unsigned col = 0; col < cols; ++col)
      scales[col] = (1u + col % 13u) * (1.0f / 4096.0f);
    for (unsigned row = 0; row < rows; ++row)
      for (unsigned col = 0; col < cols; ++col) {
        seed = seed * 1664525u + 1013904223u;
        const int8_t v = static_cast<int8_t>((seed >> 28u) - 8);
        matrix[size_t(row) * cols + col] = v;
        scales[cols + col] += v;
      }
    nntrainer::whPack(matrix.data(), rows, cols, bytes.data());
  }
};

struct Model {
  Weight in{K, 3u * C}, out{C, N};
  std::vector<float> conv = std::vector<float>(3u * C);
  std::vector<float> gamma = std::vector<float>(K);
  nntrainer::ComputeOps *ops = nntrainer::get_htp_ops();
  Model() {
    for (unsigned i = 0; i < 3u * C; ++i)
      conv[i] = (int(i % 29u) - 14) * (1.0f / 64.0f);
    for (unsigned i = 0; i < K; ++i)
      gamma[i] = 0.5f + (i % 23u) * (1.0f / 32.0f);
  }
  void normalize(unsigned M, const float *x, const float *res, float *sum,
                 float *norm) {
    bool ok;
    if (res) {
      ok = ops->add_rmsnorm_fp32(M, K, eps, x, res, gamma.data(), sum, norm);
    } else {
      ok = ops->inference_math_fp32(
        nntrainer::ComputeOps::InferenceMath::RmsNorm, M, K, K, eps, 1.0f, x,
        nullptr, gamma.data(), norm);
    }
    if (!ok)
      throw std::runtime_error("reference norm rejected");
  }
  void block(unsigned M, const float *input, float *result, float *state) {
    ops->gemm_qs4cx_conv_block_fp32(
      in.bytes.data(), conv.data(), out.bytes.data(), input, result, state, M,
      K, C, N, in.scales.data(), out.scales.data(), true);
  }
  void run(bool fused, unsigned M, const float *x, const float *res, float *sum,
           float *norm, float *result, float *state) {
    if (fused) {
      const bool accepted = ops->defer_prefill_conv_norm(
        M, K, eps, x, res, gamma.data(), sum, norm);
      if (accepted != (M > 1))
        throw std::runtime_error("unexpected prefill deferral dispatch");
      if (!accepted)
        normalize(M, x, res, sum, norm);
    } else {
      normalize(M, x, res, sum, norm);
    }
    block(M, norm, result, state);
  }
};

void activation(std::vector<float> &x, std::vector<float> &res,
                bool rounding_cases = false) {
  uint32_t seed = 47u;
  for (size_t i = 0; i < x.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    x[i] = (int((seed >> 9u) & 32767u) - 16384) * (1.0f / 2048.0f);
    const unsigned row = static_cast<unsigned>(i / K);
    if (row % 13u == 0u)
      x[i] = 0.0f;
    else if (row % 13u == 1u)
      x[i] = -0.375f;
    else if (i % 97u == 0u)
      x[i] *= 16.0f;
    res[i] = i % 11u == 0u ? -x[i] : (int(i % 31u) - 15) / 8.0f;
    if (rounding_cases && i % 97u == 0u) {
      // Halfway additions must round to f32 before qf32 sum-of-squares.
      const uint32_t xb[] = {0x3f800001u, 0x3f7fffffu, 0x4b000001u,
                             0x4affffffu, 0xbf800001u, 0xbf7fffffu,
                             0xcb000001u, 0xcaffffffu};
      const uint32_t rb[] = {0x33800000u, 0x33000000u, 0x3f000000u,
                             0x3e800000u, 0xb3800000u, 0xb3000000u,
                             0xbf000000u, 0xbe800000u};
      const unsigned k = static_cast<unsigned>((i / 97u) % 8u);
      std::memcpy(&x[i], &xb[k], sizeof(float));
      std::memcpy(&res[i], &rb[k], sizeof(float));
    }
  }
}

void check_lifetime(Model &model) {
  const unsigned M = 2;
  const size_t count = size_t(M) * K;
  Shared x(count), res(count), sum(count), norm(count), result(count);
  Shared expected_sum(count), expected_norm(count), expected_result(count);
  std::vector<float> raw(count), residual(count), state(2u * C, 0.0f);
  std::vector<float> expected_state(state), untouched(count);
  activation(raw, residual, true);
  std::copy(raw.begin(), raw.end(), x.p);
  std::copy(residual.begin(), residual.end(), res.p);
  std::memcpy(untouched.data(), sum.p, count * sizeof(float));
  model.run(false, M, x.p, res.p, expected_sum.p, expected_norm.p,
            expected_result.p, expected_state.data());
  auto *ops = model.ops;
  auto prefill = [&] {
    if (!ops->defer_prefill_conv_norm(M, K, eps, x.p, res.p, model.gamma.data(),
                                      sum.p, norm.p))
      throw std::logic_error("lifetime prefill deferral rejected");
  };
  auto decode = [&] {
    if (!ops->defer_decode_norm("conv", K, eps, x.p, res.p, model.gamma.data(),
                                sum.p, norm.p))
      throw std::logic_error("lifetime decode deferral rejected");
  };
  auto must_throw = [](auto &&call) {
    bool threw = false;
    try {
      call();
    } catch (const std::runtime_error &) {
      threw = true;
    }
    if (!threw)
      throw std::runtime_error("pending norm collision was accepted");
  };
  auto unwritten = [&] {
    equal("deferred input changed", x.p, raw.data(), count);
    equal("deferred residual changed", res.p, residual.data(), count);
    equal("sum written before consumption", sum.p, untouched.data(), count);
    equal("norm written before consumption", norm.p, untouched.data(), count);
    equal("Conv written before consumption", result.p, untouched.data(), count);
  };

  prefill();
  unwritten();
  must_throw(prefill);
  must_throw(decode);
  ops->discard_deferred_norm();
  ops->discard_deferred_norm();
  unwritten();

  decode();
  must_throw(prefill);
  ops->discard_deferred_norm();
  unwritten();

  prefill();
  must_throw([&] { model.block(M, sum.p, result.p, state.data()); });
  unwritten();
  prefill(); // A consumer mismatch must leave no pending borrowed pointers.
  ops->discard_deferred_norm();
  unwritten();

  // Model cache eviction may invalidate the caller's gamma after deferral.
  // Poison those values to verify that the consumer owns its gamma snapshot.
  const std::vector<float> original_gamma(model.gamma);
  prefill();
  std::fill(model.gamma.begin(), model.gamma.end(), -7.0f);
  model.block(M, norm.p, result.p, state.data());
  model.gamma = original_gamma;
  equal("deferred gamma sum", sum.p, expected_sum.p, count);
  equal("deferred gamma norm", norm.p, expected_norm.p, count);
  equal("deferred gamma Conv output", result.p, expected_result.p, count);
  equal("deferred gamma Conv state", state.data(), expected_state.data(),
        state.size());
  for (const Shared *s : {&x, &res, &sum, &norm, &result})
    s->guards();
  ops->discard_deferred_norm();
  std::printf("CONV_NORM_LIFETIME_CHECKS_PASSED collisions/discard/mismatch/"
              "gamma_snapshot/no_early_writes\n");
}

void check_stats_bypass(Model &model) {
  const size_t count = 2u * K;
  Shared x(count), res(count), sum(count), norm(count);
  std::vector<float> untouched(count);
  std::memcpy(untouched.data(), norm.p, count * sizeof(float));
  if (model.ops->defer_prefill_conv_norm(2u, K, eps, x.p, res.p,
                                         model.gamma.data(), sum.p, norm.p))
    throw std::runtime_error("ACT_STATS accepted a deferred output");
  equal("stats sum changed", sum.p, untouched.data(), count);
  equal("stats norm changed", norm.p, untouched.data(), count);
  model.ops->discard_deferred_norm();
  std::printf("CONV_NORM_STATS_BYPASS_PASSED\n");
}

void check(Model &model) {
  unsigned cases = 0;
  for (unsigned M : {1u, 2u, 15u, 16u, 17u, 63u, 64u, 65u, 129u, 1024u}) {
    const size_t count = size_t(M) * K;
    Shared x(count), res(count), sum(count), norm(count), result(count);
    Shared esum(count), enorm(count), expected(count);
    std::vector<float> raw(count), residual(count), state(2u * C);
    std::vector<float> expected_state(2u * C), initial_state(2u * C);
    activation(raw, residual, true);
    for (unsigned i = 0; i < 2u * C; ++i)
      initial_state[i] = (int(i % 19u) - 9) * (1.0f / 256.0f);
    for (bool add : {false, true}) {
      std::copy(raw.begin(), raw.end(), x.p);
      std::copy(residual.begin(), residual.end(), res.p);
      expected_state = initial_state;
      model.run(false, M, x.p, add ? res.p : nullptr, add ? esum.p : nullptr,
                enorm.p, expected.p, expected_state.data());
      const unsigned aliases = M == 17u || M == 65u ? 11u : 1u;
      for (unsigned alias = 0; alias < aliases; ++alias) {
        std::copy(raw.begin(), raw.end(), x.p);
        std::copy(residual.begin(), residual.end(), res.p);
        state = initial_state;
        float *sums[] = {sum.p, x.p,   res.p, sum.p, sum.p, x.p,
                         res.p, sum.p, sum.p, sum.p, sum.p};
        float *norms[] = {norm.p, norm.p, norm.p, x.p,    res.p, res.p,
                          x.p,    norm.p, norm.p, norm.p, norm.p};
        float *outputs[] = {result.p, result.p, result.p, result.p,
                            result.p, result.p, result.p, x.p,
                            res.p,    norm.p,   sum.p};
        model.run(true, M, x.p, add ? res.p : nullptr,
                  add ? sums[alias] : nullptr, norms[alias], outputs[alias],
                  state.data());
        if (add && alias != 10u)
          equal("published sum", sums[alias], esum.p, count);
        if (alias != 9u)
          equal("published norm", norms[alias], enorm.p, count);
        equal("Conv output", outputs[alias], expected.p, count);
        equal("Conv state", state.data(), expected_state.data(), state.size());
        x.guards();
        res.guards();
        sum.guards();
        norm.guards();
        result.guards();
        ++cases;
      }
    }
    std::printf("CONV_NORM_CHECK M=%u K=%u C=%u residual/norm/output/state "
                "bitwise=PASS\n",
                M, K, C);
  }
  // Rejected prefill deferrals must leave no pending consumer.
  {
    const unsigned M = 2;
    const size_t count = size_t(M) * K;
    Shared x(count + 1), res(count), sum(count), norm(count);
    std::vector<float> plain(count);
    auto *ops = model.ops;
    if (ops->defer_prefill_conv_norm(M, K, eps, plain.data(), res.p,
                                     model.gamma.data(), sum.p, norm.p) ||
        ops->defer_prefill_conv_norm(M, K, eps, x.p, res.p, model.gamma.data(),
                                     x.p + 1, norm.p) ||
        ops->defer_prefill_conv_norm(M, K, eps, x.p, res.p, model.gamma.data(),
                                     sum.p, sum.p) ||
        ops->defer_prefill_conv_norm(M, K, eps, x.p, res.p, nullptr, sum.p,
                                     norm.p))
      throw std::runtime_error("unsafe deferral accepted");
    x.guards();
    res.guards();
    sum.guards();
    norm.guards();
  }
  check_lifetime(model);
  std::printf(
    "ALL_CONV_NORM_CHECKS_PASSED cases=%u plus rejection and lifetime "
    "checks\n",
    cases);
}

void benchmark(Model &model, unsigned repeats) {
  const unsigned M = 1024;
  const size_t count = size_t(M) * K;
  Shared x(count), res(count), sum(count), norm(count), result(count);
  std::vector<float> raw(count), residual(count), state(2u * C);
  activation(raw, residual);
  std::copy(raw.begin(), raw.end(), x.p);
  std::copy(residual.begin(), residual.end(), res.p);
  std::vector<double> times[2];
  for (unsigned r = 0; r < repeats + 3u; ++r) {
    for (unsigned order = 0; order < 2u; ++order) {
      const unsigned fused = (r + order) & 1u;
      std::fill(state.begin(), state.end(), 0.0f);
      const auto start = std::chrono::steady_clock::now();
      model.run(fused, M, x.p, res.p, sum.p, norm.p, result.p, state.data());
      const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      if (r >= 3u)
        times[fused].push_back(us);
    }
  }
  for (unsigned fused = 0; fused < 2u; ++fused) {
    auto &t = times[fused];
    std::sort(t.begin(), t.end());
    std::printf("CONV_NORM_TIMING M=%u K=%u mode=%s repeats=%u "
                "median_us=%.3f min_us=%.3f max_us=%.3f\n",
                M, K, fused ? "fused" : "separate", repeats,
                (t[(t.size() - 1) / 2] + t[t.size() / 2]) * 0.5, t.front(),
                t.back());
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    setenv("NNTR_HTP_PREFILL_CONV_NORM", "1", 1);
    const bool stats = argc > 1 && std::string(argv[1]) == "stats";
    if (stats)
      setenv("NNTR_ACT_STATS", "1", 1);
    nntrainer::init_backend();
    Model model;
    if (stats)
      check_stats_bypass(model);
    else if (argc > 1 && std::string(argv[1]) == "bench")
      benchmark(model, argc > 2 ? std::max(1, std::atoi(argv[2])) : 30u);
    else
      check(model);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  return 0;
}
