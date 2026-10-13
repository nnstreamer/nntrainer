// SPDX-License-Identifier: Apache-2.0
/** @file qkv_norm_lifetime_device_check.cpp
 * @brief Public API lifetime checks for deferred prefill RMSNorm/QKV fusion.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include <algorithm>
#include <compute_ops.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <htp_backend.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned K = 2048;
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
    bytes(size_t((rows + 1u) / 2u) * cols), scales(cols, 0.0f) {
    uint32_t seed = rows * 7u + cols * 13u;
    for (unsigned col = 0; col < cols; ++col)
      scales[col] = (1u + col % 13u) * (1.0f / 4096.0f);
    for (unsigned row = 0; row < rows; ++row)
      for (unsigned col = 0; col < cols; ++col) {
        seed = seed * 1664525u + 1013904223u;
        const int8_t v = static_cast<int8_t>((seed >> 28u) - 8);
        // QS4CX projection takes q+8 nibbles in N-major rows.
        bytes[size_t(col) * ((rows + 1u) / 2u) + row / 2u] |=
          static_cast<uint8_t>((v + 8) << (4u * (row & 1u)));
      }
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

void require(bool ok, const char *what) {
  if (!ok)
    throw std::runtime_error(what);
}

template <typename Call> void must_throw(Call &&call, const char *what) {
  bool threw = false;
  try {
    call();
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, what);
}

struct Model {
  std::vector<float> gamma = std::vector<float>(K);
  nntrainer::ComputeOps *ops = nntrainer::get_htp_ops();
  Model() {
    for (unsigned k = 0; k < K; ++k)
      gamma[k] = 0.5f + (k % 23u) * (1.0f / 32.0f);
  }
  void normalize(unsigned rows, const float *x, const float *res, float *sum,
                 float *norm) {
    const bool ok =
      res ? ops->add_rmsnorm_fp32(rows, K, eps, x, res, gamma.data(), sum, norm)
          : ops->inference_math_fp32(
              nntrainer::ComputeOps::InferenceMath::RmsNorm, rows, K, K, eps,
              1.0f, x, nullptr, gamma.data(), norm);
    require(ok, "reference norm rejected");
  }
};

struct Fixture {
  unsigned rows;
  size_t count;
  Shared x, res, sum, norm, expected_sum, expected_norm;
  std::vector<float> raw, residual, untouched;
  explicit Fixture(unsigned m) :
    rows(m),
    count(size_t(m) * K),
    x(count),
    res(count),
    sum(count),
    norm(count),
    expected_sum(count),
    expected_norm(count),
    raw(count),
    residual(count),
    untouched(count) {
    activation(raw, residual, true);
    std::memcpy(untouched.data(), norm.p, count * sizeof(float));
    reset();
  }
  void reset() {
    std::copy(raw.begin(), raw.end(), x.p);
    std::copy(residual.begin(), residual.end(), res.p);
    std::copy(untouched.begin(), untouched.end(), sum.p);
    std::copy(untouched.begin(), untouched.end(), norm.p);
  }
  void unwritten() const {
    equal("deferred x changed", x.p, raw.data(), count);
    equal("deferred residual changed", res.p, residual.data(), count);
    equal("sum written before consume", sum.p, untouched.data(), count);
    equal("norm written before consume", norm.p, untouched.data(), count);
  }
  void guards() const {
    for (const Shared *s :
         {&x, &res, &sum, &norm, &expected_sum, &expected_norm})
      s->guards();
  }
};

struct QkvModel {
  Weight q{K, K}, k{K, 512}, v{K, 512};
  std::vector<float> gamma = std::vector<float>(128);
  std::vector<float> extra_gamma = std::vector<float>(K);
  explicit QkvModel() {
    for (unsigned i = 0; i < gamma.size(); ++i)
      gamma[i] = .25f + (i % 13u) / 16.f;
    for (unsigned i = 0; i < K; ++i)
      extra_gamma[i] = .75f + (i % 7u) / 32.f;
  }
  void project(Model &model, unsigned rows, float *input, float *qout,
               float *kout, float *vout, bool extra_norm = false,
               bool malformed = false) {
    model.ops->gemm_q4_0_batch_norm_fp32(
      {q.bytes.data(), k.bytes.data(), v.bytes.data()},
      {q.scales.data(), k.scales.data(), v.scales.data()}, input,
      {qout, kout, vout}, rows,
      malformed ? std::vector<unsigned>{K, 512}
                : std::vector<unsigned>{K, 512, 512},
      K, extra_norm ? extra_gamma.data() : nullptr, {64u, 64u, 0u},
      gamma.data(), 1.e-6f, nullptr, 0u, 0u, false);
  }
};

void qkv_lifetime(Model &model, QkvModel &qkv) {
  Fixture f(65);
  auto *ops = model.ops;
  auto defer = [&](unsigned consumer) {
    bool accepted;
    if (consumer == 0u)
      accepted = ops->defer_prefill_qkv_norm(
        f.rows, K, eps, f.x.p, f.res.p, model.gamma.data(), f.sum.p, f.norm.p);
    else if (consumer == 1u)
      accepted = ops->defer_prefill_conv_norm(
        f.rows, K, eps, f.x.p, f.res.p, model.gamma.data(), f.sum.p, f.norm.p);
    else if (consumer == 2u)
      accepted = ops->defer_prefill_router_norm(
        f.rows, K, eps, f.x.p, f.res.p, model.gamma.data(), f.sum.p, f.norm.p);
    else
      accepted = ops->defer_decode_norm("conv", K, eps, f.x.p, f.res.p,
                                        model.gamma.data(), f.sum.p, f.norm.p);
    require(accepted, "norm descriptor rejected");
  };
  defer(0);
  f.unwritten();
  for (unsigned c = 0; c < 4; ++c)
    must_throw([&] { defer(c); }, "pending QKV collision accepted");
  ops->discard_deferred_norm();
  ops->discard_deferred_norm();
  for (unsigned c = 1; c < 4; ++c) {
    defer(c);
    must_throw([&] { defer(0); }, "QKV accepted with other pending consumer");
    ops->discard_deferred_norm();
  }
  f.unwritten();
  for (unsigned bad = 0; bad < 3; ++bad) {
    defer(0);
    must_throw(
      [&] {
        ops->materialize_prefill_qkv_norm(bad == 0 ? f.sum.p : f.norm.p,
                                          bad == 1 ? f.rows + 1u : f.rows,
                                          bad == 2 ? K + 128u : K);
      },
      "QKV mismatched consumer accepted");
    f.unwritten();
    defer(0);
    ops->discard_deferred_norm();
  }
  // A validation exception clears the borrowed descriptor before registration.
  Shared q(f.count), k(size_t(f.rows) * 512), v(size_t(f.rows) * 512);
  defer(0);
  bool threw = false;
  try {
    qkv.project(model, f.rows, f.norm.p, q.p, k.p, v.p, false, true);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "malformed QKV vector was accepted");
  f.unwritten();
  defer(0);
  ops->discard_deferred_norm();
  // Generic layer paths explicitly materialize before reading the input.
  model.normalize(f.rows, f.x.p, f.res.p, f.expected_sum.p, f.expected_norm.p);
  defer(0);
  f.unwritten();
  const auto gamma = model.gamma;
  std::fill(model.gamma.begin(), model.gamma.end(), -7.f);
  ops->materialize_prefill_qkv_norm(f.norm.p, f.rows, K);
  model.gamma = gamma;
  equal("generic fallback sum", f.sum.p, f.expected_sum.p, f.count);
  equal("generic fallback norm", f.norm.p, f.expected_norm.p, f.count);
  ops->materialize_prefill_qkv_norm(f.norm.p, f.rows, K);
  equal("generic materialize twice", f.norm.p, f.expected_norm.p, f.count);
  f.reset();
  defer(0);
  ops->discard_deferred_norm();
  f.unwritten();
  f.guards();
  std::printf("QKV_NORM_LIFETIME_PASSED collisions=7 mismatches=3 "
              "discard/no_early_writes/invalid_vector/generic_materialize\n");
}

void qkv_parity(Model &model, QkvModel &qkv) {
  unsigned cases = 0;
  for (unsigned rows : {65u, 1024u}) {
    Fixture f(rows);
    Shared eq(f.count), ek(size_t(rows) * 512), ev(size_t(rows) * 512),
      q(f.count), k(size_t(rows) * 512), v(size_t(rows) * 512);
    for (bool add : {false, true}) {
      f.reset();
      model.normalize(rows, f.x.p, add ? f.res.p : nullptr,
                      add ? f.expected_sum.p : nullptr, f.expected_norm.p);
      qkv.project(model, rows, f.expected_norm.p, eq.p, ek.p, ev.p);
      const unsigned aliases = rows == 65u ? (add ? 8u : 2u) : 1u;
      for (unsigned alias = 0; alias < aliases; ++alias) {
        f.reset();
        float *sums[] = {f.sum.p, f.x.p, f.res.p, f.sum.p,
                         f.sum.p, f.x.p, f.res.p, f.sum.p};
        float *norms[] = {f.norm.p, f.norm.p, f.norm.p, f.x.p,
                          f.res.p,  f.res.p,  f.x.p,    f.norm.p};
        float *sp = add ? sums[alias] : nullptr;
        float *np = add ? norms[alias] : alias ? f.x.p : f.norm.p;
        float *qp = alias == 7u ? f.x.p : q.p;
        float *kp = alias == 7u ? f.res.p : k.p;
        require(model.ops->defer_prefill_qkv_norm(rows, K, eps, f.x.p,
                                                  add ? f.res.p : nullptr,
                                                  model.gamma.data(), sp, np),
                "QKV gamma deferral rejected");
        f.unwritten();
        const auto gamma = model.gamma;
        std::fill(model.gamma.begin(), model.gamma.end(), -7.f);
        qkv.project(model, rows, np, qp, kp, v.p);
        model.gamma = gamma;
        if (add)
          equal("QKV published sum", sp, f.expected_sum.p, f.count);
        equal("QKV published norm", np, f.expected_norm.p, f.count);
        equal("QKV Q", qp, eq.p, eq.size);
        equal("QKV K", kp, ek.p, ek.size);
        equal("QKV V", v.p, ev.p, ev.size);
        f.guards();
        q.guards();
        k.guards();
        v.guards();
        ++cases;
      }
    }
    if (rows == 65u) {
      // Both a nonshared result and an explicit second pre-norm force the
      // ordinary projection path. The preceding norm must be published once.
      for (unsigned fallback = 0; fallback < 3u; ++fallback) {
        f.reset();
        const bool extra = fallback == 1u;
        model.normalize(rows, f.x.p, f.res.p, f.expected_sum.p,
                        f.expected_norm.p);
        qkv.project(model, rows, f.expected_norm.p, eq.p, ek.p, ev.p, extra);
        std::vector<float> host_q(f.count);
        float *qp = fallback == 0u   ? host_q.data()
                    : fallback == 2u ? f.norm.p
                                     : q.p;
        require(model.ops->defer_prefill_qkv_norm(rows, K, eps, f.x.p, f.res.p,
                                                  model.gamma.data(), f.sum.p,
                                                  f.norm.p),
                "QKV fallback deferral rejected");
        f.unwritten();
        qkv.project(model, rows, f.norm.p, qp, k.p, v.p, extra);
        equal("QKV fallback sum", f.sum.p, f.expected_sum.p, f.count);
        if (fallback != 2u)
          equal("QKV fallback norm", f.norm.p, f.expected_norm.p, f.count);
        equal("QKV fallback Q", qp, eq.p, eq.size);
        equal("QKV fallback K", k.p, ek.p, ek.size);
        equal("QKV fallback V", v.p, ev.p, ev.size);
        ++cases;
      }
    }
  }
  model.ops->discard_deferred_norm();
  std::printf("QKV_NORM_GAMMA_ALIAS_FALLBACK_PARITY_PASSED cases=%u "
              "sum/norm/Q/K/V bitwise\n",
              cases);
}

void qkv_bypass(Model &model, bool stats) {
  Fixture f(65);
  const bool accepted = model.ops->defer_prefill_qkv_norm(
    f.rows, K, eps, f.x.p, f.res.p, model.gamma.data(), f.sum.p, f.norm.p);
  require(accepted == !stats, "unexpected QKV instrumentation bypass");
  f.unwritten();
  model.ops->discard_deferred_norm();
  f.unwritten();
  f.guards();
  std::printf("QKV_NORM_%s_POLICY_PASSED accepted=%u no_writes\n",
              stats ? "STATS" : "PROFILE3", unsigned(accepted));
}
} // namespace
int main(int argc, char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "default";
    setenv("NNTR_HTP_PREFILL_CONV_NORM", "1", 1);
    setenv("NNTR_HTP_PREFILL_ROUTER_NORM", "1", 1);
    setenv("NNTR_HTP_PREFILL_QKV_NORM", "1", 1);
    setenv("NNTR_HTP_ROUTER_Q4_QUANT_REUSE", "1", 1);
    setenv("NNTR_HTP_MOE_INPUT_REUSE", "1", 1);
    setenv("NNTR_HTP_NORM_CONSUMER_FUSION", "1", 1);
    setenv("NNTR_ACT_STATS", mode == "stats" ? "1" : "0", 1);
    setenv("NNTR_HTP_PROFILE", mode == "profile" ? "3" : "0", 1);
    nntrainer::init_backend();
    Model model;
    if (mode == "stats" || mode == "profile")
      qkv_bypass(model, mode == "stats");
    else {
      require(mode == "default", "unknown checker mode");
      QkvModel qkv;
      qkv_lifetime(model, qkv);
      qkv_parity(model, qkv);
      std::printf("ALL_QKV_NORM_LIFETIME_CHECKS_PASSED\n");
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  return 0;
}
