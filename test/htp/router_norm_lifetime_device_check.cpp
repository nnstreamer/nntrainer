// SPDX-License-Identifier: Apache-2.0
/** @file router_norm_lifetime_device_check.cpp
 * @brief Public API lifetime checks for deferred prefill RMSNorm/router fusion.
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
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned K = 2048, E = 64, top_k = 4, n_sel = 8;
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
        // QS4CX router entry takes q+8 nibbles in N-major rows.
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
  Weight gate{K, E};
  std::vector<float> gamma = std::vector<float>(K);
  std::vector<float> bias = std::vector<float>(E);
  nntrainer::ComputeOps *ops = nntrainer::get_htp_ops();
  Model() {
    for (unsigned k = 0; k < K; ++k)
      gamma[k] = 0.5f + (k % 23u) * (1.0f / 32.0f);
    for (unsigned e = 0; e < E; ++e)
      bias[e] = (int(e % 7u) - 3) * (1.0f / 128.0f);
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
  bool route(unsigned rows, const float *norm, float *logits, unsigned *sel,
             float *weight, unsigned width = K, unsigned experts = E) {
    return ops->router_qs4cx_prefill(gate.bytes.data(), gate.scales.data(),
                                     norm, logits, rows, width, experts, top_k,
                                     n_sel, bias.data(), sel, weight);
  }
};

struct Fixture {
  unsigned rows;
  size_t count;
  Shared x, res, sum, norm, expected_sum, expected_norm;
  std::vector<float> raw, residual, untouched, logits, weights,
    expected_weights;
  std::vector<unsigned> selection, expected_selection;
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
    untouched(count),
    logits(size_t(m) * E),
    weights(size_t(m) * top_k),
    expected_weights(size_t(m) * top_k),
    selection(size_t(m) * n_sel),
    expected_selection(size_t(m) * n_sel) {
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
  void expected(Model &model, bool add) {
    model.normalize(rows, x.p, add ? res.p : nullptr,
                    add ? expected_sum.p : nullptr, expected_norm.p);
    require(model.route(rows, expected_norm.p, logits.data(),
                        expected_selection.data(), expected_weights.data()),
            "reference route rejected");
  }
  void parity(const float *actual_sum, const float *actual_norm, bool add) {
    if (add)
      equal("published sum", actual_sum, expected_sum.p, count);
    equal("published norm", actual_norm, expected_norm.p, count);
    require(selection == expected_selection, "router selection differs");
    equal("router weight", weights.data(), expected_weights.data(),
          weights.size());
  }
};

struct Lease {
  nntrainer::ComputeOps *ops;
  uint64_t token;
  Lease(nntrainer::ComputeOps *o, const float *input, unsigned rows) :
    ops(o), token(o->begin_moe_input_reuse(input, rows, K)) {
    require(token != 0, "input lease rejected");
  }
  ~Lease() { ops->end_moe_input_reuse(token); }
  Lease(const Lease &) = delete;
  Lease &operator=(const Lease &) = delete;
};

void check_lifetime(Model &model) {
  Fixture f(65);
  auto *ops = model.ops;
  auto router = [&] {
    require(ops->defer_prefill_router_norm(f.rows, K, eps, f.x.p, f.res.p,
                                           model.gamma.data(), f.sum.p,
                                           f.norm.p),
            "router deferral rejected");
  };
  auto conv = [&] {
    require(ops->defer_prefill_conv_norm(f.rows, K, eps, f.x.p, f.res.p,
                                         model.gamma.data(), f.sum.p, f.norm.p),
            "Conv deferral rejected");
  };
  auto decode = [&] {
    require(ops->defer_decode_norm("conv", K, eps, f.x.p, f.res.p,
                                   model.gamma.data(), f.sum.p, f.norm.p),
            "decode deferral rejected");
  };
  router();
  f.unwritten();
  must_throw(router, "router/router collision accepted");
  must_throw(conv, "router/Conv collision accepted");
  must_throw(decode, "router/decode collision accepted");
  ops->discard_deferred_norm();
  ops->discard_deferred_norm();
  f.unwritten();
  conv();
  must_throw(router, "Conv/router collision accepted");
  ops->discard_deferred_norm();
  decode();
  must_throw(router, "decode/router collision accepted");
  ops->discard_deferred_norm();
  f.unwritten();

  for (unsigned mismatch = 0; mismatch < 3; ++mismatch) {
    router();
    must_throw(
      [&] {
        model.route(mismatch == 1 ? f.rows + 1 : f.rows,
                    mismatch == 0 ? f.sum.p : f.norm.p, f.logits.data(),
                    f.selection.data(), f.weights.data(),
                    mismatch == 2 ? K + 128 : K);
      },
      "router consumer mismatch accepted");
    f.unwritten();
    router(); // A rejected consumer must drop its borrowed descriptor.
    ops->discard_deferred_norm();
  }
  f.expected(model, true);

  // A missing MoE input lease must publish both norm outputs before the
  // ordinary router reads the activation.
  router();
  require(model.route(f.rows, f.norm.p, f.logits.data(), f.selection.data(),
                      f.weights.data()),
          "no-lease ordinary router rejected");
  f.parity(f.sum.p, f.norm.p, true);
  f.reset();
  // Unsupported router shape returns false to the layer's generic FC path.
  // Its promised normalized input must already be materialized.
  router();
  require(!model.route(f.rows, f.norm.p, f.logits.data(), f.selection.data(),
                       f.weights.data(), K, E - 1),
          "unsupported router did not return false");
  equal("false fallback sum", f.sum.p, f.expected_sum.p, f.count);
  equal("false fallback norm", f.norm.p, f.expected_norm.p, f.count);
  router();
  ops->discard_deferred_norm();
  f.guards();
  std::printf("ROUTER_NORM_LIFETIME_PASSED collisions=5 mismatches=3 "
              "discard/no_early_writes/no_lease/fallback_false\n");
}

void check_owned_gamma(Model &model) {
  unsigned cases = 0;
  for (unsigned rows : {65u, 1024u}) {
    Fixture f(rows);
    for (bool add : {false, true}) {
      f.reset();
      f.expected(model, add);
      for (unsigned alias = 0; alias < (rows == 65 ? (add ? 7u : 2u) : 1u);
           ++alias) {
        f.reset();
        float *sums[] = {f.sum.p, f.x.p, f.res.p, f.sum.p,
                         f.sum.p, f.x.p, f.res.p};
        float *norms[] = {f.norm.p, f.norm.p, f.norm.p, f.x.p,
                          f.res.p,  f.res.p,  f.x.p};
        float *actual_sum = add ? sums[alias] : nullptr;
        float *actual_norm = add ? norms[alias] : alias ? f.x.p : f.norm.p;
        Lease lease(model.ops, actual_norm, rows);
        require(model.ops->defer_prefill_router_norm(
                  rows, K, eps, f.x.p, add ? f.res.p : nullptr,
                  model.gamma.data(), actual_sum, actual_norm),
                "owned-gamma deferral rejected");
        f.unwritten();
        const auto original_gamma = model.gamma;
        std::fill(model.gamma.begin(), model.gamma.end(), -7.0f);
        require(model.route(rows, actual_norm, f.logits.data(),
                            f.selection.data(), f.weights.data()),
                "fused owned-gamma router rejected");
        model.gamma = original_gamma;
        f.parity(actual_sum, actual_norm, add);
        f.guards();
        ++cases;
      }
    }
  }
  model.ops->discard_deferred_norm();
  std::printf("ROUTER_NORM_GAMMA_ALIAS_PARITY_PASSED cases=%u "
              "sum/norm/selection/weight bitwise\n",
              cases);
}

void check_fp32_fallback(Model &model) {
  Fixture f(65);
  std::vector<float> gate(size_t(K) * E), expected_logits(size_t(f.rows) * E);
  uint32_t seed = 193u;
  for (float &w : gate) {
    seed = seed * 1664525u + 1013904223u;
    w = (int((seed >> 17u) & 2047u) - 1024) * (1.0f / 65536.0f);
  }
  auto route = [&](const float *input, float *logits, unsigned *selection,
                   float *weights) {
    model.ops->router_logits_fp32(f.rows, K, E, input, nullptr, eps,
                                  gate.data(), logits, top_k, n_sel,
                                  model.bias.data(), selection, weights, true);
  };
  model.normalize(f.rows, f.x.p, f.res.p, f.expected_sum.p, f.expected_norm.p);
  route(f.expected_norm.p, expected_logits.data(), f.expected_selection.data(),
        f.expected_weights.data());
  f.reset();
  Lease lease(model.ops, f.norm.p, f.rows);
  require(model.ops->defer_prefill_router_norm(f.rows, K, eps, f.x.p, f.res.p,
                                               model.gamma.data(), f.sum.p,
                                               f.norm.p),
          "FP32 consumer deferral rejected");
  f.unwritten();
  const auto original_gamma = model.gamma;
  std::fill(model.gamma.begin(), model.gamma.end(), -7.0f);
  route(f.norm.p, f.logits.data(), f.selection.data(), f.weights.data());
  model.gamma = original_gamma;
  f.parity(f.sum.p, f.norm.p, true);
  equal("FP32 router logits", f.logits.data(), expected_logits.data(),
        expected_logits.size());
  require(model.ops->defer_prefill_router_norm(f.rows, K, eps, f.x.p, f.res.p,
                                               model.gamma.data(), f.sum.p,
                                               f.norm.p),
          "FP32 fallback retained pending descriptor");
  model.ops->discard_deferred_norm();
  f.guards();
  std::printf("ROUTER_NORM_FP32_FALLBACK_PARITY_PASSED "
              "sum/norm/logits/selection/weight bitwise gamma_snapshot\n");
}

void check_bypass(Model &model, bool stats) {
  Fixture f(65);
  require(!model.ops->defer_prefill_router_norm(f.rows, K, eps, f.x.p, f.res.p,
                                                model.gamma.data(), f.sum.p,
                                                f.norm.p),
          "instrumentation accepted deferred router output");
  const bool conv = model.ops->defer_prefill_conv_norm(
    f.rows, K, eps, f.x.p, f.res.p, model.gamma.data(), f.sum.p, f.norm.p);
  require(conv == !stats, "unexpected Conv instrumentation bypass");
  f.unwritten();
  model.ops->discard_deferred_norm();
  f.unwritten();
  f.guards();
  std::printf("ROUTER_NORM_%s_BYPASS_PASSED no_writes\n",
              stats ? "STATS" : "PROFILE3");
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "default";
    setenv("NNTR_HTP_PREFILL_CONV_NORM", "1", 1);
    setenv("NNTR_HTP_PREFILL_ROUTER_NORM", "1", 1);
    setenv("NNTR_HTP_ROUTER_Q4_QUANT_REUSE", "1", 1);
    setenv("NNTR_HTP_MOE_INPUT_REUSE", "1", 1);
    setenv("NNTR_HTP_NORM_CONSUMER_FUSION", "1", 1);
    setenv("NNTR_ACT_STATS", mode == "stats" ? "1" : "0", 1);
    setenv("NNTR_HTP_PROFILE", mode == "profile" ? "3" : "0", 1);
    nntrainer::init_backend();
    Model model;
    if (mode == "stats" || mode == "profile")
      check_bypass(model, mode == "stats");
    else {
      require(mode == "default", "unknown checker mode");
      check_lifetime(model);
      check_owned_gamma(model);
      check_fp32_fallback(model);
      std::printf("ALL_ROUTER_NORM_LIFETIME_CHECKS_PASSED\n");
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  return 0;
}
