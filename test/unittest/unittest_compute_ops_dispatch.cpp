// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_compute_ops_dispatch.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Verify that a per-Context ComputeOps installed via ContextData
 *         actually reaches Tensor::dot / multiply / add through the
 *         tensor's attached ContextData. End-to-end check that
 *         vendor backend dispatch works with virtual dispatch (no
 *         preprocessor branches at the call site).
 */

#include <cmath>
#include <compute_ops.h>
#include <context_data.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include <quantizer.h>
#include <string>
#include <tensor.h>

#include <atomic>
#include <memory>

#ifdef ENABLE_HEXKL
#include <htp_backend.h>
#include <htp_wh_layout.h>
#include <nntr_hvx.h>
#endif

namespace {

/**
 * @brief Per-test counters incremented by the mock subclass.
 *
 * The mock forwards each op to the real (CPU) backend so that result
 * correctness is preserved; the counters confirm dispatch reached the
 * mock and not the global singleton directly.
 */
struct CallCounters {
  std::atomic<int> sgemm{0};
  std::atomic<int> sgemv{0};
  std::atomic<int> ele_mul{0};
  std::atomic<int> ele_add{0};
  std::atomic<int> scopy{0};
};

/**
 * @brief Mock ComputeOps subclass: forwards to a "real" backend
 *        (the global one) for correctness while bumping per-op counters.
 *
 * Because every base-class default just throws, this only overrides
 * the ops the tests exercise. Anything the test setup happens to
 * trigger that's not overridden here would throw — the tests stay
 * inside the overridden subset (sgemm/sgemv/ele_mul/ele_add/scopy).
 */
class MockComputeOps : public nntrainer::ComputeOps {
public:
  MockComputeOps(nntrainer::ComputeOps *real, CallCounters *c) :
    real_(real), counters_(c) {}

  void sgemm_fp32(unsigned int o, bool tA, bool tB, unsigned int M,
                  unsigned int N, unsigned int K, float a, const float *A,
                  unsigned int lda, const float *B, unsigned int ldb, float b,
                  float *C, unsigned int ldc) override {
    counters_->sgemm++;
    real_->sgemm_fp32(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void sgemv_fp32(unsigned int o, bool tA, unsigned int M, unsigned int N,
                  float a, const float *A, unsigned int lda, const float *X,
                  unsigned int iX, float b, float *Y,
                  unsigned int iY) override {
    counters_->sgemv++;
    real_->sgemv_fp32(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  void ele_mul_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    counters_->ele_mul++;
    real_->ele_mul_fp32(N, X, Y, Z, a, b, is, os);
  }
  void ele_add_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    counters_->ele_add++;
    real_->ele_add_fp32(N, X, Y, Z, a, b, is, os);
  }
  void scopy_fp32(unsigned int N, const float *X, unsigned int iX, float *Y,
                  unsigned int iY) override {
    counters_->scopy++;
    real_->scopy_fp32(N, X, iX, Y, iY);
  }

private:
  nntrainer::ComputeOps *real_;
  CallCounters *counters_;
};

/**
 * @brief Test fixture wiring a MockComputeOps into a fresh ContextData for
 *        each test, so dispatch tests can assert on per-call counters.
 */
class ComputeOpsDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    counters = std::make_unique<CallCounters>();
    nntrainer::ensureComputeOps();
    mock_ops = std::make_unique<MockComputeOps>(nntrainer::getComputeOps(),
                                                counters.get());
    ct_data = std::make_shared<nntrainer::ContextData>();
    ct_data->setComputeOps(mock_ops.get());
  }

  std::unique_ptr<CallCounters> counters;
  std::unique_ptr<MockComputeOps> mock_ops;
  std::shared_ptr<nntrainer::ContextData> ct_data;
};

} // namespace

/**
 * @brief A Tensor with no attached ContextData should fall back to the
 *        global ops; no mock counter increments.
 */
TEST_F(ComputeOpsDispatchTest, FallbackToGlobalWhenNoContextData) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  auto out = a.dot(b);

  EXPECT_EQ(counters->sgemm.load(), 0);
  EXPECT_EQ(counters->sgemv.load(), 0);
}

/**
 * @brief When a ContextData with the mock subclass is attached to a
 *        Tensor, calling .dot dispatches through the mock.
 */
TEST_F(ComputeOpsDispatchTest, DotDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);
  auto out = a.dot(b);

  EXPECT_GT(counters->sgemm.load() + counters->sgemv.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 4.0f);
}

/**
 * @brief Element-wise multiply through the attached ContextData should
 *        invoke the mock's ele_mul_fp32.
 */
TEST_F(ComputeOpsDispatchTest, MultiplyDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.multiply(b, out);

  EXPECT_GT(counters->ele_mul.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 6.0f);
}

/**
 * @brief Element-wise add through the attached ContextData should
 *        invoke the mock's ele_add_fp32.
 */
TEST_F(ComputeOpsDispatchTest, AddDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.add(b, out);

  EXPECT_GT(counters->ele_add.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 5.0f);
}

/**
 * @brief Result tensor of a binary op inherits ContextData from `this`,
 *        so a chained op on the result keeps dispatching through the
 *        same backend.
 */
TEST_F(ComputeOpsDispatchTest, ResultInheritsContextDataFromOperand) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);

  auto first = a.dot(b);
  EXPECT_EQ(first.getContextData().get(), ct_data.get());

  int before = counters->sgemm.load() + counters->sgemv.load();
  auto second = first.dot(b);
  int after = counters->sgemm.load() + counters->sgemv.load();
  EXPECT_GT(after, before);
}

/**
 * @brief Replacing the attached ContextData with a different subclass
 *        rebinds dispatch on subsequent calls — the runtime swap
 *        property required for hot-swapping vendor contexts.
 */
TEST_F(ComputeOpsDispatchTest, SwappingContextDataRebindsDispatch) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out1(1, 1, 1, 8);
  nntrainer::Tensor out2(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.multiply(b, out1);
  int after_first = counters->ele_mul.load();
  EXPECT_GT(after_first, 0);

  // Swap to a fresh ContextData carrying its own MockComputeOps with
  // independent counters. Subsequent ops bump fresh counters only.
  auto fresh_counters = std::make_unique<CallCounters>();
  auto fresh_mock = std::make_unique<MockComputeOps>(nntrainer::getComputeOps(),
                                                     fresh_counters.get());
  auto fresh_ct = std::make_shared<nntrainer::ContextData>();
  fresh_ct->setComputeOps(fresh_mock.get());
  a.setContextData(fresh_ct);

  a.multiply(b, out2);
  EXPECT_EQ(counters->ele_mul.load(), after_first);
  EXPECT_GT(fresh_counters->ele_mul.load(), 0);
}

/**
 * @brief Cross-vendor mismatch — when two operands of a binary op
 *        carry DIFFERENT ContextData (e.g. one CPU-resident tensor,
 *        one OpenCL-resident tensor), the op must throw rather than
 *        silently dispatch through one side's ops onto the other
 *        side's incompatible memory. This is the assertion that
 *        protects against the most insidious mixed-backend bug.
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpThrowsOnContextMismatch) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  // Two distinct ContextData instances simulate two different vendor
  // contexts (e.g. CPU + OpenCL) — same kind of mock here, but the
  // identity of the ContextData pointers differs.
  auto ct_other = std::make_shared<nntrainer::ContextData>();
  ct_other->setComputeOps(mock_ops.get());

  a.setContextData(ct_data);
  b.setContextData(ct_other);

  EXPECT_THROW(a.multiply(b, out), std::invalid_argument);
  EXPECT_THROW(a.add(b, out), std::invalid_argument);
  EXPECT_THROW(a.divide(b, out), std::invalid_argument);
}

/**
 * @brief Same ContextData identity on both operands → no throw.
 *        Confirms the mismatch check is keyed on identity, not on
 *        nullness alone (which would be backward-compatibility break).
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpAcceptsSameContext) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  b.setContextData(ct_data); // same instance, not a copy

  EXPECT_NO_THROW(a.multiply(b, out));
}

/**
 * @brief One operand has no ContextData → permissive (legacy code
 *        path). A tensor created without ever touching ContextData
 *        falls back to the global ops table; binary-op'ing it with
 *        a context-attached tensor must NOT throw — that would break
 *        every existing test and call site.
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpAcceptsOneSideUnattached) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  // b has no ContextData — legacy code path

  EXPECT_NO_THROW(a.multiply(b, out));
}

/**
 * @brief Tensor::to(target_ct) deep-copies and re-tags. Result owns
 *        the new ContextData; original is unchanged. After to(), a
 *        previously-mismatched binary op on the migrated tensor
 *        succeeds.
 */
TEST_F(ComputeOpsDispatchTest, ToMigratesContextDataAndUnblocksOp) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  auto ct_other = std::make_shared<nntrainer::ContextData>();
  ct_other->setComputeOps(mock_ops.get());

  a.setContextData(ct_data);
  b.setContextData(ct_other);

  // Migrate b onto a's context. Original b stays on ct_other.
  nntrainer::Tensor b_migrated = b.to(ct_data);
  EXPECT_EQ(b_migrated.getContextData().get(), ct_data.get());
  EXPECT_EQ(b.getContextData().get(), ct_other.get()); // unchanged

  // Now a.multiply(b_migrated) is on the same context — no throw.
  EXPECT_NO_THROW(a.multiply(b_migrated, out));
}

/**
 * @brief A Q4_0 dot on a context-attached tensor must still compute.
 *
 * The scalar Q4_0 GEMM is a CPU-table op. An accelerator ComputeOps only
 * implements the accel-only ops it advertises through supports_*() -- the mock
 * here, like every real accelerator table, implements none of the Q4_0 ops and
 * inherits the base defaults, which throw. M == 1 is the decode step, which
 * never takes the accel batch path, so before the fix this threw
 * "ComputeOps::gemm_q4_0_fp32 not implemented by this backend" on every decode
 * step of a Q4_0 model running on a gpu context.
 */
TEST_F(ComputeOpsDispatchTest, ScalarQ4_0GemmTakesTheCpuTable) {
  nntrainer::init_backend();

  constexpr unsigned int K = 512;
  constexpr unsigned int N = 512;

  nntrainer::Tensor w_fp32(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  for (unsigned int k = 0; k < K; ++k)
    for (unsigned int n = 0; n < N; ++n)
      w_fp32.setValue(0, 0, k, n,
                      0.01f * static_cast<float>((k + 3 * n) % 7) - 0.03f);

  auto quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q4_0);
  nntrainer::Tensor w_q40 =
    quantizer->quantize(w_fp32, nntrainer::Tdatatype::Q4_0);
  ASSERT_EQ(w_q40.getDataType(), nntrainer::Tdatatype::Q4_0);

  // M == 1 is the decode step.
  nntrainer::Tensor act(1, 1, 1, K);
  for (unsigned int k = 0; k < K; ++k)
    act.setValue(0, 0, 0, k, 0.5f - 0.001f * static_cast<float>(k % 11));

  // Reference: no ContextData, so getOps() already resolves to the CPU table.
  nntrainer::Tensor expected(1, 1, 1, N);
  act.dot(w_q40, expected);

  nntrainer::Tensor act_ctx(act);
  act_ctx.setContextData(ct_data);

  nntrainer::Tensor got(1, 1, 1, N);
  ASSERT_NO_THROW(act_ctx.dot(w_q40, got));

  for (unsigned int n = 0; n < N; ++n)
    EXPECT_FLOAT_EQ(got.getValue<float>(0, 0, 0, n),
                    expected.getValue<float>(0, 0, 0, n));
}

#ifdef ENABLE_HEXKL
/** @brief M=1 uses the queue, M=2 uses FastRPC; identical rows must agree.
 * Run with NNTR_HTP_DSPQ_MATH=0 as well to exercise the fallback transport. */
TEST(HtpBackendSmoke, DecodeMathMatchesRpcRows) {
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  using Math = nntrainer::ComputeOps::InferenceMath;
  for (unsigned width : {64u, 2048u, 8192u, 8256u, 11776u, 16384u, 16448u}) {
    std::vector<float> x(2 * width), aux(2 * width), gamma(width), rope(256);
    for (unsigned i = 0; i < width; ++i) {
      x[i] = x[width + i] = (static_cast<int>(i % 37) - 18) * 0.0625f;
      aux[i] = aux[width + i] = (static_cast<int>(i % 31) - 15) * 0.03125f;
      gamma[i] = 0.5f + (i % 9) * 0.0625f;
    }
    for (unsigned i = 0; i < 128; ++i)
      rope[i] = rope[128 + i] = i < 64 ? 0.75f : 0.25f;
    for (unsigned kind = 0; kind < 8; ++kind) {
      const auto op = static_cast<Math>(kind);
      const bool rotate = kind == 4 || kind == 5;
      const unsigned chunk = rotate ? 64u : width;
      const float *param = rotate ? rope.data() : gamma.data();
      std::vector<float> one(width + 32, 12345), two(2 * width + 32, 12345);
      std::vector<float> sum1(width + 32, 12345), sum2(2 * width + 32, 12345);
      if (kind == 7) {
        ASSERT_TRUE(ops->add_rmsnorm_fp32(1, width, 1e-5f, x.data(), aux.data(),
                                          param, sum1.data(), one.data()));
        ASSERT_TRUE(ops->add_rmsnorm_fp32(2, width, 1e-5f, x.data(), aux.data(),
                                          param, sum2.data(), two.data()));
        EXPECT_EQ(0,
                  std::memcmp(sum1.data(), sum2.data(), width * sizeof(float)));
        EXPECT_EQ(sum1[width], 12345);
        EXPECT_EQ(sum2[2 * width], 12345);
      } else {
        ASSERT_TRUE(ops->inference_math_fp32(op, 1, width, chunk, 1e-5f, 1,
                                             x.data(), aux.data(), param,
                                             one.data()));
        ASSERT_TRUE(ops->inference_math_fp32(op, 2, width, chunk, 1e-5f, 1,
                                             x.data(), aux.data(), param,
                                             two.data()));
      }
      const size_t bytes = width * ((kind == 5 || kind == 6) ? 2u : 4u);
      EXPECT_EQ(0, std::memcmp(one.data(), two.data(), bytes))
        << "width=" << width << " kind=" << kind;
      EXPECT_EQ(one[width], 12345);
      EXPECT_EQ(two[2 * width], 12345);
    }
  }
  float dummy = 0;
  EXPECT_FALSE(ops->inference_math_fp32(Math::Add, 1, 31, 31, 1e-5f, 1, &dummy,
                                        &dummy, nullptr, &dummy));
}

/** @brief Queue router preserves logits, tie order and routing weights. */
TEST(HtpBackendSmoke, DecodeRouterMatchesRpcRows) {
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  constexpr unsigned K = 2048, E = 64, top = 4, selected = 9;
  std::vector<float> x(2 * K), w(K * E), gamma(K), bias(E);
  for (unsigned k = 0; k < K; ++k) {
    gamma[k] = 0.5f + (k % 7) * 0.125f;
    for (unsigned e = 0; e < E; ++e)
      w[k * E + e] = (static_cast<int>((k * 3 + e * 7) % 31) - 15) * 0.001f;
  }
  for (unsigned e = 0; e < E; ++e)
    bias[e] = (e % 5) * 0.01f;
  for (unsigned run = 0; run < 8; ++run) {
    for (unsigned k = 0; k < K; ++k)
      x[k] = x[K + k] = (static_cast<int>((k + 3 * run) % 37) - 18) * 0.03125f;
    std::vector<float> l1(E), l2(2 * E), p1(top), p2(2 * top);
    std::vector<unsigned> s1(selected), s2(2 * selected);
    const float *g = run % 2 ? gamma.data() : nullptr;
    ASSERT_NO_THROW(ops->router_logits_fp32(
      1, K, E, x.data(), g, 1e-5f, w.data(), l1.data(), top, selected,
      bias.data(), s1.data(), p1.data(), true));
    ASSERT_NO_THROW(ops->router_logits_fp32(
      2, K, E, x.data(), g, 1e-5f, w.data(), l2.data(), top, selected,
      bias.data(), s2.data(), p2.data(), true));
    EXPECT_EQ(0, std::memcmp(l1.data(), l2.data(), E * sizeof(float)));
    EXPECT_EQ(0,
              std::memcmp(s1.data(), s2.data(), selected * sizeof(unsigned)));
    EXPECT_EQ(0, std::memcmp(p1.data(), p2.data(), top * sizeof(float)));
  }
}

TEST(HtpBackendSmoke, FusedQkvRopeMatchesSeparateCalls) {
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  constexpr unsigned Q = 2048, K = 512, D = 64;
  using Op = nntrainer::ComputeOps::InferenceMath;
  std::vector<float> q(Q), k(K), v(K), cs(2 * D), expected_q(Q);
  std::vector<uint16_t> ko(K + 1), vo(K + 1), expected_k(K), expected_v(K);
  for (unsigned i = 0; i < D; ++i) {
    cs[i] = std::cos(float(i) * 0.03f);
    cs[D + i] = std::sin(float(i) * 0.03f);
  }
  for (unsigned run = 0; run < 8; ++run) {
    for (unsigned i = 0; i < Q; ++i)
      q[i] = (int((i + run) % 31) - 15) * 0.0625f;
    for (unsigned i = 0; i < K; ++i) {
      k[i] = (int((i + run) % 29) - 14) * 0.125f;
      v[i] = (int((i + run) % 23) - 11) * 0.25f;
    }
    ASSERT_TRUE(ops->inference_math_fp32(Op::Rope, 1, Q, D, 0, 1, q.data(),
                                         nullptr, cs.data(),
                                         expected_q.data()));
    ASSERT_TRUE(ops->inference_math_fp32(Op::RopeFp16, 1, K, D, 0, 1, k.data(),
                                         nullptr, cs.data(),
                                         expected_k.data()));
    ASSERT_TRUE(ops->inference_math_fp32(Op::CastFp16, 1, K, D, 0, 1, v.data(),
                                         nullptr, nullptr, expected_v.data()));
    ko[K] = vo[K] = 0x55aa;
    ASSERT_TRUE(ops->rope_qkv_decode(Q, K, D, cs.data(), q.data(), k.data(),
                                     v.data(), q.data(), ko.data(), vo.data()));
    EXPECT_EQ(q, expected_q);
    EXPECT_EQ(0,
              std::memcmp(ko.data(), expected_k.data(), K * sizeof(uint16_t)));
    EXPECT_EQ(0,
              std::memcmp(vo.data(), expected_v.data(), K * sizeof(uint16_t)));
    EXPECT_EQ(ko[K], 0x55aa);
    EXPECT_EQ(vo[K], 0x55aa);
  }
  EXPECT_FALSE(ops->rope_qkv_decode(Q, K, 32, cs.data(), q.data(), k.data(),
                                    v.data(), q.data(), ko.data(), vo.data()));
}

/** Deferred inputs are captured before graph storage can be recycled. */
TEST(HtpBackendSmoke, DeferredNormRouterPreservesResultsAndInputLifetime) {
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  constexpr unsigned K = 2048, E = 64, top = 4, selected = 9;
  std::vector<float> x(K), residual(K), gamma(K), w(K * E), bias(E);
  std::vector<float> norm(K + 1), sum(K + 1), expected_norm(K), expected_sum(K);
  std::vector<float> logits(E), expected_logits(E), weight(top),
    expected_weight(top);
  std::vector<unsigned> ids(selected), expected_ids(selected);
  for (unsigned k = 0; k < K; ++k) {
    gamma[k] = 0.75f + (k % 7) * 0.0625f;
    for (unsigned e = 0; e < E; ++e)
      w[k * E + e] = (int((k * 3 + e * 7) % 31) - 15) * 0.001f;
  }
  for (unsigned e = 0; e < E; ++e)
    bias[e] = (e % 5) * 0.01f;
  for (unsigned run = 0; run < 8; ++run) {
    const bool add = run % 2;
    for (unsigned k = 0; k < K; ++k) {
      x[k] = (int((k + run * 3) % 37) - 18) * 0.03125f;
      residual[k] = (int((k + run) % 17) - 8) * 0.0625f;
    }
    if (add) {
      ASSERT_TRUE(ops->add_rmsnorm_fp32(1, K, 1e-5f, x.data(), residual.data(),
                                        gamma.data(), expected_sum.data(),
                                        expected_norm.data()));
    } else {
      expected_sum = x;
      ASSERT_TRUE(ops->inference_math_fp32(
        nntrainer::ComputeOps::InferenceMath::RmsNorm, 1, K, K, 1e-5f, 1,
        x.data(), nullptr, gamma.data(), expected_norm.data()));
    }
    const float *router_gamma = run % 3 ? nullptr : gamma.data();
    ops->router_logits_fp32(1, K, E, expected_norm.data(), router_gamma, 1e-5f,
                            w.data(), expected_logits.data(), top, selected,
                            bias.data(), expected_ids.data(),
                            expected_weight.data(), true);
    std::fill(norm.begin(), norm.end(), 12345.f);
    std::fill(sum.begin(), sum.end(), 12345.f);
    ASSERT_TRUE(ops->defer_decode_norm(
      "router", K, 1e-5f, x.data(), add ? residual.data() : nullptr,
      gamma.data(), add ? sum.data() : nullptr, norm.data()));
    EXPECT_EQ(norm[0], 12345.f);
    EXPECT_THROW(ops->defer_decode_norm("router", K, 1e-5f, x.data(), nullptr,
                                        gamma.data(), nullptr, norm.data()),
                 std::runtime_error);
    // Simulate the graph assigning the original input allocation to another
    // node.
    std::fill(x.begin(), x.end(), -999.f);
    std::fill(residual.begin(), residual.end(), -999.f);
    EXPECT_THROW(ops->router_logits_fp32(1, K, E, x.data(), router_gamma, 1e-5f,
                                         w.data(), logits.data(), top, selected,
                                         bias.data(), ids.data(), weight.data(),
                                         true),
                 std::runtime_error);
    ASSERT_NO_THROW(ops->router_logits_fp32(
      1, K, E, norm.data(), router_gamma, 1e-5f, w.data(), logits.data(), top,
      selected, bias.data(), ids.data(), weight.data(), true));
    EXPECT_EQ(
      0, std::memcmp(norm.data(), expected_norm.data(), K * sizeof(float)));
    if (add)
      EXPECT_EQ(
        0, std::memcmp(sum.data(), expected_sum.data(), K * sizeof(float)));
    EXPECT_EQ(
      0, std::memcmp(logits.data(), expected_logits.data(), E * sizeof(float)));
    EXPECT_EQ(ids, expected_ids);
    EXPECT_EQ(weight, expected_weight);
    EXPECT_EQ(norm[K], 12345.f);
    EXPECT_EQ(sum[K], 12345.f);
  }
}

// Exercise buffer swapping, host mutation and an intervening queue call.
TEST(HtpBackendSmoke, Q4DecodeRouterPreservesSelectionAndDeferredNorm) {
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  constexpr unsigned K = 2048, E = 64, top = 4, selected = 9;
  std::vector<uint8_t> w(K * E / 2);
  std::vector<float> scales(E), bias(E), x(K), residual(K), gamma(K);
  std::vector<float> ref_norm(K), norm(K), ref_sum(K), sum(K), logits(E);
  std::vector<unsigned> ref_ids(selected), ids(selected);
  std::vector<float> ref_weights(top), weights(top);
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = static_cast<uint8_t>(17 * i + i / 13);
  for (unsigned e = 0; e < E; ++e)
    scales[e] = 0.001f * (1 + e % 7);
  for (unsigned run = 0; run < 12; ++run) {
    for (unsigned k = 0; k < K; ++k) {
      x[k] = run ? (int((k * 3 + run) % 31) - 15) * .03125f : 0.f;
      residual[k] = (int(k % 13) - 6) * .0625f;
      gamma[k] = .75f + (k % 5) * .125f;
    }
    for (unsigned e = 0; e < E; ++e)
      bias[e] = run ? (e % 5) * .01f : 0.f;
    const unsigned kind = run % 3;
    if (kind == 2)
      ASSERT_TRUE(ops->add_rmsnorm_fp32(1, K, 1e-5f, x.data(), residual.data(),
                                        gamma.data(), ref_sum.data(),
                                        ref_norm.data()));
    else if (kind == 1)
      ASSERT_TRUE(ops->inference_math_fp32(
        nntrainer::ComputeOps::InferenceMath::RmsNorm, 1, K, K, 1e-5f, 1.f,
        x.data(), nullptr, gamma.data(), ref_norm.data()));
    else
      ref_norm = x;
    ops->gemm_qs4cx_accel_fp32(w.data(), scales.data(), ref_norm.data(),
                               logits.data(), 1, E, K);
    ASSERT_TRUE(ops->sigmoid_topk_fp32(1, E, top, selected, logits.data(),
                                       bias.data(), ref_ids.data(),
                                       ref_weights.data()));
    norm = x;
    if (kind)
      ASSERT_TRUE(ops->defer_decode_norm(
        "router", K, 1e-5f, x.data(), kind == 2 ? residual.data() : nullptr,
        gamma.data(), kind == 2 ? sum.data() : nullptr, norm.data()));
    ASSERT_TRUE(ops->router_qs4cx_prefill(
      w.data(), scales.data(), norm.data(), logits.data(), 1, K, E, top,
      selected, bias.data(), ids.data(), weights.data()));
    EXPECT_EQ(ids, ref_ids) << run;
    EXPECT_EQ(weights, ref_weights) << run;
    EXPECT_EQ(norm, ref_norm) << run;
    if (kind == 2)
      EXPECT_EQ(sum, ref_sum) << run;
    if (!run)
      for (unsigned i = 0; i < selected; ++i)
        EXPECT_EQ(ids[i], i);
  }
  const auto saved = ids;
  EXPECT_FALSE(ops->router_qs4cx_prefill(
    w.data(), scales.data(), norm.data(), logits.data(), 1, K - 1, E, top,
    selected, bias.data(), ids.data(), weights.data()));
  EXPECT_EQ(ids, saved);
}

TEST(HtpBackendSmoke, RouterInputReuseRejectsChangedAndOverwrittenRows) {
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  constexpr unsigned K = 64, E = 64;
  auto make_weight = [](unsigned cols) {
    nntrainer::Tensor t(
      1, 1, K, cols,
      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QS4CX_WH});
    std::vector<int8_t> w(K * cols);
    for (unsigned k = 0; k < K; ++k)
      for (unsigned n = 0; n < cols; ++n)
        w[k * cols + n] = int((k + 3 * n) % 15) - 7;
    nntrainer::whPack(w.data(), K, cols, t.getData<uint8_t>());
    for (unsigned n = 0; n < cols; ++n) {
      t.getScale<float>()[n] = 0.03125f;
      float sum = 0;
      for (unsigned k = 0; k < K; ++k)
        sum += w[k * cols + n];
      t.getScale<float>()[cols + n] = sum;
    }
    return t;
  };
  auto gu = make_weight(2 * K), dn = make_weight(K);
  std::vector<float> x(K), gamma(K, 0.875f), norm(K), expected(K), output(K);
  std::vector<float> w(K * E, 0.03125f), logits(E), rw(1), bias(E, 0.f);
  unsigned id;
  std::vector<uint8_t> q4_w(K * E / 2, 0x79);
  std::vector<float> q4_scale(E, .03125f);
  auto moe = [&](const float *input, float *result) {
    ops->gemm_qs4cx_moe_layer_fp32(
      {gu.getData<uint8_t>()}, {gu.getScale<float>()}, {dn.getData<uint8_t>()},
      {dn.getScale<float>()}, {0u}, {1u}, {1.f}, input, result, 1, K, K, K,
      true);
  };
  for (unsigned run = 0; run < 18; ++run) {
    for (unsigned k = 0; k < K; ++k)
      x[k] = (int((k + run) % 17) - 8) * 0.125f;
    ASSERT_TRUE(ops->inference_math_fp32(
      nntrainer::ComputeOps::InferenceMath::RmsNorm, 1, K, K, 1e-5f, 1.f,
      x.data(), nullptr, gamma.data(), norm.data()));
    auto reference = norm;
    if (run % 3 == 1)
      reference[K / 2] += 0.25f;
    ASSERT_NO_THROW(moe(reference.data(), expected.data()));
    ASSERT_TRUE(ops->defer_decode_norm("router", K, 1e-5f, x.data(), nullptr,
                                       gamma.data(), nullptr, norm.data()));
    if (run < 9)
      ASSERT_NO_THROW(ops->router_logits_fp32(
        1, K, E, norm.data(), nullptr, 1e-5f, w.data(), logits.data(), 1, 1,
        bias.data(), &id, rw.data(), true));
    else
      ASSERT_TRUE(ops->router_qs4cx_prefill(q4_w.data(), q4_scale.data(),
                                            norm.data(), logits.data(), 1, K, E,
                                            1, 1, bias.data(), &id, rw.data()));
    if (run % 3 == 1)
      norm[K / 2] += 0.25f;
    if (run % 3 == 2) {
      std::vector<float> scratch(K);
      ASSERT_TRUE(ops->inference_math_fp32(
        nntrainer::ComputeOps::InferenceMath::RmsNorm, 1, K, K, 1e-5f, 1.f,
        x.data(), nullptr, gamma.data(), scratch.data()));
    }
    ASSERT_NO_THROW(moe(norm.data(), output.data()));
    EXPECT_EQ(output, expected) << "case=" << run;
  }
}

TEST(HtpBackendSmoke, DeferredNormConvPreservesOutputAndHistory) {
  // Compare both transports in the same session. Restore the environment
  // even if an assertion exits the test early.
  struct QueueMode {
    bool present = std::getenv("NNTR_HTP_DSPQ_CONV") != nullptr;
    std::string saved = present ? std::getenv("NNTR_HTP_DSPQ_CONV") : "";
    static void set(const char *value) {
#ifdef _WIN32
      _putenv_s("NNTR_HTP_DSPQ_CONV", value ? value : "");
#else
      if (value)
        setenv("NNTR_HTP_DSPQ_CONV", value, 1);
      else
        unsetenv("NNTR_HTP_DSPQ_CONV");
#endif
    }
    ~QueueMode() { set(present ? saved.c_str() : nullptr); }
  } mode;
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  constexpr unsigned K = 2048, C = 2048, N = 2048;
  auto make_weight = [](unsigned rows, unsigned cols) {
    nntrainer::Tensor t(
      1, 1, rows, cols,
      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QS4CX_WH});
    std::vector<int8_t> v(rows * cols);
    for (unsigned k = 0; k < rows; ++k)
      for (unsigned n = 0; n < cols; ++n)
        v[k * cols + n] = int((k + 3 * n) % 15) - 7;
    nntrainer::whPack(v.data(), rows, cols, t.getData<uint8_t>());
    for (unsigned n = 0; n < cols; ++n) {
      t.getScale<float>()[n] = 0.03125f;
      float sum = 0;
      for (unsigned k = 0; k < rows; ++k)
        sum += v[k * cols + n];
      t.getScale<float>()[cols + n] = sum;
    }
    return t;
  };
  auto in = make_weight(K, 3 * C), out = make_weight(C, N);
  std::vector<float> conv(3 * C), state(2 * C), expected_state(2 * C);
  std::vector<float> x(K), residual(K), gamma(K, 0.875f);
  std::vector<float> sum(K), norm(K), expected_sum(K), expected_norm(K);
  std::vector<float> y(N), expected_y(N);
  for (unsigned i = 0; i < 3 * C; ++i)
    conv[i] = (int(i % 7) - 3) * 0.0625f;
  auto call = [&](const float *input, float *output, float *history,
                  unsigned rows = 1) {
    ops->gemm_qs4cx_conv_block_fp32(in.getData<char>(), conv.data(),
                                    out.getData<char>(), input, output, history,
                                    rows, K, C, N, in.getScale<float>(),
                                    out.getScale<float>(), true);
  };
  // Prefill remains on FastRPC; carry produced there must seed queue decode.
  std::vector<float> prompt(3 * K), prefill(3 * N), expected_prefill(3 * N);
  for (unsigned k = 0; k < prompt.size(); ++k)
    prompt[k] = (int(k % 19) - 9) * 0.0625f;
  QueueMode::set("0");
  ASSERT_NO_THROW(
    call(prompt.data(), expected_prefill.data(), expected_state.data(), 3));
  QueueMode::set("1");
  ASSERT_NO_THROW(call(prompt.data(), prefill.data(), state.data(), 3));
  EXPECT_EQ(0, std::memcmp(prefill.data(), expected_prefill.data(),
                           prefill.size() * sizeof(float)));
  EXPECT_EQ(0, std::memcmp(state.data(), expected_state.data(),
                           state.size() * sizeof(float)));
  for (unsigned run = 0; run < 16; ++run) {
    SCOPED_TRACE(run);
    if (run == 2 || run == 11)
      gamma[K / 2] += 0.125f;
    if (run == 3 || run == 12)
      conv[C / 2] += 0.0625f;
    if (run == 4) {
      // A cached history must still honor caller reset and restore operations.
      std::fill(state.begin(), state.end(), 0.f);
      std::fill(expected_state.begin(), expected_state.end(), 0.f);
    }
    if (run == 5) {
      state[C / 2] += 0.125f;
      expected_state[C / 2] += 0.125f;
    }
    if (run == 6) {
      QueueMode::set("0");
      ASSERT_NO_THROW(
        call(prompt.data(), expected_prefill.data(), expected_state.data(), 3));
      QueueMode::set("1");
      ASSERT_NO_THROW(call(prompt.data(), prefill.data(), state.data(), 3));
      EXPECT_EQ(prefill, expected_prefill);
      EXPECT_EQ(state, expected_state);
    }
    const bool add = run % 2;
    for (unsigned k = 0; k < K; ++k) {
      x[k] = (int((k + run) % 17) - 8) * 0.125f;
      residual[k] = (int((3 * k + run) % 13) - 6) * 0.0625f;
    }
    if (add)
      ASSERT_TRUE(ops->add_rmsnorm_fp32(1, K, 1e-5f, x.data(), residual.data(),
                                        gamma.data(), expected_sum.data(),
                                        expected_norm.data()));
    else
      ASSERT_TRUE(ops->inference_math_fp32(
        nntrainer::ComputeOps::InferenceMath::RmsNorm, 1, K, K, 1e-5f, 1,
        x.data(), nullptr, gamma.data(), expected_norm.data()));
    QueueMode::set("0");
    ASSERT_NO_THROW(
      call(expected_norm.data(), expected_y.data(), expected_state.data()));
    QueueMode::set("1");
    auto plain_state = state;
    ASSERT_NO_THROW(call(expected_norm.data(), y.data(), plain_state.data()));
    EXPECT_EQ(
      0, std::memcmp(y.data(), expected_y.data(), y.size() * sizeof(float)))
      << "plain queue token=" << run;
    EXPECT_EQ(0, std::memcmp(plain_state.data(), expected_state.data(),
                             plain_state.size() * sizeof(float)))
      << "plain queue token=" << run;
    const bool deferred = run % 5 != 3;
    if (deferred)
      ASSERT_TRUE(ops->defer_decode_norm(
        "conv", K, 1e-5f, x.data(), add ? residual.data() : nullptr,
        gamma.data(), add ? sum.data() : nullptr, norm.data()));
    std::fill(x.begin(), x.end(), -999.f);
    std::fill(residual.begin(), residual.end(), -999.f);
    const float saved_gamma = gamma[K / 2];
    gamma[K / 2] = -999.f; // Deferred inputs include a gamma snapshot.
    // Leave the same cached state through a FastRPC decode, then resume queue.
    if (run == 9)
      QueueMode::set("0");
    ASSERT_NO_THROW(call(deferred ? norm.data() : expected_norm.data(),
                         y.data(), state.data()));
    gamma[K / 2] = saved_gamma;
    if (deferred)
      EXPECT_EQ(0, std::memcmp(norm.data(), expected_norm.data(),
                               norm.size() * sizeof(float)));
    if (deferred && add)
      EXPECT_EQ(0, std::memcmp(sum.data(), expected_sum.data(),
                               sum.size() * sizeof(float)));
    EXPECT_EQ(
      0, std::memcmp(y.data(), expected_y.data(), y.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(state.data(), expected_state.data(),
                             state.size() * sizeof(float)));
  }
  // More distinct histories than the bounded queue cache: excess entries
  // must use the original transport without changing caller-visible state.
  std::fill(expected_state.begin(), expected_state.end(), 0.f);
  QueueMode::set("0");
  ASSERT_NO_THROW(
    call(expected_norm.data(), expected_y.data(), expected_state.data()));
  QueueMode::set("1");
  std::vector<std::vector<float>> histories(70, std::vector<float>(2 * C));
  for (auto &history : histories) {
    ASSERT_NO_THROW(call(expected_norm.data(), y.data(), history.data()));
    EXPECT_EQ(
      0, std::memcmp(y.data(), expected_y.data(), y.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(history.data(), expected_state.data(),
                             history.size() * sizeof(float)));
  }
}

/** @brief WH FC dispatch preserves columns across VTCM slices and row tails. */
TEST(HtpBackendSmoke, WhFcPrefillAndDecode) {
  struct QueueModes {
    const char *keys[2] = {"NNTR_HTP_DSPQ_FC", "NNTR_HTP_DSPQ_QKV"};
    bool present[2];
    std::string saved[2];
    QueueModes() {
      for (unsigned i = 0; i < 2; ++i) {
        const char *v = std::getenv(keys[i]);
        present[i] = v != nullptr;
        saved[i] = v ? v : "";
      }
    }
    static void set(const char *key, const char *v) {
#ifdef _WIN32
      _putenv_s(key, v ? v : "");
#else
      if (v)
        setenv(key, v, 1);
      else
        unsetenv(key);
#endif
    }
    void enable(bool enabled) {
      for (const char *key : keys)
        set(key, enabled ? "1" : "0");
    }
    ~QueueModes() {
      for (unsigned i = 0; i < 2; ++i)
        set(keys[i], present[i] ? saved[i].c_str() : nullptr);
    }
  } modes;
  nntrainer::init_backend();
  auto context = std::make_shared<nntrainer::ContextData>();
  context->setComputeOps(nntrainer::get_htp_ops());
  for (const auto shape :
       {std::pair<unsigned, unsigned>{64, 64}, {2048, 2304}, {11776, 64}}) {
    const auto [K, N] = shape;
    std::vector<int8_t> values(static_cast<size_t>(K) * N);
    std::vector<float> sums(N, 0.0f);
    for (unsigned k = 0; k < K; ++k)
      for (unsigned n = 0; n < N; ++n) {
        const int v = static_cast<int>((k + 3 * n) % 15) - 7;
        values[static_cast<size_t>(k) * N + n] = v;
        sums[n] += v;
      }
    nntrainer::Tensor weights(
      1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QS4CX_WH});
    nntrainer::whPack(values.data(), K, N, weights.getData<uint8_t>());
    for (unsigned n = 0; n < N; ++n) {
      weights.getScale<float>()[n] = 0.125f;
      weights.getScale<float>()[N + n] = sums[n];
    }
    for (unsigned M : {65u, 1u}) {
      nntrainer::Tensor input(1, 1, M, K), output(1, 1, M, N);
      input.setValue(M == 1 ? 2.0f : 1.0f);
      EXPECT_THROW(input.dot(weights, output), std::invalid_argument);
      input.setContextData(context);
      ASSERT_NO_THROW(input.dot(weights, output));
      for (unsigned m = 0; m < M; ++m)
        for (unsigned n = 0; n < N; ++n)
          ASSERT_NEAR(output.getData<float>()[static_cast<size_t>(m) * N + n],
                      sums[n] * (M == 1 ? 0.25f : 0.125f), 1e-3f)
            << "K=" << K << " N=" << N << " M=" << M << " col=" << n;

      // Two normalized projections and one raw projection, as in LFM2 QKV.
      // The wide shape crosses a VTCM column slice; 65 rows tests its tail.
      for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
        input.getData<float>()[i] =
          (static_cast<int>((i * 7) % 29) - 14) * 0.0625f;
      modes.enable(false);
      ASSERT_NO_THROW(input.dot(weights, output));
      std::vector<float> reference(output.getData<float>(),
                                   output.getData<float>() + M * N);
      modes.enable(true);
      ASSERT_NO_THROW(input.dot(weights, output));
      EXPECT_EQ(0, std::memcmp(reference.data(), output.getData<float>(),
                               reference.size() * sizeof(float)));
      auto *ops = nntrainer::get_htp_ops();
      std::vector<float> gamma(128), expected(M * N), q(M * N), k(M * N),
        v(M * N);
      for (unsigned i = 0; i < 64; ++i)
        gamma[i] = gamma[64 + i] = 0.5f + i * 0.015625f;
      ASSERT_TRUE(ops->inference_math_fp32(
        nntrainer::ComputeOps::InferenceMath::RmsNorm, M, N, 64, 1e-5f, 1.0f,
        output.getData<float>(), nullptr, gamma.data(), expected.data()));
      void *w = weights.getData<uint8_t>();
      float *scale = weights.getScale<float>();
      ASSERT_NO_THROW(ops->gemm_q4_0_batch_norm_fp32(
        {w, w, w}, {scale, scale, scale}, input.getData<float>(),
        {q.data(), k.data(), v.data()}, M, {N, N, N}, K, nullptr,
        {64u, 64u, 0u}, gamma.data(), 1e-5f, nullptr, 0u, 0u, true));
      for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_FLOAT_EQ(q[i], expected[i]) << "Q at " << i;
        ASSERT_FLOAT_EQ(k[i], expected[i]) << "K at " << i;
        ASSERT_FLOAT_EQ(v[i], output.getData<float>()[i]) << "V at " << i;
      }
      if (M == 1) {
        // Include pre-norm and RoPE metadata; compare transports bit for bit.
        std::vector<float> pre(K, 0.875f), cs(128), rq(N), rk(N), rv(N);
        for (unsigned i = 0; i < 64; ++i) {
          cs[i] = std::cos(i * 0.03f);
          cs[64 + i] = std::sin(i * 0.03f);
        }
        modes.enable(false);
        ops->gemm_q4_0_batch_norm_fp32(
          {w, w, w}, {scale, scale, scale}, input.getData<float>(),
          {rq.data(), rk.data(), rv.data()}, 1, {N, N, N}, K, pre.data(),
          {64u, 64u, 0u}, gamma.data(), 1e-5f, cs.data(), 64, 2, true);
        modes.enable(true);
        ops->gemm_q4_0_batch_norm_fp32(
          {w, w, w}, {scale, scale, scale}, input.getData<float>(),
          {q.data(), k.data(), v.data()}, 1, {N, N, N}, K, pre.data(),
          {64u, 64u, 0u}, gamma.data(), 1e-5f, cs.data(), 64, 2, true);
        EXPECT_EQ(0,
                  std::memcmp(rq.data(), q.data(), rq.size() * sizeof(float)));
        EXPECT_EQ(0,
                  std::memcmp(rk.data(), k.data(), rk.size() * sizeof(float)));
        EXPECT_EQ(0,
                  std::memcmp(rv.data(), v.data(), rv.size() * sizeof(float)));
      }
      if (M == 1 && K <= 4096) {
        for (bool add : {false, true}) {
          modes.enable(false);
          std::vector<float> ng(K, 0.875f), residual(K, 0.03125f), sum(K),
            norm(K), fs(K), fn(K), fq(N), fk(N), fv(N);
          if (add) {
            ASSERT_TRUE(ops->add_rmsnorm_fp32(
              1, K, 1e-5f, input.getData<float>(), residual.data(), ng.data(),
              sum.data(), norm.data()));
          } else {
            ASSERT_TRUE(ops->inference_math_fp32(
              nntrainer::ComputeOps::InferenceMath::RmsNorm, 1, K, K, 1e-5f, 1,
              input.getData<float>(), nullptr, ng.data(), norm.data()));
          }
          ops->gemm_q4_0_batch_norm_fp32(
            {w, w, w}, {scale, scale, scale}, norm.data(),
            {q.data(), k.data(), v.data()}, 1, {N, N, N}, K, nullptr,
            {64u, 64u, 0u}, gamma.data(), 1e-5f, nullptr, 0, 0, true);
          modes.enable(true);
          ASSERT_TRUE(
            ops->defer_decode_norm("qkv", K, 1e-5f, input.getData<float>(),
                                   add ? residual.data() : nullptr, ng.data(),
                                   add ? fs.data() : nullptr, fn.data()));
          ASSERT_NO_THROW(ops->gemm_q4_0_batch_norm_fp32(
            {w, w, w}, {scale, scale, scale}, fn.data(),
            {fq.data(), fk.data(), fv.data()}, 1, {N, N, N}, K, nullptr,
            {64u, 64u, 0u}, gamma.data(), 1e-5f, nullptr, 0, 0, true));
          if (add)
            EXPECT_EQ(
              0, std::memcmp(fs.data(), sum.data(), fs.size() * sizeof(float)));
          EXPECT_EQ(
            0, std::memcmp(fn.data(), norm.data(), fn.size() * sizeof(float)));
          EXPECT_EQ(
            0, std::memcmp(fq.data(), q.data(), fq.size() * sizeof(float)));
          EXPECT_EQ(
            0, std::memcmp(fk.data(), k.data(), fk.size() * sizeof(float)));
          EXPECT_EQ(
            0, std::memcmp(fv.data(), v.data(), fv.size() * sizeof(float)));
        }
      }
    }
  }
}

/** Malformed wire metadata is rejected before cache mutation. */
TEST(HtpBackendSmoke, ProducerQ8RejectsInvalidMetadata) {
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  const int h = ops->kv_cache_q_register(0, 32, 1, 64);
  ASSERT_GE(h, 0);
  struct Release {
    nntrainer::ComputeOps *ops;
    int h;
    ~Release() { ops->kv_cache_q_release(h); }
  } release{ops, h};
  std::vector<uint8_t> k(64, 1), v(64, 2), q(64, 128);
  float scales[2] = {1.f, 1.f}, qs = 1.f;
  int32_t zp = 128;
  uint32_t stats[4] = {};
  std::vector<float> out(65, 12345.f);
  auto call = [&](int q_len, unsigned row0 = 0u, unsigned rows = 1u) {
    return nntr_hvx_attn_q8_step(
      nntrainer::HtpBackend::global().handle(), h, row0, rows, 1, 0, 1, 1, 0,
      0.f, k.data(), rows * 64, v.data(), rows * 64, scales, rows * 2, q.data(),
      q_len, &qs, 1, &zp, 1, nullptr, 0, out.data(), 64, stats, 4);
  };
  EXPECT_NE(call(63), 0);
  EXPECT_NE(call(64, 2), 0);
  qs = std::nanf("");
  EXPECT_NE(call(64), 0);
  qs = 1.f;
  zp = 256;
  EXPECT_NE(call(64), 0);
  zp = 128;
  scales[1] = 0.f;
  EXPECT_NE(call(64), 0);
  scales[1] = 1.f;
  // FastRPC rout buffers have unspecified contents on a failed call.
  EXPECT_EQ(out[64], 12345.f);
  ASSERT_EQ(call(64), 0);
  for (unsigned i = 0; i < 64; ++i)
    EXPECT_NEAR(out[i], 2.f, 1e-5f);
  EXPECT_EQ(out[64], 12345.f);
  std::fill(v.begin(), v.end(), uint8_t(9));
  scales[1] = 0.f;
  EXPECT_NE(call(64), 0);
  scales[1] = 1.f;
  // The rejected append must leave the previous V=2 row intact.
  ASSERT_EQ(call(64, 1, 0), 0);
  for (unsigned i = 0; i < 64; ++i)
    EXPECT_NEAR(out[i], 2.f, 1e-5f);
}

/** Producer quantization must preserve the legacy A8/KV8 path. */
TEST(HtpBackendSmoke, ProducerQ8AttentionMatchesLegacyAcrossRewind) {
  struct WireMode {
    bool present = std::getenv("NNTR_HTP_ATTN_Q8_WIRE") != nullptr;
    std::string saved = present ? std::getenv("NNTR_HTP_ATTN_Q8_WIRE") : "";
    static void set(const char *value) {
#ifdef _WIN32
      _putenv_s("NNTR_HTP_ATTN_Q8_WIRE", value ? value : "");
#else
      if (value)
        setenv("NNTR_HTP_ATTN_Q8_WIRE", value, 1);
      else
        unsetenv("NNTR_HTP_ATTN_Q8_WIRE");
#endif
    }
    ~WireMode() { set(present ? saved.c_str() : nullptr); }
  } mode;
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  constexpr unsigned rows = 1088, hq = 32, hk = 8, hd = 64;
  constexpr unsigned qw = hq * hd, kw = hk * hd;
  std::vector<uint16_t> k(rows * kw), v(rows * kw);
  std::vector<float> q(1024 * qw), old(1024 * qw + 1, 12345.f),
    wire(1024 * qw + 1, 12345.f);
  for (unsigned i = 0; i < k.size(); ++i) {
    k[i] = static_cast<uint16_t>(0x2800u + i % 1024u + (i % 3 ? 0 : 0x8000u));
    v[i] = static_cast<uint16_t>(0x3000u + i % 1024u + (i % 5 ? 0 : 0x8000u));
  }
  for (unsigned i = 0; i < q.size(); ++i)
    q[i] = (int(i % 29) - 14) * 0.0625f;
  const int a = ops->kv_cache_q_register(0, rows, hk, hd);
  const int b = ops->kv_cache_q_register(0, rows, hk, hd);
  struct Release {
    nntrainer::ComputeOps *ops;
    int a, b;
    ~Release() {
      ops->kv_cache_q_release(a);
      ops->kv_cache_q_release(b);
    }
  } release{ops, a, b};
  ASSERT_GE(a, 0);
  ASSERT_GE(b, 0);
  unsigned synced = 0;
  auto check = [&](unsigned from, unsigned to, bool append = true) {
    const unsigned n = to - from;
    synced = std::min(synced, from);
    const unsigned row0 = append ? synced : to,
                   count = append ? to - synced : 0;
    old[n * qw] = wire[n * qw] = 12345.f;
    WireMode::set("0");
    ASSERT_TRUE(ops->sdpa_q_kvcache(
      a, row0, count, kw, count ? k.data() + row0 * kw : nullptr,
      count ? v.data() + row0 * kw : nullptr, q.data(), qw, n, from, to, hq, hk,
      hd, 0, 0, nullptr, old.data(), qw));
    WireMode::set("1");
    ASSERT_TRUE(ops->sdpa_q_kvcache(
      b, row0, count, kw, count ? k.data() + row0 * kw : nullptr,
      count ? v.data() + row0 * kw : nullptr, q.data(), qw, n, from, to, hq, hk,
      hd, 0, 0, nullptr, wire.data(), qw));
    double error = 0.0, signal = 0.0;
    for (unsigned i = 0; i < n * qw; ++i) {
      ASSERT_TRUE(std::isfinite(wire[i]));
      const double diff = wire[i] - old[i];
      error += diff * diff;
      signal += double(old[i]) * old[i];
    }
    EXPECT_LT(error, std::max(1e-18, signal * 1e-5)) << "from=" << from;
    EXPECT_EQ(old[n * qw], 12345.f);
    EXPECT_EQ(wire[n * qw], 12345.f);
    synced = to;
  };
  check(0, 1024);
  for (unsigned pos : {1024u, 1025u, 1055u, 1056u, 1024u}) {
    for (unsigned i = 0; i < kw; ++i) {
      k[pos * kw + i] ^= 0x8000u;
      v[pos * kw + i] ^= 0x8000u;
    }
    check(pos, pos + 1);
  }
  check(1024, 1025, false);
  for (unsigned i = 0; i < 257 * kw; ++i)
    v[i] ^= 0x8000u;
  check(0, 257);
  check(257, 258);
}

/** The resident FP16 path must preserve cache tails, rewinds and resets. */
TEST(HtpBackendSmoke, ResidentFp16AttentionMatchesRawAcrossRewind) {
  // This binary's first raw-attention call selects the HMX reference.
  struct HmxMode {
    bool present = std::getenv("NNTR_HTP_ATTN_DECODE_HMX") != nullptr;
    std::string saved = present ? std::getenv("NNTR_HTP_ATTN_DECODE_HMX") : "";
    static void set(const char *value) {
#ifdef _WIN32
      _putenv_s("NNTR_HTP_ATTN_DECODE_HMX", value ? value : "");
#else
      if (value)
        setenv("NNTR_HTP_ATTN_DECODE_HMX", value, 1);
      else
        unsetenv("NNTR_HTP_ATTN_DECODE_HMX");
#endif
    }
    HmxMode() { set("1"); }
    ~HmxMode() { set(present ? saved.c_str() : nullptr); }
  } mode;
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  constexpr unsigned rows = 1088, hq = 32, hk = 8, hd = 64;
  constexpr unsigned qw = hq * hd, kw = hk * hd;
  std::vector<uint16_t> k(rows * kw), v(rows * kw);
  std::vector<float> q(1024 * qw), raw(1024 * qw + 1, 12345.f),
    resident(1024 * qw + 1, 12345.f);
  for (unsigned i = 0; i < k.size(); ++i) {
    k[i] = static_cast<uint16_t>(0x2800u + i % 1024u + (i % 3 ? 0 : 0x8000u));
    v[i] = static_cast<uint16_t>(0x3000u + i % 1024u + (i % 5 ? 0 : 0x8000u));
  }
  for (unsigned i = 0; i < q.size(); ++i)
    q[i] = (int(i % 29) - 14) * 0.0625f;
  const int handle = ops->kv_cache_q_register(16, rows, hk, hd);
  ASSERT_GE(handle, 0);
  struct Release {
    nntrainer::ComputeOps *ops;
    int handle;
    ~Release() { ops->kv_cache_q_release(handle); }
  } release{ops, handle};
  unsigned synced = 0;
  auto check = [&](unsigned from, unsigned to) {
    const unsigned n = to - from;
    synced = std::min(synced, from);
    raw[n * qw] = resident[n * qw] = 12345.f;
    ASSERT_TRUE(ops->sdpa_fp16_kvcache(q.data(), qw, k.data(), v.data(), kw, n,
                                       from, to, hq, hk, hd, 0, 0, nullptr,
                                       raw.data(), qw));
    ASSERT_TRUE(ops->sdpa_q_kvcache(
      handle, synced, to - synced, kw, k.data() + synced * kw,
      v.data() + synced * kw, q.data(), qw, n, from, to, hq, hk, hd, 0, 0,
      nullptr, resident.data(), qw));
    EXPECT_EQ(0,
              std::memcmp(raw.data(), resident.data(), n * qw * sizeof(float)))
      << "from=" << from << " to=" << to;
    EXPECT_EQ(raw[n * qw], 12345.f);
    EXPECT_EQ(resident[n * qw], 12345.f);
    synced = to;
  };
  check(0, 1024);
  for (unsigned pos : {1024u, 1025u, 1055u, 1056u, 1024u}) {
    for (unsigned i = 0; i < kw; ++i) {
      k[pos * kw + i] ^= 0x8000u;
      v[pos * kw + i] ^= 0x8000u;
    }
    check(pos, pos + 1);
  }
  // A fresh prompt reuses the allocation but rewrites its prefix.
  for (unsigned i = 0; i < 257 * kw; ++i)
    v[i] ^= 0x8000u;
  check(0, 257);
  check(257, 258);
}

/**
 * @brief The HTP ops table resolves and every op on it still computes.
 *
 * No kernel is wired to the NPU yet, so an engine="htp" model has to keep
 * working by falling through to the CPU implementation. Asserting a real
 * result -- rather than merely that the pointer is non-null -- is what
 * catches a table whose entries throw, which is what the base ComputeOps
 * defaults do. The accelerator predicates arrive with the first kernel.
 */
TEST(HtpBackendSmoke, TableFallsThroughToCpu) {
  auto *htp = nntrainer::get_htp_ops();
  ASSERT_NE(htp, nullptr);
  nntrainer::ensureComputeOps();

  constexpr unsigned int N = 4;
  const float x[N] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float y[N] = {10.0f, 20.0f, 30.0f, 40.0f};
  float z[N] = {0.0f, 0.0f, 0.0f, 0.0f};

  ASSERT_NO_THROW(htp->ele_add_fp32(N, x, y, z, 1.0f, 0.0f, 1, 1));
  for (unsigned int i = 0; i < N; ++i)
    EXPECT_FLOAT_EQ(z[i], x[i] + y[i]);
}
#endif // ENABLE_HEXKL

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
