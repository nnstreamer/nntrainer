// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_hvx_attn.cpp
 * @date   07 Aug 2026
 * @brief  Device gate for PHASE A (S = Q.Kt) of the fused attention path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The oracle is mha_htp_host_scores -- the same PHASE A mha_htp_host_forward
 * runs, not a second opinion written beside it. Both sides do identical
 * integer arithmetic (u8 activation x iX weight -> int32 -> the same dequant
 * formula), so agreement should be near exact; the tolerance asserted is
 * MHA_HTP_U8_TASKS.md Task 3's, and the observed value is reported so the
 * margin is visible rather than assumed.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

#include "mha_htp_host_model.h"

namespace {

constexpr int kDspOffset = 0x80000400;

std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

/**
 * @brief Opens an unsigned-PD CDSP session for each test.
 *
 * Same shape as unittest_hvx_softmax.cpp's fixture: a failure here is a hard
 * FAIL rather than a skip, because proving the DSP comes up is part of what
 * this test measures.
 */
class HtpSession : public ::testing::Test {
protected:
  void SetUp() override {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);

    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";
  }

  void TearDown() override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }

  remote_handle64 handle_ = 0;
};

class HvxAttnScores : public HtpSession {};

/** @brief fp32 -> fp16 bits, matching what the KV cache holds. */
uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t man = x & 0x7FFFFFu;
  if (exp <= 0) {
    return (uint16_t)sign;
  }
  if (exp >= 31) {
    return (uint16_t)(sign | 0x7C00u);
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

struct AttnCfg {
  uint32_t kv_len, n_query, gqa, nch, head_dim, T;
  hexkl_w_width w_k, w_v;
};

const char *wname(hexkl_w_width w) { return w == HEXKL_W_I4 ? "I4" : "I8"; }

/**
 * @brief Registers, appends the whole cache, and returns the device S band for
 *        one head, alongside the host model's.
 */
void run_case(remote_handle64 handle, const AttnCfg &c, uint32_t head,
              std::vector<float> *dev, std::vector<float> *ref) {
  const uint32_t M = c.n_query * c.gqa;
  const uint32_t n_blocks = (c.kv_len + c.T - 1u) / c.T;
  const uint32_t band = n_blocks * M * c.T;

  std::mt19937 rng(c.kv_len * 7919u + c.n_query * 131u + c.gqa * 17u +
                   c.head_dim + head);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);

  /* Realistic KV: a shared per-channel component plus per-position noise.
   * i.i.d. content makes the attention output a near-total cancellation and
   * tells you nothing (MHA_HTP_PLAN.md §9.2). */
  std::vector<float> base(c.nch * c.head_dim);
  for (float &x : base) {
    x = d(rng);
  }
  std::vector<uint16_t> k16((size_t)c.kv_len * c.nch * c.head_dim);
  std::vector<uint16_t> v16(k16.size());
  for (uint32_t r = 0; r < c.kv_len; ++r) {
    for (uint32_t h = 0; h < c.nch; ++h) {
      for (uint32_t dd = 0; dd < c.head_dim; ++dd) {
        const size_t i = ((size_t)r * c.nch + h) * c.head_dim + dd;
        k16[i] = f32_to_f16(base[h * c.head_dim + dd] + 0.5f * d(rng));
        v16[i] = f32_to_f16(base[h * c.head_dim + dd] + 0.5f * d(rng));
      }
    }
  }
  std::vector<float> q_band((size_t)M * c.head_dim);
  for (float &x : q_band) {
    x = d(rng);
  }

  uint32_t h_attn = 0;
  int err =
    nntr_hvx_attn_register(handle, c.nch, c.gqa, c.head_dim, c.kv_len, c.T,
                           (uint32_t)c.w_k, (uint32_t)c.w_v, &h_attn);
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_register: " << hex(err);

  err = nntr_hvx_attn_kv_append(handle, h_attn, 0u, c.kv_len, k16.data(),
                                (int)k16.size(), v16.data(), (int)v16.size());
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_kv_append: " << hex(err);

  dev->assign(band, 0.0f);
  err = nntr_hvx_attn_scores_debug(handle, h_attn, head, M, q_band.data(),
                                   (int)q_band.size(), dev->data(), (int)band);
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_scores_debug: " << hex(err);

  err = nntr_hvx_attn_release(handle, h_attn);
  ASSERT_EQ(err, AEE_SUCCESS) << "attn_release: " << hex(err);

  ref->assign(band, 0.0f);
  mha_htp_host_scores(c.kv_len, c.nch, c.head_dim, c.T, M, head, c.w_k,
                      q_band.data(), k16.data(), ref->data());
}

/**
 * @brief max|a - b| normalized by max|b|, and @a den_out is that denominator.
 *
 * The denominator is returned, not swallowed, because a zero one makes this
 * metric report a perfect 0.00e+00 for a comparison of two all-zero buffers.
 * The caller asserts it is non-zero, so "matched exactly" cannot be a way of
 * saying "computed nothing".
 */
double max_rel(const std::vector<float> &a, const std::vector<float> &b,
               double *den_out) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    num = std::max(num, std::fabs((double)a[i] - (double)b[i]));
    den = std::max(den, std::fabs((double)b[i]));
  }
  *den_out = den;
  return (den > 0.0) ? num / den : num;
}

double tol_for(hexkl_w_width w_k) {
  /* Task 3's out tolerances, applied to the stage that feeds them. S is where
   * K's width shows up, so the K half of the pair is what selects. */
  return (w_k == HEXKL_W_I8) ? 5e-3 : 5e-2;
}

} // namespace

TEST_F(HvxAttnScores, MatchesHostModelOverTheShapeMatrix) {
  const hexkl_w_width widths[4][2] = {{HEXKL_W_I8, HEXKL_W_I8},
                                      {HEXKL_W_I4, HEXKL_W_I4},
                                      {HEXKL_W_I8, HEXKL_W_I4},
                                      {HEXKL_W_I4, HEXKL_W_I8}};
  double worst = 0.0;
  double weakest_ref = 0.0; /* smallest S dynamic range any case produced */
  std::string worst_where;
  size_t cases = 0;

  std::cout << "\nS band vs host model -- max|dev - host| / max|host|\n"
            << "shape (kv,nq,gqa,nch,hd,T)     w_k,w_v   max_rel_err\n";

  for (uint32_t kv : {32u, 33u, 256u, 257u, 1024u}) {
    for (uint32_t nq : {1u, 33u, 128u}) {
      if (nq > kv) {
        continue; /* kv_len is kv_from + n_query; more queries than positions
                     is not representable */
      }
      for (uint32_t gqa : {1u, 8u}) {
        for (uint32_t head_dim : {64u, 128u}) {
          for (uint32_t T : {64u, 256u}) {
            for (int w = 0; w < 4; ++w) {
              const AttnCfg c{kv,       nq, gqa,          1u,
                              head_dim, T,  widths[w][0], widths[w][1]};
              std::vector<float> dev, ref;
              run_case(handle_, c, 0u, &dev, &ref);
              if (::testing::Test::HasFatalFailure()) {
                return;
              }
              double den = 0.0;
              const double e = max_rel(dev, ref, &den);
              /* A case whose reference S is all zeros would report a perfect
               * match while proving nothing. */
              ASSERT_GT(den, 0.0)
                << "reference S is identically zero -- the comparison is "
                << "vacuous for kv=" << kv << " nq=" << nq << " T=" << T;
              if (cases == 0 || den < weakest_ref) {
                weakest_ref = den;
              }

              std::ostringstream shape;
              shape << kv << "," << nq << "," << gqa << ",1," << head_dim << ","
                    << T;
              if (e > worst) {
                worst = e;
                worst_where =
                  shape.str() + " " + wname(c.w_k) + "," + wname(c.w_v);
              }
              if (w == 0 || e > 1e-6) {
                std::cout << std::left << std::setw(30) << shape.str()
                          << std::setw(10)
                          << (std::string(wname(c.w_k)) + "," + wname(c.w_v))
                          << std::scientific << std::setprecision(2) << e
                          << "\n";
              }
              EXPECT_LE(e, tol_for(c.w_k))
                << shape.str() << " (" << wname(c.w_k) << "," << wname(c.w_v)
                << ")";
              ++cases;
            }
          }
        }
      }
    }
  }

  std::cout << "ATTN_FIELD path=scores field=cases value=" << cases
            << std::endl;
  std::cout << "ATTN_FIELD path=scores field=max_rel_err value=" << worst
            << std::endl;
  std::cout << "ATTN_FIELD path=scores field=worst_shape value=" << worst_where
            << std::endl;
  /* Publish the weakest denominator so a future 0.00e+00 can be read as
     "bit-identical" rather than "both sides were empty". */
  std::cout << "ATTN_FIELD path=scores field=min_ref_dynamic_range value="
            << weakest_ref << std::endl;
}

/**
 * @brief V's width must not reach S at all.
 *
 * S = Q.Kt never touches V, so the two runs must be BITWISE identical. A
 * difference here is a plumbing bug -- the wrong registry or the wrong ops
 * being handed to the score call -- not numerical noise, which is why this is
 * bit-exact rather than a tolerance.
 */
TEST_F(HvxAttnScores, VWidthDoesNotReachS) {
  for (uint32_t kv : {33u, 257u}) {
    for (hexkl_w_width w_k : {HEXKL_W_I8, HEXKL_W_I4}) {
      const AttnCfg a{kv, 33u, 8u, 1u, 128u, 64u, w_k, HEXKL_W_I8};
      const AttnCfg b{kv, 33u, 8u, 1u, 128u, 64u, w_k, HEXKL_W_I4};
      std::vector<float> da, ra, db, rb;

      run_case(handle_, a, 0u, &da, &ra);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }
      run_case(handle_, b, 0u, &db, &rb);
      if (::testing::Test::HasFatalFailure()) {
        return;
      }

      ASSERT_EQ(da.size(), db.size());
      for (size_t i = 0; i < da.size(); ++i) {
        ASSERT_EQ(std::memcmp(&da[i], &db[i], sizeof(float)), 0)
          << "kv=" << kv << " w_k=" << wname(w_k) << " i=" << i
          << " w_v=I8 gave " << da[i] << ", w_v=I4 gave " << db[i];
      }
    }
  }
  std::cout << "ATTN_FIELD path=scores field=v_width_invariant value=1"
            << std::endl;
}

/**
 * @brief A released and re-registered layer reproduces its own result.
 *
 * Catches a registry slot that is not really reset -- stale WH bytes or a
 * handle that outlives its release would show up as a difference on the second
 * pass, and only there.
 */
TEST_F(HvxAttnScores, LifecycleIsRepeatable) {
  for (hexkl_w_width w : {HEXKL_W_I8, HEXKL_W_I4}) {
    /* kv 257 with T 64 crosses several block boundaries and leaves a partial
     * tail block, which is where a stale registration would survive. */
    const AttnCfg c{257u, 33u, 8u, 1u, 128u, 64u, w, w};
    std::vector<float> first, ref1, second, ref2;

    run_case(handle_, c, 0u, &first, &ref1);
    if (::testing::Test::HasFatalFailure()) {
      return;
    }
    run_case(handle_, c, 0u, &second, &ref2);
    if (::testing::Test::HasFatalFailure()) {
      return;
    }

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i) {
      ASSERT_EQ(std::memcmp(&first[i], &second[i], sizeof(float)), 0)
        << "width=" << wname(w) << " i=" << i;
    }
  }
  std::cout << "ATTN_FIELD path=scores field=lifecycle_repeatable value=1"
            << std::endl;
}

TEST_F(HvxAttnScores, RejectsBadShapes) {
  uint32_t h = 0;
  /* T must be a multiple of 32: hexkl_mm_u8iX_plan enforces N % 32 == 0. */
  EXPECT_EQ(nntr_hvx_attn_register(handle_, 1u, 1u, 128u, 64u, 48u,
                                   (uint32_t)HEXKL_W_I8, (uint32_t)HEXKL_W_I8,
                                   &h),
            AEE_EBADPARM + kDspOffset)
    << "T not a multiple of 32 must be rejected";
  EXPECT_EQ(nntr_hvx_attn_register(handle_, 1u, 1u, 128u, 64u, 64u, 5u,
                                   (uint32_t)HEXKL_W_I8, &h),
            AEE_EBADPARM + kDspOffset)
    << "width other than 4 or 8 must be rejected";
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
