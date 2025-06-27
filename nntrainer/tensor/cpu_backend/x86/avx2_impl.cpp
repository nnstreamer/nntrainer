// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_impl.cpp
 * @date   20 Feb 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source for AVX implementation
 *
 */

#include "avx2_impl.h"
#include <array>
#if __has_include(<bit>)
#include <bit>
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#if __has_include(<numbers>)
#include <numbers>
#endif
#include <type_traits>
#if __has_include(<version>)
#include <version>
#endif
#include <util_func.h>

#if !defined(__has_constexpr_builtin)
#define __has_constexpr_builtin(x) (0)
#endif

#if !defined(__has_cpp_attribute)
#define __has_cpp_attribute(x) (0)
#endif

// VECTORCALL calling-conv (default for x86_64-linux-gnu)
#if _MSC_VER >= 1700
#define _nnt_CC_VECTORCALL __vectorcall
#else
#define _nnt_CC_VECTORCALL
#endif

// Flatten attribute
#if _MSC_VER >= 1700 || __has_cpp_attribute(msvc::flatten)
#define _nnt_ATTR_FLATTEN [[msvc::flatten]]
#elif __has_cpp_attribute(gnu::flatten)
// clang, g++
#define _nnt_ATTR_FLATTEN [[gnu::flatten]]
#else
#define _nnt_ATTR_FLATTEN
#endif

#if _MSC_VER >= 1700 || __has_cpp_attribute(msvc::noinline)
#define _nnt_ATTR_NOINLINE [[msvc::noinline]]
#elif __has_cpp_attribute(gnu::flatten)
// clang, g++
#define _nnt_ATTR_NOINLINE [[gnu::noinline]]
#else
#define _nnt_ATTR_NOINLINE
#endif

#if _MSC_VER >= 1700 || __has_cpp_attribute(msvc::forceinline)
#define _nnt_ATTR_ALWAYS_INLINE [[msvc::forceinline]]
#elif __has_cpp_attribute(gnu::always_inline)
#define _nnt_ATTR_ALWAYS_INLINE [[gnu::always_inline]]
#endif

#if __has_cpp_attribute(unikely)
#define UNLIKELY [[unlikely]]
#else
#define UNLIKELY
#endif

#if !defined(_MSC_VER) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wattributes"
#endif

namespace {

template <typename To_, typename From_>
constexpr inline bool concept17_BinaryCastable =
  sizeof(To_) == sizeof(From_) &&
  std::is_trivially_copyable_v<From_> &&std::is_trivially_copyable_v<To_>;

template <class To_, class From_>
auto compat_bit_cast(const From_ &src) noexcept
  -> std::enable_if_t<concept17_BinaryCastable<To_, From_>, To_> {
#if __cpp_lib_bit_cast >= 201806L
  return std::bit_cast<To_>(src);
#else
  To_ dst;
  std::memcpy(&dst, &src, sizeof(To_));
  return dst;
#endif
}

[[nodiscard]] constexpr inline unsigned
constexpr_popcount(uint32_t v) noexcept {
#if __cpp_lib_bitops >= 201907L
  return std::popcount(v);
#else
  // Popcount via bit-hack
  v = v - ((v >> 1) & 0x55555555);                // reuse input as temporary
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333); // temp
  auto c = (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24; // count
  return c;
#endif
}

template <unsigned I_>
constexpr inline bool concept17_PowerOfTwo = (constexpr_popcount(I_) == 1);

namespace numbers {

#if __has_include(<numbers>) && __cpp_lib_math_constants >= 201907L
using std::numbers::ln2_v;
using std::numbers::log2e_v;
#else
template <typename Float_> constexpr inline auto ln2_v = Float_{M_LN2};

template <typename Float_> constexpr inline auto log2e_v = Float_{M_LOG2E};
#endif
} // namespace numbers

constexpr inline float EXP_ARG_MIN = -87.0;
constexpr inline float EXP_ARG_MAX = +88.3762626647949f;

// @brief Precalculated lookup table for 2^x calculation
template <unsigned N_, typename Ty_ = uint32_t, typename Float_ = float,
          typename = std::enable_if_t<concept17_PowerOfTwo<N_>>>
struct Exp2Table {

  constexpr static inline auto MANTISSA_BITS =
    std::numeric_limits<Float_>::digits - 1;

#if __cpp_consteval >= 201811L && __has_constexpr_builtin(__builtin_exp2)
  [[nodiscard]] static consteval auto calculate() noexcept {
    std::array<Ty_, N_> t;
    for (unsigned i = 0; i < N_; ++i)
      t[i] = std::bit_cast<Ty_>(std::exp2(Float_{1.0} * i / N_)) -
             ((i << MANTISSA_BITS) / N_);
    return t;
  }
#endif
};

#if !__has_constexpr_builtin(__builtin_exp2) || !(__cpp_consteval >= 201811L)

// @brief Precalculated lookup table for 2^x calculation when we don't have
// constexpr math functions
template <> struct Exp2Table<8, uint32_t, float> {
  [[nodiscard]] static constexpr auto calculate() noexcept {
    std::array<uint32_t, 8> t = {0x3f800000U, 0x3f7b95c2U, 0x3f7837f0U,
                                 0x3f75fed7U, 0x3f7504f3U, 0x3f75672aU,
                                 0x3f7744fdU, 0x3f7ac0c7U};
    return t;
  }
};

#endif

template <unsigned N_> // requires PowerOfTwo<N_>
alignas(__m256) inline constexpr auto exp2_table_v = Exp2Table<N_>::calculate();

// Scalar version of expf() approximation with 3-th deg polynominal of
// fractional part
//
// The error with regards to std::expf less than 5e-6
// It is valid in range [-87, +88.37) - not handling +INF, NaN etc.
// The function domain is clamped to valid function range.
//
// The strategy picked is to approximate exp as 2^K*2^F
template <unsigned N_, typename = std::enable_if_t<concept17_PowerOfTwo<N_>>>
[[nodiscard]] _nnt_ATTR_ALWAYS_INLINE _nnt_ATTR_FLATTEN inline float
approx_exp_exp2_lookup(float x) noexcept {
  constexpr static unsigned N_MASK = uint32_t(N_ - 1U);
  constexpr static unsigned FLT_MANTISSA_BITS =
    std::numeric_limits<float>::digits - 1U;

  x = std::max(x, EXP_ARG_MIN);
  x *= float(0x1.0p1 / numbers::ln2_v<double> * N_);
  x = std::min(x, float(EXP_ARG_MAX / numbers::ln2_v<double> * N_));

  // Round nearest and convert integer part to an int (std::modf)
  // NB: This way doesn't handle ties even.
  auto x_int = x + 0x1.8p23f;
  auto x_uint = compat_bit_cast<uint32_t>(x_int);
  x_int -= 0x1.8p23f;
  auto x_frac = x - x_int;

  auto s_int = exp2_table_v<N_>[x_uint & N_MASK];
  auto x_uint_shifted = x_uint
                        << (FLT_MANTISSA_BITS - constexpr_popcount(N_MASK));
  auto s_int_2 = s_int + x_uint_shifted;
  auto s = compat_bit_cast<float>(s_int_2);

  // Polynominal of form C0*x^3 + C1*x^2 + C2*x^1 + 1.0
  static constexpr float poly_4[] = {0x1.c6af84b912394p-5 / N_ / N_ / N_,
                                     0x1.ebfce50fac4f3p-3 / N_ / N_,
                                     0x1.62e42ff0c52d6p-1 / N_};

  auto q0 = std::fma(x_frac, poly_4[0], poly_4[1]);
  auto x_frac_pow2 = x_frac * x_frac;
  auto q2 = x_frac * poly_4[2]; // not adding +1.0

  x = std::fma(q0, x_frac_pow2, q2);

  return std::fma(x, s, s); // NB: (handles (x+1) by addition of s)
}

// Vectorized version of above
template <unsigned N_, typename = std::enable_if_t<concept17_PowerOfTwo<N_>>>
_nnt_ATTR_ALWAYS_INLINE _nnt_ATTR_FLATTEN inline auto _nnt_CC_VECTORCALL
avx2_approx_exp_e2lookup(__m256 xs) noexcept {

  constexpr static uint32_t N_MASK = uint32_t(N_ - 1U);
  alignas(64) constexpr static auto EXP2_TBL = exp2_table_v<N_>;
  constexpr static unsigned MANTISSA_BITS =
    std::numeric_limits<float>::digits - 1;

  // Ensure arg in range [exp_arg_min, exp_arg_max]
  xs = _mm256_max_ps(xs, _mm256_set1_ps(EXP_ARG_MIN));
  // Would clamp to EXP_ARG_MAX but we move it after multiplication for IPC:
  // xs = _mm256_min_ps(xs, _mm256_set1_ps(EXP_ARG_MAX));

  xs =
    _mm256_mul_ps(xs, _mm256_set1_ps(float(1.0 / numbers::ln2_v<double> * N_)));
  // Clamp EXP_ARG_MAX after multiply
  xs = _mm256_min_ps(
    xs, _mm256_set1_ps(float(EXP_ARG_MAX / numbers::ln2_v<double> * N_)));

  // Mostly equivalent to, doesn't round ties to even
  // auto xs_int = _mm256_round_ps(xs, _MM_FROUND_TO_NEAREST_INT |
  // _MM_FROUND_NO_EXC); auto xs_int_as_u32 = _mm256_cvtps_epi32(xs_int);
  auto xs_int = _mm256_add_ps(xs, _mm256_set1_ps(0x1.8p23f));
  auto xs_int_as_u32 = _mm256_castps_si256(xs_int);
  xs_int = _mm256_sub_ps(xs_int, _mm256_set1_ps(0x1.8p23f));

  // Calculate fractional part
  auto xs_frac = _mm256_sub_ps(xs, xs_int);
  // Indices for lookup (modulo N_)
  auto exp2_idxs = _mm256_and_si256(xs_int_as_u32, _mm256_set1_epi32(N_MASK));

  __m256i s_ints;

  // Lookup e^xs_int s factor
  if constexpr (N_ == 8) {
    // Lookup by vector permute
    auto tbl = _mm256_load_si256((__m256i *)EXP2_TBL.data());
    s_ints = _mm256_permutevar8x32_epi32(tbl, exp2_idxs);
  } else {
    // Falback for not fitting number of vector elements
    s_ints = _mm256_i32gather_epi32(EXP2_TBL.data(), exp2_idxs, 1);
  }

  auto xs_uint_shifted = _mm256_slli_epi32(
    xs_int_as_u32, MANTISSA_BITS - constexpr_popcount(N_MASK));
  auto s_ints_2 = _mm256_add_epi32(s_ints, xs_uint_shifted);
  auto s_floats = _mm256_castsi256_ps(s_ints_2);

  static constexpr float poly_d4[] = {0x1.c6af84b912394p-5 / N_ / N_ / N_,
                                      0x1.ebfce50fac4f3p-3 / N_ / N_,
                                      0x1.62e42ff0c52d6p-1 / N_};

  const auto C0 = _mm256_set1_ps(poly_d4[0]);
  const auto C1 = _mm256_set1_ps(poly_d4[1]);
  const auto C2 = _mm256_set1_ps(poly_d4[2]);

  auto qs0 = _mm256_fmadd_ps(xs_frac, C0, C1);
  auto xs_frac_pow2 = _mm256_mul_ps(xs_frac, xs_frac);
  auto qs2 = _mm256_mul_ps(xs_frac, C2);

  xs = _mm256_fmadd_ps(qs0, xs_frac_pow2, qs2);

  return _mm256_fmadd_ps(xs, s_floats, s_floats);
}

_nnt_ATTR_ALWAYS_INLINE _nnt_ATTR_FLATTEN auto _nnt_CC_VECTORCALL
avx2_negate_ps(__m256 x) noexcept -> __m256 {
  constexpr auto SIGN_SHIFT = sizeof(float) * 8 - 1;
  const auto UNDEF = _mm256_undefined_si256();
  const auto sign_bit =
    _mm256_slli_epi32(_mm256_cmpeq_epi16(UNDEF, UNDEF), SIGN_SHIFT);
  auto flt_sign_bit = _mm256_castsi256_ps(sign_bit);
  auto neg_x = _mm256_xor_ps(x, flt_sign_bit);
  return neg_x;
}

_nnt_ATTR_ALWAYS_INLINE _nnt_ATTR_FLATTEN auto _nnt_CC_VECTORCALL
avx2_approx_swiglu(__m256 x, __m256 s) noexcept -> __m256 {
  auto neg_x = avx2_negate_ps(x);
  auto inv_sigmoid =
    _mm256_add_ps(avx2_approx_exp_e2lookup<8>(neg_x), _mm256_set1_ps(1.0f));
  auto swiglu_nonscaled = _mm256_div_ps(x, inv_sigmoid);
  return _mm256_mul_ps(swiglu_nonscaled, s);
}
} // namespace

namespace nntrainer::avx2 {

bool is_valid(const unsigned int N, const float *input) {
  assert(N != 0);
  assert(input != NULL);

  int temp = 0;
  unsigned int idx = 0;

  const __m256 SIGN_MASK = _mm256_set1_ps(-0.0);
  const __m256 INF = _mm256_set1_ps(std::numeric_limits<float>::infinity());

  // 16 single-precision check : ( X != X )
  for (; N - idx >= 16; idx += 16) {
    __m256 vec0 = _mm256_loadu_ps(input);
    __m256 vec1 = _mm256_loadu_ps(input + 8);
    input += 16;
    __m256 res = _mm256_cmp_ps(vec0, vec0, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return false;

    // check infinity in vec0
    vec0 = _mm256_andnot_ps(SIGN_MASK, vec0);
    vec0 = _mm256_cmp_ps(vec0, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec0);
    if (temp)
      return false;

    __m256 res1 = _mm256_cmp_ps(vec1, vec1, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res1);

    if (temp)
      return false;

    // check infinity in vec1
    vec1 = _mm256_andnot_ps(SIGN_MASK, vec1);
    vec1 = _mm256_cmp_ps(vec1, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec1);

    if (temp)
      return false;
  }

  // 8 single-precision check : ( X != X )
  for (; N - idx >= 8; idx += 8) {
    __m256 vec = _mm256_loadu_ps(input);
    input += 8;
    __m256 res = _mm256_cmp_ps(vec, vec, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return false;

    // check infinity in vec
    vec = _mm256_andnot_ps(SIGN_MASK, vec);
    vec = _mm256_cmp_ps(vec, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec);

    if (temp)
      return false;
  }

  while (idx < N) {
    if (!isFloatValid(*input)) {
      return false;
    }
    ++input;
    ++idx;
  }

  return true;
}

void custom_scopy(const unsigned int N, const float *X, const int incX,
                  float *Y, const int incY) {
  unsigned int N8 = (N >> 3) << 3;
  for (unsigned int i = 0; i < N8; i += 8) {
#if defined(_WIN32)
    __m256 temp = _mm256_loadu_ps(&X[i]);
    _mm256_storeu_ps(&Y[i], temp);
#else
    __asm__ __volatile__("vmovups (%1), %%ymm0\n\t"
                         "vmovups %%ymm0, (%0)\n\t"
                         :
                         : "r"(&Y[i]), "r"(&X[i])
                         : "ymm0", "memory");
#endif
  }
  for (unsigned int i = N8; i < N; ++i) {
    Y[i] = X[i];
  }
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst) {
  unsigned int vindexm[8] = {0,          ld_src,     ld_src * 2, ld_src * 3,
                             ld_src * 4, ld_src * 5, ld_src * 6, ld_src * 7};
  __m256i vindex = _mm256_loadu_si256((__m256i *)&vindexm[0]);
  __m256 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8;

  unsigned int M8 = (M & ~(7));
  unsigned int N8 = (N & ~(7));
  for (unsigned int i = 0; i < M8; i += 8) {
    for (unsigned int j = 0; j < N8; j += 8) {
      // loading from columns
      vec1 = _mm256_i32gather_ps(&src[ld_src * i + j + 0], vindex, 4);
      vec2 = _mm256_i32gather_ps(&src[ld_src * i + j + 1], vindex, 4);
      vec3 = _mm256_i32gather_ps(&src[ld_src * i + j + 2], vindex, 4);
      vec4 = _mm256_i32gather_ps(&src[ld_src * i + j + 3], vindex, 4);
      vec5 = _mm256_i32gather_ps(&src[ld_src * i + j + 4], vindex, 4);
      vec6 = _mm256_i32gather_ps(&src[ld_src * i + j + 5], vindex, 4);
      vec7 = _mm256_i32gather_ps(&src[ld_src * i + j + 6], vindex, 4);
      vec8 = _mm256_i32gather_ps(&src[ld_src * i + j + 7], vindex, 4);

      // storing to the rows
      _mm256_storeu_ps(&dst[(j + 0) * ld_dst + i], vec1);
      _mm256_storeu_ps(&dst[(j + 1) * ld_dst + i], vec2);
      _mm256_storeu_ps(&dst[(j + 2) * ld_dst + i], vec3);
      _mm256_storeu_ps(&dst[(j + 3) * ld_dst + i], vec4);
      _mm256_storeu_ps(&dst[(j + 4) * ld_dst + i], vec5);
      _mm256_storeu_ps(&dst[(j + 5) * ld_dst + i], vec6);
      _mm256_storeu_ps(&dst[(j + 6) * ld_dst + i], vec7);
      _mm256_storeu_ps(&dst[(j + 7) * ld_dst + i], vec8);
    }
  }

  // tailing right
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = N8; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }

  // tailing bottom
  for (unsigned int i = M8; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

void swiglu(const unsigned int N, float *X, const float *Y, const float *Z) {
  size_t i = 0;

  auto oldcsr = _mm_getcsr();
  // We don't need denormals, enable:
  // DAZ = Denormals Are Zero
  // FTZ = Flush To Zero
  _mm_setcsr(oldcsr | 0x8040);

  for (; i + 16 < N; i += 16) {
    auto y0 = _mm256_loadu_ps(Y + i);
    auto y1 = _mm256_loadu_ps(Y + i + 8);
    auto z0 = _mm256_loadu_ps(Z + i);
    auto z1 = _mm256_loadu_ps(Z + i + 8);

    _mm256_storeu_ps(X + i, avx2_approx_swiglu(y0, z0));
    _mm256_storeu_ps(X + i + 8, avx2_approx_swiglu(y1, z1));
  }

  if (i + 8 < N)
    UNLIKELY {
      auto y0 = _mm256_loadu_ps(Y + i);
      auto z0 = _mm256_loadu_ps(Z + i);
      _mm256_storeu_ps(X + i, avx2_approx_swiglu(y0, z0));
      i += 8;
    }

  if (i > N)
    UNLIKELY {
      alignas(64) int mask[] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                0,  0,  0,  0,  0,  0,  0,  0};

      auto vmask = _mm256_loadu_si256((__m256i *)(mask + (i & 7)));
      auto ym = _mm256_maskload_ps(Y + i, vmask);
      auto zm = _mm256_maskload_ps(Z + i, vmask);
      _mm256_maskstore_ps(X + i, vmask, avx2_approx_swiglu(ym, zm));
    }
  _mm_setcsr(oldcsr);
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int N8 = (N & ~(7));
    if (i_stride == 0) {
      float vy8[8] = {Y[0], Y[0], Y[0], Y[0], Y[0], Y[0], Y[0], Y[0]};
      auto y = _mm256_loadu_ps(&vy8[0]);
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto z = _mm256_mul_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += i_stride * 8;
        Z += 8;
      }
    } else {
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto y = _mm256_loadu_ps(Y);
        auto z = _mm256_mul_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += i_stride * 8;
        Z += 8;
      }
    }
    for (unsigned int i = N8; i < N; ++i) {
      *Z = *X * *Y;
      X++;
      Y += i_stride;
      Z++;
    }
  } else {
    // TODO: AVX2 implementation if used
    for (unsigned int i = 0; i < N; ++i) {
      *Z = *X * alpha * *Y + beta * *Z;
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int N8 = (N & ~(7));
    if (i_stride == 0) {
      float vy8[8] = {Y[0], Y[0], Y[0], Y[0], Y[0], Y[0], Y[0], Y[0]};
      auto y = _mm256_loadu_ps(&vy8[0]);
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto z = _mm256_add_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += i_stride * 8;
        Z += 8;
      }
    } else {
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto y = _mm256_loadu_ps(Y);
        auto z = _mm256_add_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += i_stride * 8;
        Z += 8;
      }
    }
    for (unsigned int i = N8; i < N; ++i) {
      *Z = *X + *Y;
      X++;
      Y += i_stride;
      Z++;
    }
  } else {
    // TODO: AVX2 implementation if used
    for (unsigned int i = 0; i < N; ++i) {
      *Z = *X + alpha * *Y + beta * *Z;
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

} // namespace nntrainer::avx2
