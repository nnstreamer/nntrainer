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
#include <fp16.h>
#include <immintrin.h>
#include <limits>
#if __has_include(<numbers>)
#include <numbers>
#endif
#include <type_traits>
#if __has_include(<version>)
#include <version>
#endif
#include <fallback_internal.h>
#include <util_func.h>
#include <vector>

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
      *Z = *X * alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
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
      *Z = *X + alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
      X += o_stride;
      Y += i_stride;
      Z += o_stride;
    }
  }
}

static inline __m256 exp256_ps(__m256 x) {
  /*  Low-Precision Version I*/
  // const __m256 c1 = _mm256_set1_ps(12102203.0f);
  // const __m256 c2 = _mm256_set1_ps(1065353216.0f);
  // __m256 fx = _mm256_add_ps(_mm256_mul_ps(x, c1),c2);
  // return _mm256_castsi256_ps(_mm256_cvtps_epi32(fx));

  /* Low-Precision Version II*/
  /*    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.44269504089f); // 1 / ln(2)

    // Range reduction: x = n * ln2 + r,  where n is integer and |r| <= ln2/2
    __m256 fx = _mm256_mul_ps(x, inv_ln2);
    fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i emm0 = _mm256_cvtps_epi32(fx);

    __m256 tmp = _mm256_mul_ps(fx, ln2);
    __m256 r = _mm256_sub_ps(x, tmp);

    // Compute polynomial approximation of exp(r)
    const __m256 c1 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 c2 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 c3 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 c4 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 c5 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 c6 = _mm256_set1_ps(5.0000001201E-1f);

  //  __m256 r2 = _mm256_mul_ps(r, r);
  //  __m256 r3 = _mm256_mul_ps(r2, r);
  //  __m256 r4 = _mm256_mul_ps(r2, r2);

    __m256 y = _mm256_fmadd_ps(c1, r, c2);
    y = _mm256_fmadd_ps(y, r, c3);
    y = _mm256_fmadd_ps(y, r, c4);
    y = _mm256_fmadd_ps(y, r, c5);
    y = _mm256_fmadd_ps(y, r, c6);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));

    // Reconstruct exp(x) = 2^n * exp(r)
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    return _mm256_mul_ps(y, pow2n);
  */
  /* Low-Precision Versino III */
  const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f); // 1 / ln(2)
  const __m256 LN2 = _mm256_set1_ps(0.6931471805599453f);     // ln(2)

  // Clamp input to range to prevent overflow/underflow
  const __m256 max_x = _mm256_set1_ps(88.3762626647949f);  // log(FLT_MAX)
  const __m256 min_x = _mm256_set1_ps(-88.3762626647949f); // log(FLT_MIN)
  x = _mm256_max_ps(min_x, _mm256_min_ps(max_x, x));

  // Range reduction: x = n * ln2 + r
  __m256 fx = _mm256_mul_ps(x, LOG2EF); // x * (1/ln(2))
  fx = _mm256_floor_ps(_mm256_add_ps(fx, _mm256_set1_ps(0.5f)));

  __m256 tmp = _mm256_mul_ps(fx, LN2); // n * ln(2)
  __m256 r = _mm256_sub_ps(x, tmp);    // r = x - n * ln2

  // Compute exp(r) using 10th-order polynomial (Horner's method)
  const __m256 c0 = _mm256_set1_ps(1.0f);
  const __m256 c1 = _mm256_set1_ps(1.0f);
  const __m256 c2 = _mm256_set1_ps(0.5f);
  const __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
  const __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
  const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);
  const __m256 c6 = _mm256_set1_ps(1.0f / 720.0f);
  const __m256 c7 = _mm256_set1_ps(1.0f / 5040.0f);
  const __m256 c8 = _mm256_set1_ps(1.0f / 40320.0f);
  const __m256 c9 = _mm256_set1_ps(1.0f / 362880.0f);
  const __m256 c10 = _mm256_set1_ps(1.0f / 3628800.0f);

  __m256 y = c10;
  y = _mm256_fmadd_ps(y, r, c9);
  y = _mm256_fmadd_ps(y, r, c8);
  y = _mm256_fmadd_ps(y, r, c7);
  y = _mm256_fmadd_ps(y, r, c6);
  y = _mm256_fmadd_ps(y, r, c5);
  y = _mm256_fmadd_ps(y, r, c4);
  y = _mm256_fmadd_ps(y, r, c3);
  y = _mm256_fmadd_ps(y, r, c2);
  y = _mm256_fmadd_ps(y, r, c1);
  y = _mm256_fmadd_ps(y, r, c0); // final y = (((...r+...)*r+...)*r + 1)

  // Reconstruct exp(x) = 2^n * exp(r)
  __m256i emm0 = _mm256_cvtps_epi32(fx);
  emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
  emm0 = _mm256_slli_epi32(emm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(emm0);

  return _mm256_mul_ps(y, pow2n);
}

void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads) {
  size_t row_range = end_row - start_row;
  const size_t full_blocks = (num_heads / 8) * 8;
  // const size_t remainder = num_heads % 8;

  float *max_vals = new float[num_heads];
  float *sum_vals = new float[num_heads];
  // 1. max
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r)
      max_val = std::max(max_val, qk_out[r * num_heads + c]);
    max_vals[c] = max_val;
  }

  // 2. inplace exp + sum
  for (size_t c = 0; c < full_blocks; c += 8) {
    __m256 maxv = _mm256_loadu_ps(&max_vals[c]);
    __m256 sum = _mm256_setzero_ps();
    for (size_t r = 0; r < row_range; ++r) {
      float *ptr = &qk_out[(start_row + r) * num_heads + c];
      __m256 val = _mm256_loadu_ps(ptr);
      __m256 e = exp256_ps(_mm256_sub_ps(val, maxv));
      _mm256_storeu_ps(ptr, e); // overwrite qk_out
      sum = _mm256_add_ps(sum, e);
    }
    _mm256_storeu_ps(&sum_vals[c], sum);
  }

  for (size_t c = full_blocks; c < num_heads; ++c) {
    float sum = 0.0f;
    float maxv = max_vals[c];
    for (size_t r = 0; r < row_range; ++r) {
      float &a = qk_out[(start_row + r) * num_heads + c];
      a = std::exp(a - maxv); // overwrite qk_out
      sum += a;
    }
    sum_vals[c] = sum;
  }
  // 3. softmax = exp / sum (inplace)
  for (size_t r = 0; r < row_range; ++r) {
    for (size_t c = 0; c < full_blocks; c += 8) {
      float *ptr = &qk_out[(start_row + r) * num_heads + c];
      __m256 val = _mm256_loadu_ps(ptr); // already exp(x - max)
      __m256 sumv = _mm256_loadu_ps(&sum_vals[c]);
      __m256 soft = _mm256_div_ps(val, sumv);
      _mm256_storeu_ps(ptr, soft);
    }
    for (size_t c = full_blocks; c < num_heads; ++c) {
      qk_out[(start_row + r) * num_heads + c] /= sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads) {
  const size_t full_block = (num_heads / 8) * 8;

  float *max_vals = new float[num_heads];
  float *sum_vals = new float[num_heads];

  // 1. Find Max along with col
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r) {
      max_val = std::max(max_val, qk_out[r * num_heads + c]);
    }
    max_vals[c] = max_val;
  }

  // 2. Compute sum along with col (exp vectorized)
  for (size_t c = 0; c < full_block; c += 8) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t r = start_row; r < end_row; ++r) {
      __m256 val = _mm256_loadu_ps(&qk_out[r * num_heads + c]);
      __m256 maxv = _mm256_loadu_ps(&max_vals[c]);
      __m256 sub = _mm256_sub_ps(val, maxv);
      __m256 e = exp256_ps(sub);
      sum = _mm256_add_ps(sum, e);
    }
    _mm256_storeu_ps(&sum_vals[c], sum);
  }

  for (size_t c = full_block; c < num_heads; ++c) {
    float sum = 0.0f;
    for (size_t r = start_row; r < end_row; ++r) {
      sum += std::exp(qk_out[r * num_heads + c] - max_vals[c]);
    }
    sum_vals[c] = sum;
  }

  // 3. apply softmax
  for (size_t r = start_row; r < end_row; ++r) {
    for (size_t c = 0; c < full_block; c += 8) {
      __m256 val = _mm256_loadu_ps(&qk_out[r * num_heads + c]);
      __m256 maxv = _mm256_loadu_ps(&max_vals[c]);
      __m256 sub = _mm256_sub_ps(val, maxv);
      __m256 e = exp256_ps(sub);
      __m256 sumv = _mm256_loadu_ps(&sum_vals[c]);
      __m256 softmax = _mm256_div_ps(e, sumv);
      _mm256_storeu_ps(&qk_out[r * num_heads + c], softmax);
    }
    for (size_t c = full_block; c < num_heads; ++c) {
      qk_out[r * num_heads + c] =
        std::exp(qk_out[r * num_heads + c] - max_vals[c]) / sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

static inline float convert_scalar_f16_to_f32(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

void compute_fp16vcache_fp32_transposed(int iter, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int seq, int num_cache_head,
                                        int gqa_size, int head_dim,
                                        bool process_all) {
#if defined(__TIZEN__) && !defined(__F16C__)
  __fallback_compute_fp16vcache_fp32_transposed(iter, in, vcache, output, seq,
                                                num_cache_head, gqa_size,
                                                head_dim, process_all);
#else
  // cpu_set_t cpu_set;
  // CPU_ZERO(&cpu_set);
  // std::vector<bool> affinity(8, false);
  // affinity[i % affinity.size()] = true;

  // for (std::size_t j = 0;
  //      j < std::min<std::size_t>(affinity.size(), CPU_SETSIZE); ++j) {
  //   if (affinity[j])
  //     CPU_SET(j, &cpu_set);
  // }
  // pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);

  std::vector<float> tmp_fp32(head_dim);
  int a_row_start =
    process_all ? ((iter * (iter + 1)) / 2) * num_cache_head * gqa_size : 0;
  int out_offset = process_all ? iter : 0;

  int num_blocks = head_dim / 8;
  __m256 *sumVec = new __m256[std::max(1, num_blocks * gqa_size)];

  for (int n = 0; n < num_cache_head; ++n) {
    int rem = head_dim % 8;

    /* Declaration: std::vector<__m256> sumVec(num_blocks * gqa_size,
     * _mm256_setzero_ps()); caused warning: ignoring attributes on template
     * argument ‘__m256’ [-Wignored-attributes].
     * So it is implemented that way.
     */
    for (int i = 0; i < num_blocks * gqa_size; i++) {
      sumVec[i] = _mm256_setzero_ps();
    }
    std::vector<float> sumRem((size_t)gqa_size * rem, 0.0f);

    for (int j = 0; j <= iter; ++j) {
      if (j + 1 < seq) {
        const uint16_t *next_vptr =
          vcache + ((j + 1) * num_cache_head + n) * head_dim;
        _mm_prefetch(reinterpret_cast<const char *>(next_vptr), _MM_HINT_T0);
      }

      const uint16_t *vptr = vcache + (j * num_cache_head + n) * head_dim;

      int d0 = 0;
      for (; d0 + 8 <= head_dim; d0 += 8) {
        __m128i half =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(vptr + d0));
        __m256 f32 = _mm256_cvtph_ps(half);
        _mm256_storeu_ps(&tmp_fp32[d0], f32);
      }
      for (; d0 < head_dim; ++d0) {
        tmp_fp32[d0] = convert_scalar_f16_to_f32(vptr[d0]);
      }

      for (int h = 0; h < gqa_size; ++h) {
        // float a_val = in[a_row_start + (j * gqa_size + h) * num_cache_head +
        // n];
        float a_val =
          in[a_row_start + (j * gqa_size * num_cache_head + n * gqa_size + h)];
        __m256 inVec = _mm256_set1_ps(a_val);

        for (int b = 0; b < num_blocks; ++b) {
          __m256 bVec = _mm256_loadu_ps(&tmp_fp32[b * 8]);
          sumVec[h * num_blocks + b] =
            _mm256_fmadd_ps(inVec, bVec, sumVec[h * num_blocks + b]);
        }

        float *remPtr = &sumRem.data()[h * rem];
        int base = num_blocks * 8;
        for (int r = 0; r < rem; ++r) {
          remPtr[r] += a_val * tmp_fp32[base + r];
        }
      }
    }

    for (int h = 0; h < gqa_size; ++h) {
      for (int b = 0; b < num_blocks; ++b) {
        int out_base =
          ((out_offset * num_cache_head + n) * gqa_size + h) * head_dim + b * 8;
        _mm256_storeu_ps(&output[out_base], sumVec[h * num_blocks + b]);
      }

      float *remPtr = &sumRem.data()[h * rem];
      // float *remPtr = &sumRem[h * rem];
      int base = num_blocks * 8;
      for (int r = 0; r < rem; ++r) {
        int out_idx =
          ((out_offset * num_cache_head + n) * gqa_size + h) * head_dim + base +
          r;
        output[out_idx] = remPtr[r];
      }
    }
  }
  delete[] sumVec;
#endif
}

#if !defined(__TIZEN__) || defined(__F16C__)
static inline __m256 load_fp16_8_avx(const uint16_t *src) {
  __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  return _mm256_cvtph_ps(in);
}

static inline void load_fp16_8_to_chunk(const uint16_t *b_src, float *temp_row,
                                        int chunk_size) {
  int i = 0;
  for (; i + 8 <= chunk_size; i += 8) {
    __m256 f32 = load_fp16_8_avx((const uint16_t *)(b_src + i));
    _mm256_storeu_ps(temp_row + i, f32);
  }
  for (; i < chunk_size; ++i) {
    temp_row[i] = convert_scalar_f16_to_f32(b_src[i]);
  }
}
#endif

template <>
void compute_kcaches(const float *A, const uint16_t *B, float *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
#if defined(__TIZEN__) && !defined(__F16C__)
  __fallback_compute_kcaches<uint16_t>(A, B, output, num_rows, N, chunk_size,
                                       group_size, tile_size);
#else
  using BType = uint16_t;
  int row_stride = N * chunk_size;
  const int group_stride = group_size * chunk_size;
  const int tile_count = (num_rows + tile_size - 1) / tile_size;

  // FP32 Cache Buffer
  thread_local std::vector<float> temp_tile_buf((size_t)tile_size * chunk_size);

  for (int n = 0; n < N; ++n) {
    for (int t = 0; t < tile_count; ++t) {
      int row_tile_start = t * tile_size;
      int tile_rows = std::min(tile_size, num_rows - row_tile_start);

      // FP16 to FP32 Conversion : preprocessing (row unit)
      if constexpr (!std::is_same<BType, float>::value) {
        for (int row = 0; row < tile_rows; ++row) {
          const BType *b_src =
            B + (row_tile_start + row) * row_stride + n * chunk_size;
          float *dst = temp_tile_buf.data() + row * chunk_size;
          load_fp16_8_to_chunk(b_src, dst, chunk_size);
        }
      }

      for (int g = 0; g < group_size; ++g) {
        const float *a_ptr = A + n * group_stride + g * chunk_size;
        for (int row = 0; row < tile_rows; ++row) {
          const float *b_row;
          if constexpr (std::is_same<BType, float>::value) {
            b_row = reinterpret_cast<const float *>(
              B + (row_tile_start + row) * row_stride + n * chunk_size);
          } else {
            b_row = temp_tile_buf.data() + row * chunk_size;
          }

          float sum = 0.0f;
          int i = 0;
          __m256 acc = _mm256_setzero_ps();
          for (; i + 8 <= chunk_size; i += 8) {
            __m256 va = _mm256_loadu_ps(a_ptr + i);
            __m256 vb = _mm256_loadu_ps(b_row + i);
            acc = _mm256_fmadd_ps(va, vb, acc);
          }

          __m128 low = _mm256_castps256_ps128(acc);
          __m128 high = _mm256_extractf128_ps(acc, 1);
          __m128 sum128 = _mm_add_ps(low, high);
          sum128 = _mm_hadd_ps(sum128, sum128);
          sum128 = _mm_hadd_ps(sum128, sum128);
          sum += _mm_cvtss_f32(sum128);

          for (; i < chunk_size; ++i)
            sum += a_ptr[i] * b_row[i];

          output[(row_tile_start + row) * N * group_size + n * group_size + g] =
            sum / sqrt((float)chunk_size);
        }
      }
    }
  }
#endif
}

#ifdef _WIN32
#define COMPUTE_FP16_TO_FP32(x)                                                \
  _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define COMPUTE_FP32_TO_FP16(x)                                                \
  _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
#if defined(__TIZEN__) && !defined(__F16C__)
  __fallback_compute_rotary_emb_value(width, dim, half_, inout, output, cos_,
                                      sin_, only_convert_to_fp16);
#else
  enum class OutputType { FP16, FP32 };

  OutputType out_type = OutputType::FP32;
  if (output != nullptr)
    out_type = OutputType::FP16;

  for (unsigned int w = 0; w < width; w += dim) {
    unsigned int k = 0;
    for (; k + 7 < half_; k += 8) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;

      __m256 a = _mm256_loadu_ps(&inout[i0]);
      __m256 b = _mm256_loadu_ps(&inout[i1]);

      if (only_convert_to_fp16) {
        if (out_type == OutputType::FP16) {
          __m128i a_fp16 = _mm256_cvtps_ph(a, 0);
          __m128i b_fp16 = _mm256_cvtps_ph(b, 0);

          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i0),
            a_fp16);
          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i1),
            b_fp16);
        }

      } else {
        __m256 cos_v = _mm256_loadu_ps(&cos_[k]);
        __m256 sin_v = _mm256_loadu_ps(&sin_[k]);

        __m256 out0 =
          _mm256_sub_ps(_mm256_mul_ps(a, cos_v), _mm256_mul_ps(b, sin_v));
        __m256 out1 =
          _mm256_add_ps(_mm256_mul_ps(a, sin_v), _mm256_mul_ps(b, cos_v));

        if (out_type == OutputType::FP16) {
          __m128i out0_fp16 = _mm256_cvtps_ph(out0, 0);
          __m128i out1_fp16 = _mm256_cvtps_ph(out1, 0);

          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i0),
            out0_fp16);
          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i1),
            out1_fp16);

        } else if (out_type == OutputType::FP32) {
          _mm256_storeu_ps(&inout[i0], out0);
          _mm256_storeu_ps(&inout[i1], out1);
        }
      }
    }

    for (; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      // assert(i1 < width && "Scalar i1 overflow!");
      float a = inout[i0];
      float b = inout[i1];

      if (only_convert_to_fp16) {
        static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(a);
        static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(b);

      } else {

        float c = cos_[k];
        float s = sin_[k];

        float out0 = a * c - b * s;
        float out1 = a * s + b * c;

        if (out_type == OutputType::FP16) {
          static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(out0);
          static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(out1);
        } else if (out_type == OutputType::FP32) {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
  }
#endif
}

} // namespace nntrainer::avx2
