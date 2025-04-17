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
#include <cassert>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <numbers>
#include <type_traits>

namespace {

using std::integral_constant;
using std::bool_constant;

template <auto V>
inline constexpr static
   auto SizeConstant = integral_constant<size_t, V>{};

template <auto V>
   requires std::convertible_to<decltype(V), bool>
inline constexpr static
   auto BoolConstant = bool_constant<static_cast<bool>(V)>{};

template <typename>
inline constexpr bool is_size_constant_v = false;

template <auto V>
inline constexpr bool is_size_constant_v<integral_constant<size_t, V>> = true;

template <auto V>
using SizeConstantType = integral_constant<size_t, V>;


// Helper for loop unrolling
//   https://en.wikipedia.org/wiki/Loop_unrolling
//
// template argument N is number of reapeated loop body
// As first argument of each loop body lambda (or a function)
// There will be passed integral constant
template <unsigned N>
[[using gnu: always_inline, flatten]]
[[using msvc: forceinline, flatten]]
constexpr auto loop_unroll(auto && loop_body, auto && ... args) noexcept
{
   using std::index_sequence;
   using std::make_index_sequence;

   // the index_sequence<0, ... N-1>
   constexpr auto Indicies = make_index_sequence<N>{};

   return [body = static_cast<decltype(loop_body)&&>(loop_body)]
      <size_t... Is_>(index_sequence<Is_...>, auto && ... args)
   constexpr noexcept {
      // For each SizeConstant [0..N)
      // directly invoke loop body,
      // passing number of iteration as the first argument
      return (body(SizeConstant<Is_>, static_cast<decltype(args)&&>(args)...), ...);
      // equivalent to:
      //    body(SizeConstant<0>), body(SizeConstant<1>)...
      // etc.
   }(Indicies, static_cast<decltype(args)&&>(args)...);
}

constexpr inline float exp_arg_min = -87.0;
constexpr inline float exp_arg_max = +88.3762626647949f;

// remez approx of (2^x - 1) in range [0,1)
constexpr inline std::array<float, 8> exp_frac_corr_p7 = {
1.5509058e-5,
1.5669828e-4,
1.3331374e-3,
9.6164606e-3,
5.5504179e-2,
2.4022684e-1,
6.9314717e-1,
-1.0447757e-8
};
/* scalar:
template <auto Kf_Poly_>
float approx_exp_malossi_px(float x)
{
  static_assert(Kf_Poly_.size() >= 2);
  x = std::max(x, exp_arg_min);
  x = std::min(x, exp_arg_max);
  x *= std::numbers::log2e_v<float>;

  auto x_int = std::floor(x);
  auto x_frac = x - x_int;
  auto k_xf = std::fma(x_frac, Kf_Poly_[1], Kf_Poly_[0]);
  for (unsigned i=2; i<Kf_Poly_.size(); ++i) {
       k_xf = std::fma(x_frac, k_xf, Kf_Poly_[i]);
  }
  float k = k_xf + 1.0f; 

  auto k_as_u32 = std::bit_cast<uint32_t>(k);
  auto exp_as_u32 = k_as_u32 + (static_cast<uint32_t>(x_int) << 23);

  return std::bit_cast<float>(exp_as_u32);
}
*/

[[gnu::always_inline, msvc::forceinline]]
inline auto avx2_approx_exp_malossi_p7(__m256 xs) noexcept -> __m256
{
  using namespace std::numbers;
  // Ensure arg in range [exp_arg_min, exp_arg_max]
  xs = _mm256_max_ps(xs, _mm256_set1_ps(exp_arg_min));
  xs = _mm256_min_ps(xs, _mm256_set1_ps(exp_arg_max));

  // e^x *log2e = 2^x = 2^(x_i + x_f)
  xs = _mm256_mul_ps(xs, _mm256_set1_ps(log2e_v<float>));

  // 2^(x_int)
  auto xs_int = _mm256_floor_ps(xs);

  // 2^(x_frac)
  auto xs_frac = _mm256_sub_ps(xs, xs_int);

  // Horner scheme p8:
  // even elements in lower 128b, odd in upper
  const auto kf_poly_d7 = _mm256_setr_ps(exp_frac_corr_p7[0], exp_frac_corr_p7[2], exp_frac_corr_p7[4], exp_frac_corr_p7[6],
                                         exp_frac_corr_p7[1], exp_frac_corr_p7[3], exp_frac_corr_p7[5], exp_frac_corr_p7[7]);
  auto kf_curr_coef = kf_poly_d7;

  auto kf_curr_coef_e = _mm256_extractf128_ps(kf_curr_coef, 0);
  auto kf_curr_coef_o = _mm256_extractf128_ps(kf_curr_coef, 1);

  auto k_e = _mm256_broadcastss_ps(kf_curr_coef_e);
  auto k_o = _mm256_broadcastss_ps(kf_curr_coef_o);

  auto k_xsf = _mm256_fmadd_ps(xs_frac, k_o, k_e);

  loop_unroll<3>([&](auto _) constexpr noexcept
  {
    // Treat kf_curr_coef as double shift register for polynominal coefficients
    auto t = _mm256_srli_si256(_mm256_castps_si256(kf_curr_coef), sizeof(float));
    kf_curr_coef = _mm256_castsi256_ps(t);

    kf_curr_coef_e = _mm256_extractf128_ps(kf_curr_coef, 0);
    kf_curr_coef_o = _mm256_extractf128_ps(kf_curr_coef, 1);

    k_e = _mm256_broadcastss_ps(kf_curr_coef_e);
    k_o = _mm256_broadcastss_ps(kf_curr_coef_o);

    k_xsf = _mm256_fmadd_ps(xs_frac, k_xsf, k_e);
    k_xsf = _mm256_fmadd_ps(xs_frac, k_xsf, k_o);
  });

  k_xsf = _mm256_add_ps(k_xsf, _mm256_set1_ps(1.0f));

  auto ks_as_u32 = _mm256_castps_si256(k_xsf);
  auto xs_int_as_u32 = _mm256_cvtps_epi32(xs_int);
  
  xs_int_as_u32 = _mm256_slli_epi32(xs_int_as_u32, 23);
  auto exps_as_u32 = _mm256_add_epi32(xs_int_as_u32, ks_as_u32);

  return _mm256_castsi256_ps(exps_as_u32);
}

[[gnu::always_inline, msvc::forceinline]]
auto avx2_approx_swiglu(__m256 x, __m256 s) noexcept -> __m256
{
  auto neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-1.0f));
  auto inv_sigmoid = _mm256_add_ps(avx2_approx_exp_malossi_p7(neg_x), _mm256_set1_ps(1.0f));
  auto swiglu_nonscaled = _mm256_div_ps(x, inv_sigmoid);
  return _mm256_mul_ps(swiglu_nonscaled, s);
}

}

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

  // remain check : ( X != X )
  while (idx < N) {
    if (*input != *input || *input == std::numeric_limits<float>::infinity()) {
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

  for (; i+16 < N; i += 16)
  {
    auto y0 = _mm256_loadu_ps(Y+i);
    auto y1 = _mm256_loadu_ps(Y+i+8);
    auto z0 = _mm256_loadu_ps(Z+i);
    auto z1 = _mm256_loadu_ps(Z+i+8);

    _mm256_storeu_ps(X+i, avx2_approx_swiglu(y0, z0));
    _mm256_storeu_ps(X+i+8, avx2_approx_swiglu(y1, z1));
  }

  if (i+8 < N) [[unlikely]]
  {
    auto y0 = _mm256_loadu_ps(Y+i);
    auto z0 = _mm256_loadu_ps(Z+i);
    _mm256_storeu_ps(X+i, avx2_approx_swiglu(y0, z0));
    i += 8;
  }

  if (i> N) [[unlikely]]
  {
    alignas(64) int mask[] = 
    {
      -1, -1, -1, -1, -1, -1, -1, -1,
       0,  0,  0,  0,  0,  0,  0,  0
    };

    auto vmask = _mm256_loadu_si256((__m256i*)(mask+(i&7)));
    auto ym = _mm256_maskload_ps(Y+i, vmask);
    auto zm = _mm256_maskload_ps(Z+i, vmask);
    _mm256_maskstore_ps(X+i, vmask, avx2_approx_swiglu(ym, zm));
  }
}


} // namespace nntrainer::avx2
