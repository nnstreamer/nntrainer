// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_simd_quant.h
 * @date   15 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  SIMD-optimized quantization functions for GGML interface
 */

#pragma once

#include <cstdint>
#include <cstring>

// Runtime CPU feature detection
namespace nntrainer {
namespace simd {

/**
 * @brief CPU feature detection and capability flags
 */
struct CPUFeatures {
  bool has_neon = false;
  bool has_avx2 = false;
  bool has_sse42 = false;
  
  static const CPUFeatures& getInstance() {
    static CPUFeatures instance;
    return instance;
  }
  
private:
  CPUFeatures() {
    detectFeatures();
  }
  
  void detectFeatures();
};

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

/**
 * @brief ARM NEON optimized q8_K quantization
 */
inline void quantize_row_q8_K_neon(const float* __restrict src, void* __restrict dst, int64_t k) {
  constexpr int QK_K = 256;
  const int nb = k / QK_K;
  
  uint8_t* __restrict y = static_cast<uint8_t*>(dst);
  
  for (int i = 0; i < nb; i++) {
    const float* __restrict x = src + i * QK_K;
    
    // Find absolute maximum using NEON
    float32x4_t max_vec = vdupq_n_f32(0.0f);
    
    for (int j = 0; j < QK_K; j += 16) {
      float32x4_t v0 = vld1q_f32(x + j);
      float32x4_t v1 = vld1q_f32(x + j + 4);
      float32x4_t v2 = vld1q_f32(x + j + 8);
      float32x4_t v3 = vld1q_f32(x + j + 12);
      
      v0 = vabsq_f32(v0);
      v1 = vabsq_f32(v1);
      v2 = vabsq_f32(v2);
      v3 = vabsq_f32(v3);
      
      max_vec = vmaxq_f32(max_vec, v0);
      max_vec = vmaxq_f32(max_vec, v1);
      max_vec = vmaxq_f32(max_vec, v2);
      max_vec = vmaxq_f32(max_vec, v3);
    }
    
    // Horizontal max reduction
    float32x2_t max_pair = vmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
    float amax = vmaxv_f32(max_pair);
    
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    
    // Store scale factor
    memcpy(y + i * (QK_K + sizeof(float)), &d, sizeof(float));
    
    // Quantize using NEON
    float32x4_t id_vec = vdupq_n_f32(id);
    int8_t* __restrict qs = reinterpret_cast<int8_t*>(y + i * (QK_K + sizeof(float)) + sizeof(float));
    
    for (int j = 0; j < QK_K; j += 16) {
      float32x4_t v0 = vld1q_f32(x + j);
      float32x4_t v1 = vld1q_f32(x + j + 4);
      float32x4_t v2 = vld1q_f32(x + j + 8);
      float32x4_t v3 = vld1q_f32(x + j + 12);
      
      v0 = vmulq_f32(v0, id_vec);
      v1 = vmulq_f32(v1, id_vec);
      v2 = vmulq_f32(v2, id_vec);
      v3 = vmulq_f32(v3, id_vec);
      
      int32x4_t i0 = vcvtnq_s32_f32(v0);
      int32x4_t i1 = vcvtnq_s32_f32(v1);
      int32x4_t i2 = vcvtnq_s32_f32(v2);
      int32x4_t i3 = vcvtnq_s32_f32(v3);
      
      int16x4_t s0 = vqmovn_s32(i0);
      int16x4_t s1 = vqmovn_s32(i1);
      int16x4_t s2 = vqmovn_s32(i2);
      int16x4_t s3 = vqmovn_s32(i3);
      
      int16x8_t s01 = vcombine_s16(s0, s1);
      int16x8_t s23 = vcombine_s16(s2, s3);
      
      int8x8_t q0 = vqmovn_s16(s01);
      int8x8_t q1 = vqmovn_s16(s23);
      
      vst1_s8(qs + j, q0);
      vst1_s8(qs + j + 8, q1);
    }
  }
}

#endif // ARM_NEON

#if defined(__AVX2__)
#include <immintrin.h>

/**
 * @brief x64 AVX2 optimized q8_K quantization
 */
inline void quantize_row_q8_K_avx2(const float* __restrict src, void* __restrict dst, int64_t k) {
  constexpr int QK_K = 256;
  const int nb = k / QK_K;
  
  uint8_t* __restrict y = static_cast<uint8_t*>(dst);
  
  for (int i = 0; i < nb; i++) {
    const float* __restrict x = src + i * QK_K;
    
    // Find absolute maximum using AVX2
    __m256 max_vec = _mm256_setzero_ps();
    
    for (int j = 0; j < QK_K; j += 32) {
      __m256 v0 = _mm256_loadu_ps(x + j);
      __m256 v1 = _mm256_loadu_ps(x + j + 8);
      __m256 v2 = _mm256_loadu_ps(x + j + 16);
      __m256 v3 = _mm256_loadu_ps(x + j + 24);
      
      // Apply sign mask to get absolute values
      const __m256 sign_mask = _mm256_set1_ps(-0.0f);
      v0 = _mm256_andnot_ps(sign_mask, v0);
      v1 = _mm256_andnot_ps(sign_mask, v1);
      v2 = _mm256_andnot_ps(sign_mask, v2);
      v3 = _mm256_andnot_ps(sign_mask, v3);
      
      max_vec = _mm256_max_ps(max_vec, v0);
      max_vec = _mm256_max_ps(max_vec, v1);
      max_vec = _mm256_max_ps(max_vec, v2);
      max_vec = _mm256_max_ps(max_vec, v3);
    }
    
    // Horizontal max reduction
    __m128 max_low = _mm256_castps256_ps128(max_vec);
    __m128 max_high = _mm256_extractf128_ps(max_vec, 1);
    __m128 max_combined = _mm_max_ps(max_low, max_high);
    
    max_combined = _mm_max_ps(max_combined, _mm_shuffle_ps(max_combined, max_combined, _MM_SHUFFLE(2, 3, 0, 1)));
    max_combined = _mm_max_ps(max_combined, _mm_shuffle_ps(max_combined, max_combined, _MM_SHUFFLE(1, 0, 3, 2)));
    
    float amax = _mm_cvtss_f32(max_combined);
    
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    
    // Store scale factor
    memcpy(y + i * (QK_K + sizeof(float)), &d, sizeof(float));
    
    // Quantize using AVX2
    __m256 id_vec = _mm256_set1_ps(id);
    int8_t* __restrict qs = reinterpret_cast<int8_t*>(y + i * (QK_K + sizeof(float)) + sizeof(float));
    
    for (int j = 0; j < QK_K; j += 32) {
      __m256 v0 = _mm256_loadu_ps(x + j);
      __m256 v1 = _mm256_loadu_ps(x + j + 8);
      __m256 v2 = _mm256_loadu_ps(x + j + 16);
      __m256 v3 = _mm256_loadu_ps(x + j + 24);
      
      v0 = _mm256_mul_ps(v0, id_vec);
      v1 = _mm256_mul_ps(v1, id_vec);
      v2 = _mm256_mul_ps(v2, id_vec);
      v3 = _mm256_mul_ps(v3, id_vec);
      
      __m256i i0 = _mm256_cvtps_epi32(v0);
      __m256i i1 = _mm256_cvtps_epi32(v1);
      __m256i i2 = _mm256_cvtps_epi32(v2);
      __m256i i3 = _mm256_cvtps_epi32(v3);
      
      __m256i packed_16_01 = _mm256_packs_epi32(i0, i1);
      __m256i packed_16_23 = _mm256_packs_epi32(i2, i3);
      
      __m256i packed_8 = _mm256_packs_epi16(packed_16_01, packed_16_23);
      
      // Fix lane ordering for AVX2
      packed_8 = _mm256_permute4x64_epi64(packed_8, _MM_SHUFFLE(3, 1, 2, 0));
      
      _mm256_storeu_si256((__m256i*)(qs + j), packed_8);
    }
  }
}

#endif // AVX2

/**
 * @brief CPU feature detection implementation
 */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

inline void CPUFeatures::detectFeatures() {
#ifdef _WIN32
  int info[4];
  __cpuid(info, 1);
  has_sse42 = (info[2] & (1 << 20)) != 0;
  
  __cpuid(info, 7);
  has_avx2 = (info[1] & (1 << 5)) != 0;
#else
  unsigned int eax, ebx, ecx, edx;
  
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    has_sse42 = (ecx & (1 << 20)) != 0;
  }
  
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    has_avx2 = (ebx & (1 << 5)) != 0;
  }
#endif
}

#elif defined(__ARM_NEON) || defined(__aarch64__)

inline void CPUFeatures::detectFeatures() {
  has_neon = true; // If compiled with NEON, assume it's available
}

#else

inline void CPUFeatures::detectFeatures() {
  // No SIMD support detected
}

#endif

/**
 * @brief Runtime dispatch for optimized quantization
 */
inline void quantize_row_q8_K_optimized(const float* src, void* dst, int64_t k) {
  const auto& features = CPUFeatures::getInstance();
  
#if defined(__AVX2__)
  if (features.has_avx2) {
    quantize_row_q8_K_avx2(src, dst, k);
    return;
  }
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
  if (features.has_neon) {
    quantize_row_q8_K_neon(src, dst, k);
    return;
  }
#endif

  // Fallback to scalar implementation
  // This would call the original ggml quantization function
  ::quantize_row_q8_K(src, dst, k);
}

} // namespace simd
} // namespace nntrainer