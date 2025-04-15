// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ggml_avx_macro.h
 * @date	03 April 2025
 * @brief	This is avx macros for ggml usage
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */
#include <immintrin.h>

typedef uint16_t ggml_fp16_t;

#if defined(__AVX__)
#if defined(__F16C__)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)     _mm512_cvtph_ps(_mm256_set_m128i(_mm_loadu_si128((const __m128i *)(y)), _mm_loadu_si128((const __m128i *)(x))))
#define GGML_F32Cx16_REPEAT_LOAD(x)  _mm512_cvtph_ps(_mm256_set_m128i(x, x))
#endif
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     _mm256_cvtph_ps(_mm_shuffle_epi32(_mm_maskload_epi32((int const*)(x), loadMask), 68))
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask))
#else
#if defined(__AVX512F__)
static inline __m512 __avx512_f32cx8x2_load(ggml_fp16_t *x, ggml_fp16_t *y) {
    float tmp[16];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
    }

    for (int i = 0; i < 8; i++) {
        tmp[i + 8] = GGML_FP16_TO_FP32(y[i]);
    }

    return _mm512_loadu_ps(tmp);
}
static inline __m512 __avx512_repeat_f32cx16_load(__m128i x) {
    float tmp[16];
    uint16_t tmphalf[8];
    _mm_storeu_si128((__m128i*)tmphalf, x);

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 4] = GGML_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 8] = GGML_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 12] = GGML_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm512_loadu_ps(tmp);
}
#endif
static inline __m256 __avx_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_repeat_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
        tmp[i + 4] = GGML_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_rearranged_f32cx8_load(ggml_fp16_t *x, __m128i arrangeMask) {
    uint16_t tmphalf[8];
    float tmp[8];

    _mm_storeu_si128((__m128i*)tmphalf, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask));
    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm256_loadu_ps(tmp);
}

#define GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     __avx_repeat_f32cx8_load(x)
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     __avx_rearranged_f32cx8_load(x, arrangeMask)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)     __avx512_f32cx8x2_load(x, y)
#define GGML_F32Cx16_REPEAT_LOAD(x)  __avx512_repeat_f32cx16_load(x)
#endif
#endif
#endif


#if defined(__AVX2__) || defined(__AVX512F__)
#if defined(__AVX512F__)
// add int16_t pairwise and return as 512 bit int vector
static inline __m512i sum_i16_pairs_int_32x16(const __m512i x) {
    const __m512i ones = _mm512_set1_epi16(1);
    return _mm512_madd_epi16(ones, x);
}

static inline __m512i mul_sum_us8_pairs_int32x16(const __m512i ax, const __m512i sy) {
#if defined(__AVX512VNNI__)
    const __m512i zero = _mm512_setzero_si512();
    return _mm512_dpbusd_epi32(zero, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
    return sum_i16_pairs_int_32x16(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as 512 bit int vector
static inline __m512i mul_sum_i8_pairs_int32x16(const __m512i x, const __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    // Get absolute values of x vectors
    const __m512i ax = _mm512_abs_epi8(x);
    // Sign the values of the y vectors
    __mmask64 blt0 = _mm512_movepi8_mask(x);
    const __m512i sy = _mm512_mask_sub_epi8(y, blt0, zero, y);
    return mul_sum_us8_pairs_int32x16(ax, sy);
}
#endif

// add int16_t pairwise and return as 256 bit int vector
static inline __m256i sum_i16_pairs_int32x8(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, x);
}

static inline __m256i mul_sum_us8_pairs_int32x8(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_dpbusd_epi32(zero, ax, sy);
#elif defined(__AVXVNNI__)
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_dpbusd_avx_epi32(zero, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_int32x8(dot);
#endif
}

// Integer variant of the function defined in ggml-quants.c
// multiply int8_t, add results pairwise twice and return as 256 bit int vector
static inline __m256i mul_sum_i8_pairs_int32x8(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_dpbssd_epi32(zero, x, y);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_int32x8(ax, sy);
#endif
}
#endif
