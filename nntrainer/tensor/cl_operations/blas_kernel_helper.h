// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file	blas_kernel_helper.h
 * @date	11 August 2025
 * @brief	OpenCL BLAS kernel helper functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __BLAS_KERNEL_HELPER_H__
#define __BLAS_KERNEL_HELPER_H__

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <string>

#if defined(__clang__) || defined(__GNUC__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

namespace nntrainer {

/**
 * @brief struct of q4_0x8 block
 */
struct block_q4_0x8 {
  uint16_t d[8];   // 16B
  uint8_t qs[128]; // 16 x u64
};

#if defined(__x86_64__) || defined(__i586__) || defined(_M_X64) ||             \
  defined(_M_IX86)

// SSSE3 nibble "butterfly" for one 16B vector in a register.
// ---------------- tuning knobs ----------------
#ifndef Q4X8_USE_STREAMING_STORES
#define Q4X8_USE_STREAMING_STORES 1 // set to 1 if qs_out won't be read soon
#endif
#ifndef Q4X8_PREFETCH_DIST
#define Q4X8_PREFETCH_DIST 8 // set 0 to disable; in "blocks" within loop
#endif
// ----------------------------------------------

static inline __m256i butterfly32(__m256i a) {
  const __m256i SHUF_EVEN = _mm256_setr_epi8(
    0, 2, 4, 6, 8, 10, 12, 14, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80, 0, 2, 4, 6, 8, 10, 12, 14,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80);
  const __m256i SHUF_ODD = _mm256_setr_epi8(
    1, 3, 5, 7, 9, 11, 13, 15, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80, 1, 3, 5, 7, 9, 11, 13, 15,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80);
  const __m256i even = _mm256_shuffle_epi8(a, SHUF_EVEN);
  const __m256i odd = _mm256_shuffle_epi8(a, SHUF_ODD);
  const __m256i LO = _mm256_set1_epi8(0x0F);
  const __m256i HI = _mm256_set1_epi8((char)0xF0);
  __m256i low =
    _mm256_or_si256(_mm256_and_si256(even, LO),
                    _mm256_slli_epi16(_mm256_and_si256(odd, LO), 4));
  __m256i high =
    _mm256_or_si256(_mm256_srli_epi16(_mm256_and_si256(even, HI), 4),
                    _mm256_and_si256(odd, HI));
  high = _mm256_slli_si256(high, 8);
  return _mm256_or_si256(low, high);
}

// Build 16B packet [d0|d1] from two 8B chunks using vector loads (no GPR
// moves).
static inline __m128i make_pkt128(const uint8_t *base_qs, int d0, int d1) {
  __m128i lo = _mm_loadl_epi64((const __m128i *)(base_qs + ((size_t)d0 << 3)));
  __m128i hi = _mm_loadl_epi64((const __m128i *)(base_qs + ((size_t)d1 << 3)));
  return _mm_unpacklo_epi64(lo, hi);
}

// ================== core template with QS unrolled by 8 blocks
// ==================
template <int UNIT, int GROUPS>
static inline void convert_q4_0x8_noshuffle(const void *src,
                                            uint16_t *RESTRICT d_out,
                                            uint8_t *RESTRICT qs_out) {
  static_assert(UNIT % 16 == 0, "UNIT must be multiple of 16");
  constexpr int BLOCKS_PER_GROUP = UNIT / 8;  // d entries per offset per group
  constexpr int PAIRS_PER_OFFSET = UNIT / 16; // 16B packets per half per offset
  static_assert((PAIRS_PER_OFFSET % 4) == 0,
                "need multiple of 4 packets (8 blocks) per iter");

  constexpr size_t D_ELEMS_PER_GROUP = 8 * BLOCKS_PER_GROUP;
  constexpr size_t QS_BYTES_PER_GROUP = (size_t)16 * UNIT;
  constexpr size_t QS_BYTES_PER_OFFSET = (size_t)2 * UNIT;

  const block_q4_0x8 *x = (const block_q4_0x8 *)src;
  const __m256i bias256 = _mm256_set1_epi8((char)0x88);

#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < GROUPS; ++b) {
    for (int offset = 0; offset < 8; ++offset) {

      // ---- D slice ----
      {
        uint16_t *d_ptr = d_out + (size_t)b * D_ELEMS_PER_GROUP +
                          (size_t)offset * BLOCKS_PER_GROUP;
        const block_q4_0x8 *xb = x + (size_t)b * BLOCKS_PER_GROUP;
        for (int i = 0; i < BLOCKS_PER_GROUP; ++i) {
          d_ptr[i] = xb[i].d[offset];
        }
      }

      // ---- QS slice (unroll 8 blocks / 128B per iter) ----
      {
        uint8_t *qs_ptr = qs_out + (size_t)b * QS_BYTES_PER_GROUP +
                          (size_t)offset * QS_BYTES_PER_OFFSET;
        const int base_q = (b * UNIT * 2) + offset;
        const int d0 = (base_q & 15), d1 = d0 ^ 8;

        auto do_half = [&](int blk_base) {
          // Each iter handles 8 consecutive blocks: j..j+7
          for (int j = 0; j < PAIRS_PER_OFFSET; j += 8) {
            const uint8_t *q0 = x[blk_base + j + 0].qs;
            const uint8_t *q1 = x[blk_base + j + 1].qs;
            const uint8_t *q2 = x[blk_base + j + 2].qs;
            const uint8_t *q3 = x[blk_base + j + 3].qs;
            const uint8_t *q4 = x[blk_base + j + 4].qs;
            const uint8_t *q5 = x[blk_base + j + 5].qs;
            const uint8_t *q6 = x[blk_base + j + 6].qs;
            const uint8_t *q7 = x[blk_base + j + 7].qs;

#if Q4X8_PREFETCH_DIST > 0
            _mm_prefetch(
              (const char *)(x[blk_base + j + Q4X8_PREFETCH_DIST].qs),
              _MM_HINT_NTA);
#endif
            // Build 8 packets in XMM regs
            __m128i pkt0 = make_pkt128(q0, d0, d1);
            __m128i pkt1 = make_pkt128(q1, d0, d1);
            __m128i pkt2 = make_pkt128(q2, d0, d1);
            __m128i pkt3 = make_pkt128(q3, d0, d1);
            __m128i pkt4 = make_pkt128(q4, d0, d1);
            __m128i pkt5 = make_pkt128(q5, d0, d1);
            __m128i pkt6 = make_pkt128(q6, d0, d1);
            __m128i pkt7 = make_pkt128(q7, d0, d1);

            // Four 32B batches: [0|1], [2|3], [4|5], [6|7]
            __m256i v01 = _mm256_set_m128i(pkt1, pkt0);
            __m256i v23 = _mm256_set_m128i(pkt3, pkt2);
            __m256i v45 = _mm256_set_m128i(pkt5, pkt4);
            __m256i v67 = _mm256_set_m128i(pkt7, pkt6);

            v01 = _mm256_xor_si256(v01, bias256);
            v23 = _mm256_xor_si256(v23, bias256);
            v45 = _mm256_xor_si256(v45, bias256);
            v67 = _mm256_xor_si256(v67, bias256);

            __m256i o01 = butterfly32(v01);
            __m256i o23 = butterfly32(v23);
            __m256i o45 = butterfly32(v45);
            __m256i o67 = butterfly32(v67);

#if Q4X8_USE_STREAMING_STORES
            _mm256_stream_si256((__m256i *)(qs_ptr + 0), o01);
            _mm256_stream_si256((__m256i *)(qs_ptr + 32), o23);
            _mm256_stream_si256((__m256i *)(qs_ptr + 64), o45);
            _mm256_stream_si256((__m256i *)(qs_ptr + 96), o67);
#else
            _mm256_storeu_si256((__m256i *)(qs_ptr + 0), o01);
            _mm256_storeu_si256((__m256i *)(qs_ptr + 32), o23);
            _mm256_storeu_si256((__m256i *)(qs_ptr + 64), o45);
            _mm256_storeu_si256((__m256i *)(qs_ptr + 96), o67);
#endif
            qs_ptr += 128;
          }
        };

        // first half
        do_half(base_q >> 4);
        // second half (same d0/d1 pattern)
        do_half((base_q + UNIT) >> 4);
      }
    }
  }

#if Q4X8_USE_STREAMING_STORES
  _mm_sfence();
#endif
}

// ================== wrappers for your K,N combinations ==================
// K = 3072 (UNIT = 768)
void convert_q4_0x8_shuffle_K3072_N98304(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out) {
  // groups = (N*8)/UNIT = 1024
  convert_q4_0x8_noshuffle<768, 1024>(src, d_out, qs_out);
}
void convert_q4_0x8_shuffle_K3072_N36864(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out) {
  // groups = 384
  convert_q4_0x8_noshuffle<768, 384>(src, d_out, qs_out);
}
void convert_q4_0x8_shuffle_K3072_N3072(const void *src, uint16_t *d_out,
                                        uint8_t *qs_out) {
  // groups = 32
  convert_q4_0x8_noshuffle<768, 32>(src, d_out, qs_out);
}

// K = 8192 (UNIT = 2048)
void convert_q4_0x8_shuffle_K8192_N98304(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out) {
  // groups = 384
  convert_q4_0x8_noshuffle<2048, 384>(src, d_out, qs_out);
}
void convert_q4_0x8_shuffle_K8192_N36864(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out) {
  // groups = 144
  convert_q4_0x8_noshuffle<2048, 144>(src, d_out, qs_out);
}
void convert_q4_0x8_shuffle_K8192_N3072(const void *src, uint16_t *d_out,
                                        uint8_t *qs_out) {
  // groups = 12
  convert_q4_0x8_noshuffle<2048, 12>(src, d_out, qs_out);
}

#elif defined(__aarch64__) || defined(__ARM_ARCH_7A__) ||                      \
  defined(__ANDROID__) || defined(__arm__)
void convert_q4_0x8_shuffle_dispatch(const void *src, uint16_t *d_out,
                                     uint8_t *qs_out, int N, int K) {
  throw std::invalid_argument("NYI in aarch64");
}

#else
void convert_q4_0x8_shuffle_dispatch(const void *src, uint16_t *d_out,
                                     uint8_t *qs_out, int N, int K) {
  throw std::invalid_argument("NYI as a fallback");
}
#endif

// Optional tiny dispatcher if you want one entry point:
void convert_q4_0x8_shuffle_dispatch(const void *src, uint16_t *d_out,
                                     uint8_t *qs_out, int N, int K) {
  if (K == 3072) {
    if (N == 98304)
      return convert_q4_0x8_shuffle_K3072_N98304(src, d_out, qs_out);
    if (N == 36864)
      return convert_q4_0x8_shuffle_K3072_N36864(src, d_out, qs_out);
    if (N == 3072)
      return convert_q4_0x8_shuffle_K3072_N3072(src, d_out, qs_out);
  } else { // K == 8192
    if (N == 98304)
      return convert_q4_0x8_shuffle_K8192_N98304(src, d_out, qs_out);
    if (N == 36864)
      return convert_q4_0x8_shuffle_K8192_N36864(src, d_out, qs_out);
    if (N == 3072)
      return convert_q4_0x8_shuffle_K8192_N3072(src, d_out, qs_out);
  }
  // If a new combo appears, fall back to a generic version (not shown here).
  assert(!"Unsupported (K,N) combination");
}

} // namespace nntrainer
#endif /* __BLAS_KERNEL_HELPER_H__ */
