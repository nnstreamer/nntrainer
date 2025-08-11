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

struct block_q4_0x8 {
  uint16_t d[8];   // 16B
  uint8_t qs[128]; // 16 x u64
};

#if defined(__x86_64__) || defined(__i586__) || defined(_M_X64) ||             \
  defined(_M_IX86)

// SSSE3 nibble "butterfly" for one 16B vector in a register.
static inline __m128i nibble_butterfly16_reg(__m128i a) {
  const __m128i SHUF_EVEN =
    _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, (char)0x80, (char)0x80, (char)0x80,
                  (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80);
  const __m128i SHUF_ODD =
    _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, (char)0x80, (char)0x80, (char)0x80,
                  (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80);

  const __m128i even = _mm_shuffle_epi8(a, SHUF_EVEN);
  const __m128i odd = _mm_shuffle_epi8(a, SHUF_ODD);

  const __m128i LO = _mm_set1_epi8(0x0F);
  const __m128i HI = _mm_set1_epi8((char)0xF0);

  __m128i low = _mm_or_si128(_mm_and_si128(even, LO),
                             _mm_slli_epi16(_mm_and_si128(odd, LO), 4));
  __m128i high = _mm_or_si128(_mm_srli_epi16(_mm_and_si128(even, HI), 4),
                              _mm_and_si128(odd, HI));
  high = _mm_slli_si128(high, 8);
  return _mm_or_si128(low, high);
}

// Set this to 1 to use non-temporal (streaming) stores for qs_out on big
// batches.
#ifndef Q4X8_USE_STREAMING_STORES
#define Q4X8_USE_STREAMING_STORES 1
#endif

template <int UNIT> // UNIT is K/4: 768 for K=3072, 2048 for K=8192
static inline void convert_q4_0x8_noshuffle(const void *src,
                                            uint16_t *RESTRICT d_out,
                                            uint8_t *RESTRICT qs_out, int N) {
  static_assert(UNIT % 16 == 0, "UNIT must be multiple of 16");
  const block_q4_0x8 *x = (const block_q4_0x8 *)src;

  // Derived compile-time constants
  constexpr int BLOCKS_PER_GROUP =
    UNIT / 8; // #blocks contributing to d per (group)
  constexpr int PAIRS_PER_OFFSET =
    UNIT / 16; // #16B qs packets per offset per half
  const int groups = (N * 8) / UNIT;

  // Bias for XOR (flip bit3 of every nibble)
  const __m256i bias256 = _mm256_set1_epi8((char)0x88);

  for (int b = 0; b < groups; ++b) {
    const int base_blocks = b * BLOCKS_PER_GROUP; // == (b*UNIT)/8

    // ---- D path: for each lane (offset), stream d[offset] from consecutive
    // blocks
    for (int offset = 0; offset < 8; ++offset) {
      // write UNIT/8 entries: x[base_blocks + i].d[offset]
      const block_q4_0x8 *xb = x + base_blocks;
      for (int i = 0; i < BLOCKS_PER_GROUP; ++i) {
        *d_out++ = xb[i].d[offset];
      }
    }

    // ---- QS path: for each lane (offset), two halves of UNIT:
    //               first half [0..UNIT), second half [UNIT..2*UNIT)
    for (int offset = 0; offset < 8; ++offset) {
      // Common parts computed once
      const int base_q = (b * UNIT * 2) + offset; // base_twox + offset
      const int base_mod = base_q & 15;           // d_idx pattern start (st=0)
      const int d0 = base_mod;
      const int d1 = base_mod ^ 8; // st=8 toggles bit4

      // First half: blocks [blk_base .. blk_base + PAIRS_PER_OFFSET-1]
      {
        int blk_base = base_q >> 4; // (base_q)/16
        // Process **two blocks per iter** → 32B batch
        for (int j = 0; j < PAIRS_PER_OFFSET; j += 2) {
          const uint64_t *qA = (const uint64_t *)(x[blk_base + j + 0].qs);
          const uint64_t *qB = (const uint64_t *)(x[blk_base + j + 1].qs);

          // Two 16B packets: [d0,d1] from block A, then from block B
          __m128i pkt0 = _mm_set_epi64x((long long)qA[d1], (long long)qA[d0]);
          __m128i pkt1 = _mm_set_epi64x((long long)qB[d1], (long long)qB[d0]);

          // 32B XOR
          __m256i v32 = _mm256_set_m128i(pkt1, pkt0);
          v32 = _mm256_xor_si256(v32, bias256);

          // butterfly each 16B and store
          __m128i v0 = _mm256_castsi256_si128(v32);
          __m128i v1 = _mm256_extracti128_si256(v32, 1);
          __m128i out0 = nibble_butterfly16_reg(v0);
          __m128i out1 = nibble_butterfly16_reg(v1);
#if Q4X8_USE_STREAMING_STORES
          _mm_stream_si128((__m128i *)(qs_out + 0), out0);
          _mm_stream_si128((__m128i *)(qs_out + 16), out1);
#else
          _mm_storeu_si128((__m128i *)(qs_out + 0), out0);
          _mm_storeu_si128((__m128i *)(qs_out + 16), out1);
#endif
          qs_out += 32;
        }
      }

      // Second half: same d_idx pattern (UNIT is multiple of 16), just start
      // from a later block
      {
        int blk_base =
          (base_q + UNIT) >> 4; // +UNIT /16 = +PAIRS_PER_OFFSET blocks
        for (int j = 0; j < PAIRS_PER_OFFSET; j += 2) {
          const uint64_t *qA = (const uint64_t *)(x[blk_base + j + 0].qs);
          const uint64_t *qB = (const uint64_t *)(x[blk_base + j + 1].qs);

          __m128i pkt0 = _mm_set_epi64x((long long)qA[d1], (long long)qA[d0]);
          __m128i pkt1 = _mm_set_epi64x((long long)qB[d1], (long long)qB[d0]);

          __m256i v32 = _mm256_set_m128i(pkt1, pkt0);
          v32 = _mm256_xor_si256(v32, bias256);

          __m128i v0 = _mm256_castsi256_si128(v32);
          __m128i v1 = _mm256_extracti128_si256(v32, 1);
          __m128i out0 = nibble_butterfly16_reg(v0);
          __m128i out1 = nibble_butterfly16_reg(v1);
#if Q4X8_USE_STREAMING_STORES
          _mm_stream_si128((__m128i *)(qs_out + 0), out0);
          _mm_stream_si128((__m128i *)(qs_out + 16), out1);
#else
          _mm_storeu_si128((__m128i *)(qs_out + 0), out0);
          _mm_storeu_si128((__m128i *)(qs_out + 16), out1);
#endif
          qs_out += 32;
        }
      }
    }
  }

#if Q4X8_USE_STREAMING_STORES
  _mm_sfence(); // ensure NT stores are globally visible
#endif
}
#elif defined(__aarch64__) || defined(__ARM_ARCH_7A__) ||                      \
  defined(__ANDROID__) || defined(__arm__)
template <int UNIT>
static inline void convert_q4_0x8_noshuffle(const void *src,
                                            uint16_t *RESTRICT d_out,
                                            uint8_t *RESTRICT qs_out, int N) {
  throw std::invalid_argument("NYI in aarch64");
}

#else
template <int UNIT>
static inline void convert_q4_0x8_noshuffle(const void *src,
                                            uint16_t *RESTRICT d_out,
                                            uint8_t *RESTRICT qs_out, int N) {
  throw std::invalid_argument("NYI in aarch64");
}
#endif

void convert_q4_0x8_noshuffle_3072(const void *src, uint16_t *RESTRICT d_out,
                                   uint8_t *RESTRICT qs_out, int N) {
  convert_q4_0x8_noshuffle<768>(src, d_out, qs_out, N);
}

void convert_q4_0x8_noshuffle_8192(const void *src, uint16_t *RESTRICT d_out,
                                   uint8_t *RESTRICT qs_out, int N) {
  convert_q4_0x8_noshuffle<2048>(src, d_out, qs_out, N);
}

} // namespace nntrainer
#endif /* __BLAS_KERNEL_HELPER_H__ */
