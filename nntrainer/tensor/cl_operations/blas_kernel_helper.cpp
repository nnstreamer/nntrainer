#include "blas_kernel_helper.h"

#include <ggml-common.h>
#include <omp.h>

#include <cassert>
#include <cstring>

#define QK4_0 32
#define QK8_0 32

template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return QK4_0;
  }
  if constexpr (K == 8) {
    return QK8_0;
  }
  return -1;
}

typedef unsigned short ggml_half;
typedef unsigned long long ull;
typedef char int8_t;
typedef unsigned char uint8_t;

template <int K, int N> struct block {
  ggml_half d[N];                     // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

// control size
static_assert(sizeof(block<4, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 2,
              "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 4,
              "wrong block<4,8> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 4,
              "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 8,
              "wrong block<8,8> size/padding");

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;

typedef struct {
  ggml_half d;
  uint8_t qs[16];
} block_q4_0;

namespace nntrainer {
void convert_st(const void *src, unsigned short *d, unsigned char *qs,
                size_t N) {
  block_q4_0x8 *x = (block_q4_0x8 *)src;
  for (int i = 0; i < N; ++i) {
    std::memcpy(&d[i * 8], x[i].d, sizeof(unsigned short) * 8);
    std::memcpy(&qs[i * 128], x[i].qs, sizeof(unsigned char) * 128);
    for (int j = 0; j < 128; j += 8) {
      ull *ptr = (ull *)&qs[i * 128 + j];
      constexpr ull mask = 0x8888888888888888ULL;
      *ptr ^= mask;
    }
  }
}

void convert_q4_0x8_st(const void *src, unsigned short *d, unsigned char *qs,
                       size_t N, int K) {
  block_q4_0x8 *x = (block_q4_0x8 *)src;
  int unit = K / 4;

  for (int b = 0; b < N * 8 / unit; ++b) {
    for (int offset = 0; offset < 8; ++offset) {
      for (int stride = 0; stride < unit; stride += 8) {
        int idx = b * unit + stride + offset;
        int blk_idx = idx / 8;
        int d_idx = idx % 8;
        *d++ = x[blk_idx].d[d_idx];
      }
    }
  }

  ull *ptr = (ull *)qs;
  for (int b = 0; b < N * 8 / unit; ++b) {
    for (int offset = 0; offset < 8; ++offset) {
      for (int st = 0; st < unit * 2; st += 8) {
        int idx = b * unit * 2 + offset + st;
        int blk_idx = idx / 16;
        int d_idx = idx % 16;
        ull *src = (ull *)&x[blk_idx].qs;
        constexpr ull mask = 0x8888888888888888ULL;
        *ptr++ = src[d_idx] ^ mask;
      }
    }
  }
}

void convert_q4_0x8_omp(const void *src, unsigned short *d, unsigned char *qs,
                        size_t N, int K) {
  block_q4_0x8 *x = (block_q4_0x8 *)src;
  int unit = K / 4;
#pragma omp parallel for
  for (int b = 0; b < N * 8 / unit; ++b) {
    for (int offset = 0; offset < 8; ++offset) {
      for (int stride = 0; stride < unit; stride += 8) {
        int idx = b * unit + stride + offset;
        int blk_idx = idx / 8;
        int d_idx = idx % 8;
        d[b * unit + offset * unit / 8 + stride / 8] = x[blk_idx].d[d_idx];
      }
    }
  }

  ull *ptr = (ull *)qs;
#pragma omp parallel for
  for (int b = 0; b < N * 8 / unit; ++b) {
    for (int offset = 0; offset < 8; ++offset) {
      for (int st = 0; st < unit * 2; st += 8) {
        int idx = b * unit * 2 + offset + st;
        int blk_idx = idx / 16;
        int d_idx = idx % 16;
        ull *src = (ull *)&x[blk_idx].qs;
        constexpr ull mask = 0x8888888888888888ULL;
        ptr[b * unit * 2 + offset * unit / 4 + st / 8] = src[d_idx] ^ mask;
      }
    }
  }
}

} // namespace nntrainer