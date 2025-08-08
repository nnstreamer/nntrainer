/**
 * Copyright (C) 2025 Daekyoung Jung <daekyoung.jung@gmail.com>
 *
 * @file	q4_0x8_tool.cpp
 * @date	08 August 2025
 * @brief	Define functions to process q4_0x8 format data
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Daekyoung Jung <daekyoung.jung@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "q4_0x8_tool.h"

#include <omp.h>

#include <cstdio>
#include <cstring>

/**
 * @brief struct for 8 q4_0 blocks
 */
struct block_q4_0x8 {
  unsigned short d[8];
  unsigned char qs[128];
};

typedef unsigned long long ull;

namespace nntrainer {

void convert_q4_0x8_shuffle(const void *src, unsigned short *d,
                            unsigned char *qs, int N, int K) {
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

  for (int b = 0; b < N * 8; ++b, qs += 16) {
    unsigned char out[16];
    for (int i = 0; i < 8; ++i) {
      unsigned char x0 = qs[2 * i];
      unsigned char x1 = qs[2 * i + 1];

      out[i + 0] =
        (unsigned char)(x0 & 0x0F) | (unsigned char)((x1 & 0x0F) << 4);
      out[i + 8] =
        (unsigned char)((x0 & 0xF0) >> 4) | (unsigned char)(x1 & 0xF0);
    }
    memcpy(qs, out, sizeof(unsigned char) * 16);
  }
}

void convert_q4_0x8_shuffle_omp(const void *src, unsigned short *d,
                                unsigned char *qs, int N, int K) {
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
#pragma omp parallel for
  for (int b = 0; b < N * 8; ++b) {
    unsigned char out[16];
    for (int i = 0; i < 8; ++i) {
      unsigned char x0 = qs[b * 16 + 2 * i];
      unsigned char x1 = qs[b * 16 + 2 * i + 1];

      out[i + 0] =
        (unsigned char)(x0 & 0x0F) | (unsigned char)((x1 & 0x0F) << 4);
      out[i + 8] =
        (unsigned char)((x0 & 0xF0) >> 4) | (unsigned char)(x1 & 0xF0);
    }
    memcpy(&qs[b * 16], out, sizeof(unsigned char) * 16);
  }
}

} // namespace nntrainer