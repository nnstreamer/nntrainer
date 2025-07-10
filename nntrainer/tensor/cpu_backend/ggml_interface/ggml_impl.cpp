#include "ggml-common.h"
#include "ggml-cpu-quants.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <ggml_interface.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

typedef struct {
  GGML_EXTENSION union {
    struct {
      ggml_half d;    // super-block scale for quantized scales
      ggml_half dmin; // super-block scale for quantized mins
    } GGML_COMMON_AGGR_S;
    ggml_half2 dm;
  } GGML_COMMON_AGGR_U;
  uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
} q4_K_qparams;

struct block_q4_Kx8_impl {
  ggml_half d[8];     // super-block scale for quantized scales
  ggml_half dmin[8];  // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};
namespace nntrainer {

template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return QK4_0;
  }
  if constexpr (K == 8) {
    return QK8_0;
  }
  return -1;
}

template <int K, int N> struct block {
  uint16_t d[N];                      // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q8_0x4 = block<8, 4>;

static inline float nntr_compute_fp16_to_fp32(uint16_t h) {
  _FP16 tmp;
  memcpy(&tmp, &h, sizeof(uint16_t));
  return (float)tmp;
}

static inline uint16_t nntr_compute_fp32_to_fp16(float f) {
  uint16_t res;
  _FP16 tmp = f;
  memcpy(&res, &tmp, sizeof(uint16_t));
  return res;
}

static inline void get_scale_min_k4(int j, const uint8_t *GGML_RESTRICT q,
                                    uint8_t *GGML_RESTRICT d,
                                    uint8_t *GGML_RESTRICT m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

static inline int nearest_int(float fval) {
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

inline void extract_qparam_q4_K(void *src, void *dst) {
  block_q4_K *GGML_RESTRICT q4_k = (block_q4_K *)src;
  q4_K_qparams *GGML_RESTRICT qparams = (q4_K_qparams *)dst;
  qparams->dm = q4_k->dm;
  memcpy(qparams->scales, q4_k->scales, sizeof(q4_k->scales));
  qparams->d = q4_k->d;
  qparams->dmin = q4_k->dmin;
}

void __nntr_quantize_row_q8_0(const _FP16 *__restrict x, void *vy, int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0 *__restrict y = (block_q8_0 *__restrict)vy;

#if defined(__ARM_NEON)
  for (int i = 0; i < nb; i++) {
    float16x8_t srcv[4];  // loaded source
    float16x8_t asrcv[4]; // absolute value of source
    float16x8_t amaxv[2]; // absolute max buffer

    for (int j = 0; j < 4; j++) {
      srcv[j] = vld1q_f16(x + i * 32 + 8 * j);
    }
    for (int j = 0; j < 4; j++) {
      asrcv[j] = vabsq_f16(srcv[j]);
    }

    for (int j = 0; j < 2; j++) {
      amaxv[j] =
        vmaxq_f16(asrcv[2 * j], asrcv[2 * j + 1]); // 0, 1 <- 0, 1 VS 2, 3
    }
    amaxv[0] = vmaxq_f16(amaxv[0], amaxv[1]); // 0 <- 0, 1

    const float amax = static_cast<float>(vmaxvq_f16(amaxv[0]));

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_compute_fp32_to_fp16(d);

    for (int j = 0; j < 4; j++) {
      const float16x8_t v = vmulq_n_f16(srcv[j], id);
      const int16x8_t vi = vcvtnq_s16_f16(v);

      y[i].qs[8 * j + 0] = vgetq_lane_s16(vi, 0);
      y[i].qs[8 * j + 1] = vgetq_lane_s16(vi, 1);
      y[i].qs[8 * j + 2] = vgetq_lane_s16(vi, 2);
      y[i].qs[8 * j + 3] = vgetq_lane_s16(vi, 3);
      y[i].qs[8 * j + 4] = vgetq_lane_s16(vi, 4);
      y[i].qs[8 * j + 5] = vgetq_lane_s16(vi, 5);
      y[i].qs[8 * j + 6] = vgetq_lane_s16(vi, 6);
      y[i].qs[8 * j + 7] = vgetq_lane_s16(vi, 7);
    }
  }
#else
  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i * QK8_0 + j];
      amax = std::max(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_compute_fp32_to_fp16(d);

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = std::roundf(x0);
    }
  }
#endif
}

void __nntr_quantize_row_q8_0_ref_lossless(const float *GGML_RESTRICT x,
                                           void *GGML_RESTRICT _y, int64_t k,
                                           void *GGML_RESTRICT _y_ref) {

  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0 *GGML_RESTRICT y_ref = (block_q8_0 * GGML_RESTRICT) _y_ref;
  block_q8_0 *GGML_RESTRICT y = (block_q8_0 * GGML_RESTRICT) _y;

  for (int i = 0; i < nb; i++) {
    const float d = nntr_compute_fp16_to_fp32(y_ref[i].d);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = y_ref[i].d;

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = roundf(x0);
    }
  }
}

void __nntr_quantize_row_q4_K_ref_lossless(const float *GGML_RESTRICT x,
                                           void *GGML_RESTRICT _y, int64_t k,
                                           void *GGML_RESTRICT _y_ref) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  block_q4_K *GGML_RESTRICT y = (block_q4_K * GGML_RESTRICT) _y;
  block_q4_K *GGML_RESTRICT y_ref = (block_q4_K * GGML_RESTRICT) _y_ref;

  uint8_t L[QK_K];

  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < QK_K / 32; ++j) {
      if (j < 4) {
        y[i].scales[j] = y_ref[i].scales[j];
        y[i].scales[j + 4] = y_ref[i].scales[j + 4];
      } else {
        y[i].scales[j + 4] = y_ref[i].scales[j + 4];
        y[i].scales[j - 4] |= y_ref[i].scales[j - 4];
        y[i].scales[j - 0] |= y_ref[i].scales[j - 0];
      }
    }
    y[i].d = y_ref[i].d;
    y[i].dmin = y_ref[i].dmin;

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = nntr_compute_fp16_to_fp32(y[i].d) * sc;
      if (!d)
        continue;
      const float dm = nntr_compute_fp16_to_fp32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }

    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l)
        q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }
    x += QK_K;
  }
}

size_t __nntr_quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
  __nntr_quantize_row_q8_0(src, dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}

void __nntr_dequantize_row_q8_0(const void *_x, _FP16 *__restrict y,
                                int64_t k) {
  static const int qk = QK8_0;
  const block_q8_0 *__restrict x = (const block_q8_0 *__restrict)_x;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    // const _FP16 d = (x[i].d); ///@todo check if this works
    const float d = nntr_compute_fp16_to_fp32(x[i].d);

    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = x[i].qs[j] * d;
    }
  }
}

void __nntr_quantize_mat_q8_0_4x8(const _FP16 *GGML_RESTRICT x,
                                  void *GGML_RESTRICT vy, int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0x4 *GGML_RESTRICT y = (block_q8_0x4 *)vy;

#if defined(__ARM_NEON)
  float16x8_t srcv[4][4];
  float id[4];

  for (int i = 0; i < nb; i++) {
    float16x8_t asrcv[4];
    float16x8_t amaxv[2];

    for (int row_iter = 0; row_iter < 4; row_iter++) {
      for (int j = 0; j < 4; j++)
        srcv[row_iter][j] = vld1q_f16(x + row_iter * k + i * 32 + 8 * j);
      for (int j = 0; j < 4; j++)
        asrcv[j] = vabsq_f16(srcv[row_iter][j]);

      for (int j = 0; j < 2; j++) {
        amaxv[j] =
          vmaxq_f16(asrcv[2 * j], asrcv[2 * j + 1]); // 0, 1 <- 0, 1 VS 2, 3
      }
      amaxv[0] = vmaxq_f16(amaxv[0], amaxv[1]); // 0 <- 0, 1

      const float amax = vmaxvq_f16(amaxv[0]);

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < 4; j++) {
      float16x8_t v = vmulq_n_f16(srcv[0][j], id[0]);
      int16x8_t vi = vcvtnq_s16_f16(v);
      y[i].qs[32 * j + 0] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 1] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 2] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 3] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 4] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 5] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 6] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 7] = vgetq_lane_s16(vi, 7);

      v = vmulq_n_f16(srcv[1][j], id[1]);
      vi = vcvtnq_s16_f16(v);
      y[i].qs[32 * j + 8] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 9] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 10] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 11] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 12] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 13] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 14] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 15] = vgetq_lane_s16(vi, 7);

      v = vmulq_n_f16(srcv[2][j], id[2]);
      vi = vcvtnq_s16_f16(v);
      y[i].qs[32 * j + 16] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 17] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 18] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 19] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 20] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 21] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 22] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 23] = vgetq_lane_s16(vi, 7);

      v = vmulq_n_f16(srcv[3][j], id[3]);
      vi = vcvtnq_s16_f16(v);
      y[i].qs[32 * j + 24] = vgetq_lane_s16(vi, 0);
      y[i].qs[32 * j + 25] = vgetq_lane_s16(vi, 1);
      y[i].qs[32 * j + 26] = vgetq_lane_s16(vi, 2);
      y[i].qs[32 * j + 27] = vgetq_lane_s16(vi, 3);
      y[i].qs[32 * j + 28] = vgetq_lane_s16(vi, 4);
      y[i].qs[32 * j + 29] = vgetq_lane_s16(vi, 5);
      y[i].qs[32 * j + 30] = vgetq_lane_s16(vi, 6);
      y[i].qs[32 * j + 31] = vgetq_lane_s16(vi, 7);
    }
  }
#else
  // scalar
  const int blck_size_interleave = 8;
  _FP16 srcv[4][QK8_0];
  float id[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max

      for (int j = 0; j < QK8_0; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
        amax = MAX(amax, fabsf(srcv[row_iter][j]));
      }

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < QK8_0 * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);

      float x0 = srcv[src_id][src_offset] * id[src_id];
      y[i].qs[j] = roundf(x0);
    }
  }
#endif
}

template <>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const _FP16 *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, _FP16 *C,
                               const unsigned int ldc) {
  int NB_COLS = 4;
  std::vector<float> C32 = std::vector<float>(M * N);

  // q40 GEMV accuracy explodes?
  if (M == 1) { // GEMV
    int n_threads = 4;
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);
    __nntr_quantize_row_q8_0(A, (void *)QA.data(), K);

#pragma omp parallel for num_threads(n_threads)
    for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      unsigned int M_step_start = (thread_idx * N) / n_threads;     // = 0
      unsigned int M_step_end = ((thread_idx + 1) * N) / n_threads; // ne01 = N

      M_step_start = (M_step_start % NB_COLS)
                       ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                       : M_step_start;
      M_step_end = (M_step_end % NB_COLS)
                     ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                     : M_step_end;

      ::ggml_gemv_q4_0_4x8_q8_0(K, (float *)((C32.data()) + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA.data(), M, M_step_end - M_step_start);
    }
  } else {
    int n_threads = 8;
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;
    unsigned int M4 = ((M - M % 4) / 4);
    int B_step = sizeof(block_q4_0) * (K / QK4_0);

    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantize 4-divisible-M row portion with matrix-wise function
    for (unsigned int i = 0; i < M4; i++) {
      __nntr_quantize_mat_q8_0_4x8(A + 4 * i * K,
                                   (void *)(QA.data() + i * qa_4_rows_size), K);
    }
    // Quantize leftover 1 ~ 3 rows with row-wise function
    for (unsigned int i = M4 * 4; i < M; i++) {
      __nntr_quantize_row_q8_0(A + i * K,
                               (void *)(QA.data() + (M4 * qa_4_rows_size) +
                                        (i - M4 * 4) * qa_row_size),
                               K);
    }

// Compute 4-divisible-M row portion with multithreaded GEMM
#pragma omp parallel for collapse(1) num_threads(n_threads)
    for (int i = 0; i < n_threads; i++) {
      unsigned int src0_start = (i * N) / n_threads;
      unsigned int src0_end = ((i + 1) * N) / n_threads;

      src0_start = (src0_start % NB_COLS)
                     ? src0_start + NB_COLS - (src0_start % NB_COLS)
                     : src0_start;
      src0_end = (src0_end % NB_COLS)
                   ? src0_end + NB_COLS - (src0_end % NB_COLS)
                   : src0_end;

      ::ggml_gemm_q4_0_4x8_q8_0(K, (float *)((C32.data()) + src0_start), ldc,
                                (void *)((char *)B + src0_start * B_step),
                                QA.data(), M4 * 4, src0_end - src0_start);
    }

    // Compute leftover 1 ~ 3 rows with multithreaded GEMV
    n_threads = 4;
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
#pragma omp parallel for num_threads(n_threads)
      for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
        unsigned int M_step_start = (thread_idx * N) / n_threads; // = 0
        unsigned int M_step_end =
          ((thread_idx + 1) * N) / n_threads; // ne01 = N

        M_step_start = (M_step_start % NB_COLS)
                         ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                         : M_step_start;
        M_step_end = (M_step_end % NB_COLS)
                       ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                       : M_step_end;

        ::ggml_gemv_q4_0_4x8_q8_0(
          K,
          (float *)(((C32.data()) + ((pb - M4 * 4) * N) + (M4 * 4 * N)) +
                    M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      }
    }
  }
#if defined(__ARM_NEON)
  for (int i = 0; i < M * N; i += 8) {
    vst1q_f16(
      (C + i),
      vcombine_f16(vcvt_f16_f32(vld1q_f32((float *)(C32.data() + i))),
                   vcvt_f16_f32(vld1q_f32((float *)(C32.data() + i + 4)))));
  }
#else
  for (unsigned int i = 0; i < M * N; i++) {
    C[i] = static_cast<_FP16>(C32[i]);
  }
#endif
}

int __ggml_unpack_q4_K_8_bl_to_q4_K(void *GGML_RESTRICT dst,
                                    const void *GGML_RESTRICT data,
                                    size_t data_size, size_t nrow, size_t k) {
  int interleave_block = 8;
  GGML_ASSERT(interleave_block ==
              8); // Only 8-way interleaving is supported for q4_K
  constexpr size_t nrows_interleaved = 8;
  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1; // nrow must be multiple of 8 and k must be divisible by 8 (block
               // size alignment)
  }

  int nblocks = k / QK_K; // number of block_q4_K per row
  // Expected size: (nrow/8 groups) * nblocks * sizeof(block_q4_Kx8_impl)
  GGML_ASSERT(data_size ==
              (nrow / nrows_interleaved) * nblocks * sizeof(block_q4_Kx8_impl));

  const block_q4_Kx8_impl *src_blocks = (const block_q4_Kx8_impl *)data;
  block_q4_K *dst_blocks = (block_q4_K *)dst;

  // Iterate over each group of 8 rows
  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    // Each group of 8 rows produced nblocks combined blocks
    for (int64_t x = 0; x < nblocks; ++x) {
      const block_q4_Kx8_impl &src_block =
        *src_blocks++; // current combined block

      // Temporary storage for 8 small blocks to reconstruct
      block_q4_K temp[8];

      // Step 3: Unpack d and dmin for each small block
      for (int j = 0; j < 8; ++j) {
        // temp[j].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = src_block.d[j];
        // temp[j].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin =
        // src_block.dmin[j];
        temp[j].d = src_block.d[j];
        temp[j].dmin = src_block.dmin[j];
      }

      // Step 4: De-interleave qs array (copy 1024 bytes back into 8x128 bytes)
      for (int chunk = 0; chunk < 128; ++chunk) {
        int j = chunk % 8;            // target small block index
        int offset = (chunk / 8) * 8; // 8-byte aligned offset in small block
        // Copy 8 bytes from combined block to the appropriate small block's qs
        memcpy(&temp[j].qs[offset], &src_block.qs[chunk * 8], 8);
      }

      // Step 5: Unpack scales for each sub-block i = 0..3
      for (int i = 0; i < 4; ++i) {
        // Offsets in the combined scales array for this sub-block
        size_t off0 = i * 12;
        size_t off1 = 48 + i * 12;

        // Extract s and m (6-bit scale/min values) from first 48 bytes
        uint8_t s_val[8], m_val[8];
        // j=0..3: directly from lower 6 bits
        for (int j = 0; j < 4; ++j) {
          s_val[j] = src_block.scales[off0 + j] & 0x3F;
          m_val[j] = src_block.scales[off0 + 4 + j] & 0x3F;
        }
        // j=4..7: reconstruct from combined bits
        // Low 4 bits from bytes off0+8..11, high 2 bits from top bits of
        // off0+0..3 or +4..7
        s_val[4] = (uint8_t)((src_block.scales[off0 + 8] & 0x0F) |
                             ((src_block.scales[off0 + 0] >> 6) & 0x03) << 4);
        s_val[5] = (uint8_t)((src_block.scales[off0 + 9] & 0x0F) |
                             ((src_block.scales[off0 + 1] >> 6) & 0x03) << 4);
        s_val[6] = (uint8_t)((src_block.scales[off0 + 10] & 0x0F) |
                             ((src_block.scales[off0 + 2] >> 6) & 0x03) << 4);
        s_val[7] = (uint8_t)((src_block.scales[off0 + 11] & 0x0F) |
                             ((src_block.scales[off0 + 3] >> 6) & 0x03) << 4);

        m_val[4] = (uint8_t)((src_block.scales[off0 + 8] >> 4 & 0x0F) |
                             ((src_block.scales[off0 + 4] >> 6) & 0x03) << 4);
        m_val[5] = (uint8_t)((src_block.scales[off0 + 9] >> 4 & 0x0F) |
                             ((src_block.scales[off0 + 5] >> 6) & 0x03) << 4);
        m_val[6] = (uint8_t)((src_block.scales[off0 + 10] >> 4 & 0x0F) |
                             ((src_block.scales[off0 + 6] >> 6) & 0x03) << 4);
        m_val[7] = (uint8_t)((src_block.scales[off0 + 11] >> 4 & 0x0F) |
                             ((src_block.scales[off0 + 7] >> 6) & 0x03) << 4);

        // Extract secondary s2 and m2 from bytes 48-95 (second half)
        uint8_t s2_val[8], m2_val[8];
        for (int j = 0; j < 4; ++j) {
          s2_val[j] = src_block.scales[off1 + j] & 0x3F;
          m2_val[j] = src_block.scales[off1 + 4 + j] & 0x3F;
        }
        s2_val[4] =
          (uint8_t)((src_block.scales[off1 + 8] & 0x0F) |
                    (((src_block.scales[off1 + 0] >> 6) & 0x03) << 4));
        s2_val[5] =
          (uint8_t)((src_block.scales[off1 + 9] & 0x0F) |
                    (((src_block.scales[off1 + 1] >> 6) & 0x03) << 4));
        s2_val[6] =
          (uint8_t)((src_block.scales[off1 + 10] & 0x0F) |
                    (((src_block.scales[off1 + 2] >> 6) & 0x03) << 4));
        s2_val[7] =
          (uint8_t)((src_block.scales[off1 + 11] & 0x0F) |
                    (((src_block.scales[off1 + 3] >> 6) & 0x03) << 4));

        m2_val[4] =
          (uint8_t)((src_block.scales[off1 + 8] >> 4 & 0x0F) |
                    (((src_block.scales[off1 + 4] >> 6) & 0x03) << 4));
        m2_val[5] =
          (uint8_t)((src_block.scales[off1 + 9] >> 4 & 0x0F) |
                    (((src_block.scales[off1 + 5] >> 6) & 0x03) << 4));
        m2_val[6] =
          (uint8_t)((src_block.scales[off1 + 10] >> 4 & 0x0F) |
                    (((src_block.scales[off1 + 6] >> 6) & 0x03) << 4));
        m2_val[7] =
          (uint8_t)((src_block.scales[off1 + 11] >> 4 & 0x0F) |
                    (((src_block.scales[off1 + 7] >> 6) & 0x03) << 4));

        // Now reconstruct the original 3 bytes for each sub-block index i
        for (int j = 0; j < 8; ++j) {
          // Byte i of scales (bits0-5 from s_val, bits6-7 from s2_val)
          uint8_t low6 = s_val[j] & 0x3F;
          uint8_t high2 = (s2_val[j] & 0x30)
                          << 2; // bits4-5 of s2_val -> bits6-7
          temp[j].scales[i] = low6 | high2;
          // Byte i+4 of scales (bits0-5 from m_val, bits6-7 from m2_val)
          low6 = m_val[j] & 0x3F;
          high2 = (m2_val[j] & 0x30) << 2;
          temp[j].scales[i + 4] = low6 | high2;
          // Byte i+8 of scales (low4 from s2_val, high4 from m2_val)
          uint8_t low4 = s2_val[j] & 0x0F;
          uint8_t high4 = (m2_val[j] & 0x0F) << 4;
          temp[j].scales[i + 8] = low4 | high4;
        }
      } // end for each sub-block i

      // Step 6: Store the reconstructed 8 blocks in the output at correct
      // positions
      for (int j = 0; j < 8; ++j) {
        // Row index = b + j, column index = x
        size_t dst_index = (b + j) * nblocks + x;
        dst_blocks[dst_index] = temp[j];
      }
    } // for each block column x
  }   // for each group of 8 rows

  return 0;
}

} // namespace nntrainer

/*
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2,
"wrong q4_K block size/padding");

struct block_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4--bit quants
};

static block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in, unsigned int
blck_size_interleave) { block_q4_Kx8 out;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto
the output interleaved structure for (int i = 0; i < 8; i++) { out.d[i] =
in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 4 / blck_size_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins
values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes (
6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for
corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub
block from different Q4_K structures uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] &
240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    return out;
}

// The method was modified (from repack_q4_0_to_q4_0_4_bl) to be used by
nntrainer int ggml_repack_q4_0_to_q4_0_4_bl(void * GGML_RESTRICT dst, int
interleave_block, const void * GGML_RESTRICT data, size_t data_size, size_t
nrow, size_t k) { GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q4_0x4 * dst_ = (block_q4_0x4 *)dst;
    const block_q4_0 * src = (const block_q4_0 *)data;
    block_q4_0 dst_tmp[4];
    int nblocks = k / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
        return -1;
    }

    for (size_t b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (size_t i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst_++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

int ggml_repack_q4_K_to_q4_K_8_bl(void * GGML_RESTRICT dst, int
interleave_block, const void * GGML_RESTRICT data, size_t data_size, size_t
nrow, size_t k) { GGML_ASSERT(interleave_block == 8); constexpr size_t
nrows_interleaved = 8;

    block_q4_Kx8 * dst_ = (block_q4_Kx8*)dst;
    const block_q4_K * src = (const block_q4_K*) data;
    block_q4_K dst_tmp[8];
    int nblocks = k / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_K));

    if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
        return -1;
    }

    for (size_t b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (size_t i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst_++ = make_block_q4_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

Examine the code above, and implement a function:
int ggml_unpack_q4_K_8_bl_to_q4_K(void * GGML_RESTRICT _dst, const void *
GGML_RESTRICT _src, size_t data_size, size_t nrow, size_t k)
{
    block_q4_K *dst = (block_q4_K *)_dst;
    block_q4_Kx8 *src = (block_q4_Kx8 *)_src;

    // This function should unpack the q4_Kx8 blocks back to q4_K blocks.
    // The implementation will depend on the structure of block_q4_Kx8 and
block_q4_K.
    // It should reverse the process done in ggml_repack_q4_K_to_q4_K_8_bl.
    // The function signature and return type should match the expected
behavior. return 0; // Placeholder return value, implement the actual logic.
}
*/
