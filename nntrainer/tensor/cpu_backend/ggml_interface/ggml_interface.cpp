// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface.cpp
 * @date   15 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use ggml lib from cpu_backend
 */

#include "ggml-common.h"
#include "ggml-cpu-quants.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <assert.h>
#include <bs_thread_pool_manager.hpp>
#include <ggml_interface.h>
#include <string>
#include <thread>
#include <vector>

namespace nntrainer {
/**
 * @brief Continuously packed 4 q8_K
 *
 */
struct block_q8_Kx4 {
  float d[4];              // delta
  int8_t qs[QK_K * 4];     // quants
  int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

/**
 * @brief struct template for q4_0 and q8_0
 *
 * @tparam K 4 or 8
 * @return constexpr int number of elements in the quantized block
 */
template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return QK4_0;
  }
  if constexpr (K == 8) {
    return QK8_0;
  }
  return -1;
}

/**
 * @brief block of q4_0 or q8_0 block
 *
 * @tparam K 4 or 8
 * @tparam N number of blocks to be packed
 */
template <int K, int N> struct block {
  ggml_half d[N];                     // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q4_0x4 = block<4, 4>;
using block_q8_0x4 = block<8, 4>;

void __ggml_init() {
  // needed to initialize f16 tables
  struct ggml_init_params params = {0, NULL, false};
  struct ggml_context *ctx = ggml_init(params);
  ggml_free(ctx);
}

size_t __ggml_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return ::quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return ::quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return ::quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

void __ggml_quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  ::quantize_q6_K(src, dst, 1, k, nullptr);
}

void __ggml_quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  ::quantize_row_q8_K(src, dst, k);
}

void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  // auto &bspool = ThreadPoolManager::getInstance();

  if (M == 1) { // GEMV
    int n_threads = 4;
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    std::vector<char> QA = std::vector<char>(qa_size);
    ::quantize_row_q8_0(A, QA.data(), K);

#pragma omp parallel for num_threads(n_threads)
    for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      unsigned int M_step_start = (thread_idx * N) / n_threads;     // = 0
      unsigned int M_step_end = ((thread_idx + 1) * N) / n_threads; // ne01 = N

      M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8)
                                        : M_step_start;
      M_step_end =
        (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

      ::ggml_gemv_q4_0_8x8_q8_0(K, (float *)((C) + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA.data(), M, M_step_end - M_step_start);
    }
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    unsigned int M4 = ((M + 3) / 4);

    unsigned int qa_size = qa_4_rows_size * M4;
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantization of activations
    /// @note Heuristic inspection conducted that applying multithreading on
    /// run-time quantization hurts model latency
    // #pragma omp parallel for collapse(1) num_threads(16)
    for (int i = 0; i < static_cast<int>(M4); i++) {
      ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K,
                                   QA.data() + i * qa_4_rows_size, K);
    }
    int delta = 8;
    int step_N = N / delta;
    int step_C = delta;
    int step_B = blocks_per_4_rows * sizeof(block_q4_0) * delta;
#pragma omp parallel for collapse(1) num_threads(16)
    for (int i = 0; i < step_N; i++) {
      ::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc, (char *)B + i * step_B,
                                QA.data(), M, delta);
    }
    /**
    @todo Add BS threadpool multithread strategy
    BS::multi_future<void> multi_future = bspool.submit_loop(0, step_N, [&](int
    i){::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc, (char *)B + i * step_B,
                                QA.data(), M, delta);});
      multi_future.wait();
     */
  }
}

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) { // GEMV
    int n_threads = 4;
    unsigned int blocks_per_row = (K + QK_K - 1) / QK_K;
    unsigned int qa_size = sizeof(block_q8_K) * blocks_per_row;
    unsigned int B_step = sizeof(block_q4_K) * (K / QK_K);

    std::vector<char> QA = std::vector<char>(qa_size);

    ::quantize_row_q8_K(A, QA.data(), K);

#pragma omp parallel for num_threads(n_threads)
    for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      unsigned int M_step_start = (thread_idx * N) / n_threads;     // = 0
      unsigned int M_step_end = ((thread_idx + 1) * N) / n_threads; // ne01 = N

      M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8)
                                        : M_step_start;
      M_step_end =
        (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

      ::ggml_gemv_q4_K_8x8_q8_K(K, (float *)((C) + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA.data(), M, M_step_end - M_step_start);
    }
  } else if (M % 4 != 0) {
    int n_threads = std::thread::hardware_concurrency();
    unsigned int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
    unsigned int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_K) * K) / QK_K;
    unsigned int M4 = ((M - M % 4) / 4);
    int B_step = sizeof(block_q4_K) * (K / QK_K);

    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantize 4-divisible-M row portion with matrix-wise function
    for (unsigned int i = 0; i < M4; i++) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K,
                                   QA.data() + i * qa_4_rows_size, K);
    }
    // Quantize leftover 1 ~ 3 rows with row-wise function
    for (unsigned int i = M4 * 4; i < M; i++) {
      ::quantize_row_q8_K(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

// Compute 4-divisible-M row portion with multithreaded GEMM
#pragma omp parallel for collapse(1) num_threads(n_threads)
    for (int i = 0; i < n_threads; i++) {
      unsigned int src0_start = (i * N) / n_threads;
      unsigned int src0_end = ((i + 1) * N) / n_threads;

      src0_start =
        (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;
      src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

      ::ggml_gemm_q4_K_8x8_q8_K(K, (float *)(C + src0_start), ldc,
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

        M_step_start = (M_step_start % 8)
                         ? M_step_start + 8 - (M_step_start % 8)
                         : M_step_start;
        M_step_end =
          (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

        ::ggml_gemv_q4_K_8x8_q8_K(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      }
    }
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
    unsigned int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    unsigned int M4 = ((M + 3) / 4);
    unsigned int B_step = sizeof(block_q4_K) * (K / QK_K);
    ///@note OpenMP thread number should be a signed integer
    int thread_num = std::thread::hardware_concurrency();

    unsigned int qa_size = qa_4_rows_size * M4;
    std::vector<char> QA = std::vector<char>(qa_size);

    // Quantization of activations
    /// @note Heuristic inspection conducted that applying multithreading on
    /// run-time quantization hurts model latency
    // #pragma omp parallel for collapse(1) num_threads(16)
    for (int i = 0; i < static_cast<int>(M4); i++) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K,
                                   QA.data() + i * qa_4_rows_size, K);
    }

#pragma omp parallel for collapse(1) num_threads(thread_num)
    for (int i = 0; i < thread_num; i++) {
      unsigned int src0_start = (i * N) / thread_num;
      unsigned int src0_end = ((i + 1) * N) / thread_num;

      src0_start =
        (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;
      src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

      ::ggml_gemm_q4_K_8x8_q8_K(K, (float *)(C + src0_start), ldc,
                                (void *)((char *)B + src0_start * B_step),
                                QA.data(), M, src0_end - src0_start);
    }
  }
}

static inline float fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||             \
     defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&                        \
  (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
    fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
    fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
    sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                        : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

void nntr_ggml_vec_dot_q6_K_q8_K(int n, float *GGML_RESTRICT s, size_t bs,
                                 const void *GGML_RESTRICT vx, size_t bx,
                                 const void *GGML_RESTRICT vy, size_t by,
                                 int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);
  UNUSED(nrc);
  UNUSED(bx);
  UNUSED(by);
  UNUSED(bs);

  const block_q6_K *GGML_RESTRICT x = vx;
  const block_q8_K *GGML_RESTRICT y = vy;

  const int nb = n / QK_K;

#ifdef __ARM_FEATURE_SVE
  const int vector_length = ggml_cpu_get_sve_cnt() * 8;
  float sum = 0;
  svuint8_t m4b = svdup_n_u8(0xf);
  svint32_t vzero = svdup_n_s32(0);
  svuint8_t mone = svdup_n_u8(0x30);
  svint8_t q6bytes_1, q6bytes_2, q6bytes_3, q6bytes_4;
  svuint8_t q6h_1, q6h_2, q6h_3, q6h_4;

  for (int i = 0; i < nb; ++i) {
    const float d_all = GGML_FP16_TO_FP32(x[i].d);

    const uint8_t *GGML_RESTRICT q6 = x[i].ql;
    const uint8_t *GGML_RESTRICT qh = x[i].qh;
    const int8_t *GGML_RESTRICT q8 = y[i].qs;

    const int8_t *GGML_RESTRICT scale = x[i].scales;

    const svbool_t pg16_8 = svptrue_pat_b16(SV_VL8);
    const svint16_t q8sums_1 = svld1_s16(pg16_8, y[i].bsums);
    const svint16_t q8sums_2 = svld1_s16(pg16_8, y[i].bsums + 8);
    const svint16_t q6scales_1 =
      svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), scale));
    const svint16_t q6scales_2 =
      svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), scale + 8));
    const svint64_t prod = svdup_n_s64(0);
    int32_t isum_mins = svaddv_s64(
      svptrue_b64(),
      svadd_s64_x(svptrue_b64(), svdot_s64(prod, q8sums_1, q6scales_1),
                  svdot_s64(prod, q8sums_2, q6scales_2)));
    int32_t isum = 0;

    switch (vector_length) {
    case 128: {
      const svbool_t pg32_4 = svptrue_pat_b32(SV_VL4);
      const svbool_t pg8_16 = svptrue_pat_b8(SV_VL16);
      svint32_t isum_tmp = svdup_n_s32(0);
      for (int j = 0; j < QK_K / 128; ++j) {
        svuint8_t qhbits_1 = svld1_u8(pg8_16, qh);
        svuint8_t qhbits_2 = svld1_u8(pg8_16, qh + 16);
        qh += 32;
        svuint8_t q6bits_1 = svld1_u8(pg8_16, q6);
        svuint8_t q6bits_2 = svld1_u8(pg8_16, q6 + 16);
        svuint8_t q6bits_3 = svld1_u8(pg8_16, q6 + 32);
        svuint8_t q6bits_4 = svld1_u8(pg8_16, q6 + 48);
        q6 += 64;
        svint8_t q8bytes_1 = svld1_s8(pg8_16, q8);
        svint8_t q8bytes_2 = svld1_s8(pg8_16, q8 + 16);
        svint8_t q8bytes_3 = svld1_s8(pg8_16, q8 + 32);
        svint8_t q8bytes_4 = svld1_s8(pg8_16, q8 + 48);
        q8 += 64;

        q6h_1 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_1, 4));
        q6h_2 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_2, 4));
        q6h_3 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_1, 2));
        q6h_4 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_2, 2));
        q6bytes_1 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_1, m4b), q6h_1));
        q6bytes_2 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_2, m4b), q6h_2));
        q6bytes_3 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_3, m4b), q6h_3));
        q6bytes_4 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_4, m4b), q6h_4));
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale[0]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale[1]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale[2]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale[3]);

        scale += 4;
        q8bytes_1 = svld1_s8(pg8_16, q8);
        q8bytes_2 = svld1_s8(pg8_16, q8 + 16);
        q8bytes_3 = svld1_s8(pg8_16, q8 + 32);
        q8bytes_4 = svld1_s8(pg8_16, q8 + 48);
        q8 += 64;

        q6h_1 = svand_u8_x(pg16_8, mone, qhbits_1);
        q6h_2 = svand_u8_x(pg16_8, mone, qhbits_2);
        q6h_3 = svand_u8_x(pg16_8, mone, svlsr_n_u8_x(pg16_8, qhbits_1, 2));
        q6h_4 = svand_u8_x(pg16_8, mone, svlsr_n_u8_x(pg16_8, qhbits_2, 2));
        q6bytes_1 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_1, 4), q6h_1));
        q6bytes_2 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_2, 4), q6h_2));
        q6bytes_3 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_3, 4), q6h_3));
        q6bytes_4 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_4, 4), q6h_4));
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale[0]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale[1]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale[2]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale[3]);
        scale += 4;
      }
      isum += svaddv_s32(pg32_4, isum_tmp);
      sum += d_all * y[i].d * (isum - 32 * isum_mins);
    } break;
    case 256:
    case 512: {
      const svbool_t pg8_2 = svptrue_pat_b8(SV_VL2);
      const svbool_t pg32_8 = svptrue_pat_b32(SV_VL8);
      const svbool_t pg8_32 = svptrue_pat_b8(SV_VL32);
      svint32_t isum_tmp = svdup_n_s32(0);
      for (int j = 0; j < QK_K / 128; j++) {
        svuint8_t qhbits_1 = svld1_u8(pg8_32, qh);
        qh += 32;
        svuint8_t q6bits_1 = svld1_u8(pg8_32, q6);
        svuint8_t q6bits_2 = svld1_u8(pg8_32, q6 + 32);
        q6 += 64;
        svint8_t q8bytes_1 = svld1_s8(pg8_32, q8);
        svint8_t q8bytes_2 = svld1_s8(pg8_32, q8 + 32);
        svint8_t q8bytes_3 = svld1_s8(pg8_32, q8 + 64);
        svint8_t q8bytes_4 = svld1_s8(pg8_32, q8 + 96);
        q8 += 128;
        q6h_1 = svand_u8_x(pg8_32, mone, svlsl_n_u8_x(pg8_32, qhbits_1, 4));
        q6h_2 = svand_u8_x(pg8_32, mone, svlsl_n_u8_x(pg8_32, qhbits_1, 2));
        q6h_3 = svand_u8_x(pg8_32, mone, qhbits_1);
        q6h_4 = svand_u8_x(pg8_32, mone, svlsr_n_u8_x(pg8_32, qhbits_1, 2));
        q6bytes_1 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_32, svand_u8_x(pg8_32, q6bits_1, m4b), q6h_1));
        q6bytes_2 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_32, svand_u8_x(pg8_32, q6bits_2, m4b), q6h_2));
        q6bytes_3 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_32, svlsr_n_u8_x(pg8_32, q6bits_1, 4), q6h_3));
        q6bytes_4 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_32, svlsr_n_u8_x(pg8_32, q6bits_2, 4), q6h_4));

        svint8_t scale_lane_1_tmp = svld1_s8(pg8_2, scale);
        scale_lane_1_tmp = svzip1_s8(scale_lane_1_tmp, scale_lane_1_tmp);
        scale_lane_1_tmp = svzip1_s8(scale_lane_1_tmp, scale_lane_1_tmp);
        svint8_t scale_lane_2_tmp = svld1_s8(pg8_2, scale + 2);
        scale_lane_2_tmp = svzip1_s8(scale_lane_2_tmp, scale_lane_2_tmp);
        scale_lane_2_tmp = svzip1_s8(scale_lane_2_tmp, scale_lane_2_tmp);
        svint8_t scale_lane_3_tmp = svld1_s8(pg8_2, scale + 4);
        scale_lane_3_tmp = svzip1_s8(scale_lane_3_tmp, scale_lane_3_tmp);
        scale_lane_3_tmp = svzip1_s8(scale_lane_3_tmp, scale_lane_3_tmp);
        svint8_t scale_lane_4_tmp = svld1_s8(pg8_2, scale + 6);
        scale_lane_4_tmp = svzip1_s8(scale_lane_4_tmp, scale_lane_4_tmp);
        scale_lane_4_tmp = svzip1_s8(scale_lane_4_tmp, scale_lane_4_tmp);
        svint32_t scale_lane_1 = svunpklo_s32(svunpklo_s16(scale_lane_1_tmp));
        svint32_t scale_lane_2 = svunpklo_s32(svunpklo_s16(scale_lane_2_tmp));
        svint32_t scale_lane_3 = svunpklo_s32(svunpklo_s16(scale_lane_3_tmp));
        svint32_t scale_lane_4 = svunpklo_s32(svunpklo_s16(scale_lane_4_tmp));

        isum_tmp =
          svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1),
                      scale_lane_1);
        isum_tmp =
          svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2),
                      scale_lane_2);
        isum_tmp =
          svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3),
                      scale_lane_3);
        isum_tmp =
          svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4),
                      scale_lane_4);
        scale += 8;
      }
      isum += svaddv_s32(pg32_8, isum_tmp);
      sum += d_all * y[i].d * (isum - 32 * isum_mins);
    } break;
    default:
      assert(false && "Unsupported vector length");
      break;
    }
  }

  *s = sum;

#elif __ARM_NEON
  float sum = 0;

  const uint8x16_t m4b = vdupq_n_u8(0xF);
  const int32x4_t vzero = vdupq_n_s32(0);
  // const int8x16_t  m32s = vdupq_n_s8(32);

  const uint8x16_t mone = vdupq_n_u8(3);

  ggml_int8x16x4_t q6bytes;
  ggml_uint8x16x4_t q6h;

  for (int i = 0; i < nb; ++i) {

    const float d_all = GGML_FP16_TO_FP32(x[i].d);

    const uint8_t *GGML_RESTRICT q6 = x[i].ql;
    const uint8_t *GGML_RESTRICT qh = x[i].qh;
    const int8_t *GGML_RESTRICT q8 = y[i].qs;

    const int8_t *GGML_RESTRICT scale = x[i].scales;

    const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(y[i].bsums);
    const int8x16_t scales = vld1q_s8(scale);
    const ggml_int16x8x2_t q6scales = {
      {vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))}};

    const int32x4_t prod =
      vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]),
                                    vget_low_s16(q6scales.val[0])),
                          vmull_s16(vget_high_s16(q8sums.val[0]),
                                    vget_high_s16(q6scales.val[0]))),
                vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]),
                                    vget_low_s16(q6scales.val[1])),
                          vmull_s16(vget_high_s16(q8sums.val[1]),
                                    vget_high_s16(q6scales.val[1]))));
    int32_t isum_mins = vaddvq_s32(prod);

    int32_t isum = 0;

    for (int j = 0; j < QK_K / 128; ++j) {

      ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh);
      qh += 32;
      ggml_uint8x16x4_t q6bits = ggml_vld1q_u8_x4(q6);
      q6 += 64;
      ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(q8);
      q8 += 64;

      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
      uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 2);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b),
      // q6h.val[0])), m32s); q6bytes.val[1] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b),
      // q6h.val[1])), m32s); q6bytes.val[2] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b),
      // q6h.val[2])), m32s); q6bytes.val[3] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b),
      // q6h.val[3])), m32s);
      q6bytes.val[0] =
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
      q6bytes.val[1] =
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
      q6bytes.val[2] =
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
      q6bytes.val[3] =
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

      isum +=
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) *
          scale[0] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) *
          scale[1] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) *
          scale[2] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) *
          scale[3];

      scale += 4;

      q8bytes = ggml_vld1q_s8_x4(q8);
      q8 += 64;

      shifted = vshrq_n_u8(qhbits.val[0], 4);
      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[0], 6);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 6);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4),
      // q6h.val[0])), m32s); q6bytes.val[1] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4),
      // q6h.val[1])), m32s); q6bytes.val[2] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4),
      // q6h.val[2])), m32s); q6bytes.val[3] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4),
      // q6h.val[3])), m32s);
      q6bytes.val[0] =
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
      q6bytes.val[1] =
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
      q6bytes.val[2] =
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
      q6bytes.val[3] =
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

      isum +=
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) *
          scale[0] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) *
          scale[1] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) *
          scale[2] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) *
          scale[3];
      scale += 4;
    }
    // sum += isum * d_all * y[i].d;
    sum += d_all * y[i].d * (isum - 32 * isum_mins);
  }
  *s = sum;

#elif defined __AVX2__

  const __m256i m4 = _mm256_set1_epi8(0xF);
  const __m256i m2 = _mm256_set1_epi8(3);
  const __m256i m32s = _mm256_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {

    const float d = y[i].d * ggml_compute_fp16_to_fp32(x[i].d);

    const uint8_t *GGML_RESTRICT q4 = x[i].ql;
    const uint8_t *GGML_RESTRICT qh = x[i].qh;
    const int8_t *GGML_RESTRICT q8 = y[i].qs;

    const __m128i scales = _mm_loadu_si128((const __m128i *)x[i].scales);

    __m256i sumi = _mm256_setzero_si256();

    int is = 0;

    for (int j = 0; j < QK_K / 128; ++j) {

      const __m128i scale_0 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
      const __m128i scale_1 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
      const __m128i scale_2 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
      const __m128i scale_3 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
      is += 4;

      const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
      q4 += 32;
      const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4);
      q4 += 32;
      const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
      qh += 32;

      const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
      const __m256i q4h_1 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
      const __m256i q4h_2 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
      const __m256i q4h_3 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

      const __m256i q4_0 =
        _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
      const __m256i q4_1 =
        _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
      const __m256i q4_2 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
      const __m256i q4_3 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

      const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;

      __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
      __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
      __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
      __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

      __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

      p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

      p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
    }

    acc =
      _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

  *s = hsum_float_8(acc);
#endif
}

float __ggml_vec_dot_q6_K_q8_K(const unsigned int K,
                               const void *GGML_RESTRICT v_q6_K,
                               const void *GGML_RESTRICT v_q8_K) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ::ggml_vec_dot_q6_K_q8_K
  nntr_ggml_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_K, by, nrc);
  return result;
}

float __ggml_vec_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f) {
  // Quantization of activations
  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  ::quantize_row_q8_K(f, v_q8_activation.data(), K);

  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_activation.data());
}

float __ggml_vec_dot_q6_K(const unsigned int K,
                          const void *GGML_RESTRICT v_q6_K,
                          const float *GGML_RESTRICT activation) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ::ggml_vec_dot_q6_K_q8_K

  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  __ggml_quantize_row_q8_K(activation, v_q8_activation.data(), K);

  ::ggml_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_activation.data(),
                           by, nrc);
  return result;
}

void __ggml_gemm_q6_K(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc) {
  static constexpr const int32_t thread_count = 16;

  static constexpr const int32_t bs = 1;  // unused in ::ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t bx = 1;  // unused in ::ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t by = 1;  // unused in ::ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t nrc = 1; // unused in ::ggml_vec_dot_q6_K_q8_K

  const int32_t blocks_per_row = (K + QK_K - 1) / QK_K;
  const int32_t A_row_size = sizeof(block_q8_K) * blocks_per_row;
  const int32_t B_row_size = sizeof(block_q6_K) * blocks_per_row;

  // GEMV
  if (M == 1) {
    std::vector<char> quantized_A(A_row_size);
    ::quantize_row_q8_K(A, quantized_A.data(), K);

    const void *const quantized_A_data = quantized_A.data();

#pragma omp parallel for collapse(1) num_threads(thread_count)
    for (int32_t thread_job = 0; thread_job < static_cast<int>(N);
         thread_job++) {
      const int32_t B_row_data_offset = B_row_size * thread_job;

      const void *const B_data = (void *)((char *)B + B_row_data_offset);

      ::ggml_vec_dot_q6_K_q8_K(K, &C[thread_job], bs, B_data, bx,
                               quantized_A_data, by, nrc);
    }
  } else { // GEMM
    const int32_t A_total_size = A_row_size * M;
    std::vector<char> quantized_A(A_total_size);

#pragma omp parallel for collapse(1) num_threads(thread_count)
    for (int32_t thread_job = 0; thread_job < static_cast<int>(M);
         thread_job++) {
      const int32_t A_row_data_offset = A_row_size * thread_job;
      void *A_data = (void *)((char *)quantized_A.data() + A_row_data_offset);
      ::quantize_row_q8_K(A + thread_job * K, A_data, K);
    }
#pragma omp parallel for collapse(1) num_threads(thread_count)
    for (int32_t thread_job = 0; thread_job < static_cast<int>(M);
         thread_job++) {
      const int32_t A_row_data_offset = A_row_size * thread_job;
      void *A_data = (void *)((char *)quantized_A.data() + A_row_data_offset);

      for (uint32_t j = 0; j < N; j++) {
        const int32_t B_row_data_offset = B_row_size * j;
        const void *const B_data = (void *)((char *)B + B_row_data_offset);

        ::ggml_vec_dot_q6_K_q8_K(K, &C[thread_job * ldc + j], bs, B_data, bx,
                                 A_data, by, nrc);
      }
    }
  }
}

void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  ::dequantize_row_q4_K((const block_q4_K *)x_raw, y, k);
}

void dequantize_row_q6_K(const block_q6_K *GGML_RESTRICT x,
                         float *GGML_RESTRICT y, int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int i = 0; i < nb; i++) {
    const float d = ggml_compute_fp16_to_fp32(x[i].d);

    const uint8_t *GGML_RESTRICT ql = x[i].ql;
    const uint8_t *GGML_RESTRICT qh = x[i].qh;
    const int8_t *GGML_RESTRICT sc = x[i].scales;

    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        const int8_t q1 =
          (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 =
          (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 =
          (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 =
          (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

void __ggml_dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  dequantize_row_q6_K((const block_q6_K *)x, y, k);
}

void __ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  ::dequantize_row_q8_K((const block_q8_K *)x, y, k);
}

void __ggml_repack_q4_0_to_q4_0_8(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  ::ggml_repack_q4_0_to_q4_0_8_bl(W, 8, repacked_W, data_size, M, N);
}

void __ggml_repack_q4_K_to_q4_K_8(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  ::ggml_repack_q4_K_to_q4_K_8_bl(W, 8, repacked_W, data_size, M, N);
}

} // namespace nntrainer
