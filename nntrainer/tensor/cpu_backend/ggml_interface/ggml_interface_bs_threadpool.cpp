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
#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <bs_thread_pool_manager.hpp>
#include <cmath>
#include <ggml_interface.h>
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

const int32_t select_k_quant_thread_count(unsigned int M, unsigned int N,
                                          unsigned int K) {
  const std::size_t max_threads = std::thread::hardware_concurrency();

  const std::size_t work_size = static_cast<std::size_t>(M * N * K);

  //  Use log-scale thresholds to reduce threads on smaller work sizes
  if (work_size < 1536 * 1536)
    return 1;
  if (work_size < 1536 * 2048)
    return 2;
  if (work_size < 2048 * 2048)
    return 4;

  std::size_t est_threads =
    static_cast<std::size_t>(std::log2(work_size / (1536 * 1536))) + 4;
  return std::min(est_threads, max_threads);
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

static inline void __ggml_q4_0_8x8_q8_0_GEMM_GEMV(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
  int qa_size = sizeof(block_q8_0) * blocks_per_row;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

  ::quantize_row_q8_0(A, qa_data, K);
  ::ggml_gemv_q4_0_8x8_q8_0(K, C, ldc, B, qa_data, M, N);
}

static inline void __ggml_q4_0_8x8_q8_0_GEMM_GEMM(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  static const int32_t MULTITHREADING_METHOD_THREADS_TOTAL = 8;
  static const int32_t MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE =
    MULTITHREADING_METHOD_THREADS_TOTAL - 1;
  int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
  int M4 = ((M + 3) / 4);

  int delta = 8;
  int step_N = N / delta;
  int step_C = delta;
  int step_B = blocks_per_4_rows * sizeof(block_q4_0) * delta;

  int qa_size = qa_4_rows_size * M4;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

  auto quantize_task_size = M4 / MULTITHREADING_METHOD_THREADS_TOTAL;
  auto gemm_task_size = step_N / MULTITHREADING_METHOD_THREADS_TOTAL;

  for (auto n = 0; n < MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE; n++) {
    bs_thread_pool.detach_task([=] {
      auto start = (n)*quantize_task_size;
      auto end = (n + 1) * quantize_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K,
                                     qa_data + i * qa_4_rows_size, K);
      }
    });
  }

  for (auto n = MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE;
       n < MULTITHREADING_METHOD_THREADS_TOTAL; n++) {
    auto start = (n)*quantize_task_size;
    auto end = (n + 1) * quantize_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                   K);
    }
  }

  bs_thread_pool.wait();

  for (auto n = 0; n < MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE; n++) {
    bs_thread_pool.detach_task([=] {
      auto start = (n)*gemm_task_size;
      auto end = (n + 1) * gemm_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc,
                                  (char *)B + i * step_B, qa_data, M, delta);
      }
    });
  }

  for (auto n = MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE;
       n < MULTITHREADING_METHOD_THREADS_TOTAL; n++) {
    auto start = (n)*gemm_task_size;
    auto end = (n + 1) * gemm_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc, (char *)B + i * step_B,
                                qa_data, M, delta);
    }
  }

  bs_thread_pool.wait();
}

void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) { // GEMV
    __ggml_q4_0_8x8_q8_0_GEMM_GEMV(M, N, K, A, lda, B, ldb, C, ldc);
  } else { // GEMM
    __ggml_q4_0_8x8_q8_0_GEMM_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
  }
}

static inline void __ggml_q4_K_8x8_q8_K_GEMM_GEMV(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int qa_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

  ::quantize_row_q8_K(A, qa_data, K);
  ::ggml_gemv_q4_K_8x8_q8_K(K, C, ldc, B, qa_data, M, N);
}

static inline void __ggml_q4_K_8x8_q8_K_GEMM_GEMM(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  static const int32_t MULTITHREADING_METHOD_THREADS_TOTAL = 8;
  static const int32_t MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE =
    MULTITHREADING_METHOD_THREADS_TOTAL - 1;

  int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
  int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
  int M4 = ((M + 3) / 4);

  int delta = 8;
  int step_N = N / delta;
  int step_C = delta;
  int step_B = blocks_per_4_rows * sizeof(block_q4_K) * delta;

  int qa_size = qa_4_rows_size * M4;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

  auto quantize_task_size = M4 / MULTITHREADING_METHOD_THREADS_TOTAL;
  auto gemm_task_size = step_N / MULTITHREADING_METHOD_THREADS_TOTAL;

  for (auto n = 0; n < MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE; n++) {
    bs_thread_pool.detach_task([=] {
      auto start = (n)*quantize_task_size;
      auto end = (n + 1) * quantize_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K,
                                     qa_data + i * qa_4_rows_size, K);
      }
    });
  }

  for (auto n = MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE;
       n < MULTITHREADING_METHOD_THREADS_TOTAL; n++) {
    auto start = (n)*quantize_task_size;
    auto end = (n + 1) * quantize_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, qa_data + i * qa_4_rows_size,
                                   K);
    }
  }

  bs_thread_pool.wait();

  for (auto n = 0; n < MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE; n++) {
    bs_thread_pool.detach_task([=] {
      auto start = (n)*gemm_task_size;
      auto end = (n + 1) * gemm_task_size;

      for (auto i = start; i < end; ++i) {
        ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc,
                                  (char *)B + i * step_B, qa_data, M, delta);
      }
    });
  }

  for (auto n = MULTITHREADING_METHOD_THREADS_TOTAL_MINUS_ONE;
       n < MULTITHREADING_METHOD_THREADS_TOTAL; n++) {
    auto start = (n)*gemm_task_size;
    auto end = (n + 1) * gemm_task_size;

    for (auto i = start; i < end; ++i) {
      ::ggml_gemm_q4_K_8x8_q8_K(K, C + i * step_C, ldc, (char *)B + i * step_B,
                                qa_data, M, delta);
    }
  }

  bs_thread_pool.wait();
}

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) { // GEMV
    __ggml_q4_K_8x8_q8_K_GEMM_GEMV(M, N, K, A, lda, B, ldb, C, ldc);
  } else { // GEMM
    __ggml_q4_K_8x8_q8_K_GEMM_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
  }
}

float __ggml_vec_dot_q6_K_q8_K(const unsigned int K,
                               const void *GGML_RESTRICT v_q6_K,
                               const void *GGML_RESTRICT v_q8_K) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ::ggml_vec_dot_q6_K_q8_K
  ::ggml_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_K, by, nrc);
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

void __ggml_gemm_q6_K(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc) {
  static constexpr const int32_t thread_count = 4;

  static constexpr const int32_t bs = 1;  // unused in ::ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t bx = 1;  // unused in ::ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t by = 1;  // unused in ::ggml_vec_dot_q6_K_q8_K
  static constexpr const int32_t nrc = 1; // unused in ::ggml_vec_dot_q6_K_q8_K

  const int32_t blocks_per_row = (K + QK_K - 1) / QK_K;
  const int32_t A_row_size = sizeof(block_q8_K) * blocks_per_row;
  const int32_t B_row_size = sizeof(block_q6_K) * blocks_per_row;

  const int32_t A_total_size = A_row_size * M;
  std::vector<char> quantized_A(A_total_size);
  void *const quantized_A_data = quantized_A.data();

#pragma omp parallel for collapse(1) num_threads(thread_count)
  for (int32_t thread_job = 0; thread_job < static_cast<int>(M); thread_job++) {
    const int32_t A_row_data_offset = A_row_size * thread_job;
    void *A_data = (void *)((char *)quantized_A_data + A_row_data_offset);
    ::quantize_row_q8_K(A + thread_job * K, A_data, K);
  }

#pragma omp parallel for collapse(2) num_threads(thread_count)
  for (int32_t thread_job = 0; thread_job < static_cast<int>(M); thread_job++) {
    for (int32_t j = 0; j < static_cast<int>(N); j++) {
      const int32_t A_row_data_offset = A_row_size * thread_job;
      void *A_data = (void *)((char *)quantized_A_data + A_row_data_offset);

      const int32_t B_row_data_offset = B_row_size * j;
      const void *const B_data = (void *)((char *)B + B_row_data_offset);

      ::ggml_vec_dot_q6_K_q8_K(K, &C[thread_job * ldc + j], bs, B_data, bx,
                               A_data, by, nrc);
    }
  }
}

void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  ::dequantize_row_q4_K((const block_q4_K *)x_raw, y, k);
}

void __ggml_dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  ::dequantize_row_q6_K((const block_q6_K *)x, y, k);
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
