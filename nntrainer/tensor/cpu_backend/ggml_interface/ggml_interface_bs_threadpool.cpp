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

#include <algorithm>
#include <bs_thread_pool_manager.hpp>
#include <cmath>
#include <ggml_interface.h>
#include <nntr_ggml_impl.h>
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
  return quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

void __ggml_quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  quantize_q6_K(src, dst, 1, k, nullptr);
}

template <>
void __ggml_quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  quantize_row_q8_K(src, dst, k);
}

static inline void __ggml_q4_0_4x8_q8_0_GEMM_GEMV(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int NB_COLS = 4;
  int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
  int qa_size = sizeof(block_q8_0) * blocks_per_row;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

  quantize_row_q8_0(A, qa_data, K);
  int B_step = sizeof(block_q4_0) * (K / QK4_0);

  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  int thread_num = bs_thread_pool.get_thread_count();
  BS::multi_future<void> loop_future =
    bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
      unsigned int M_step_start = (i * N) / thread_num;
      unsigned int M_step_end = ((i + 1) * N) / thread_num;

      M_step_start = (M_step_start % NB_COLS)
                       ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                       : M_step_start;
      M_step_end = (M_step_end % NB_COLS)
                     ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                     : M_step_end;

      nntr_gemv_q4_0_4x8_q8_0(K, (float *)(C + M_step_start), N,
                              (void *)((char *)B + M_step_start * B_step),
                              QA.data(), M, M_step_end - M_step_start);
    });
  loop_future.wait();
}

static inline void __ggml_q4_0_4x8_q8_0_GEMM_GEMM(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int NB_COLS = 4;
  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
  const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;
  unsigned int M4 = ((M - M % 4) / 4);
  int B_step = sizeof(block_q4_0) * (K / QK4_0);

  unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
  std::vector<char> QA = std::vector<char>(qa_size);

  // Quantize 4-divisible-M row portion with matrix-wise function
  for (unsigned int i = 0; i < M4; i++) {
    ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                               K);
  }
  // Quantize leftover 1 ~ 3 rows with row-wise function
  for (unsigned int i = M4 * 4; i < M; i++) {
    quantize_row_q8_0(
      (float *)A + i * K,
      (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
  }

  ///@todo Dynamic thread-number selection for GEMM problem size
  int thread_num = bs_thread_pool.get_thread_count();
  BS::multi_future<void> multi_future =
    bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
      unsigned int M_step_start = (i * N) / thread_num;
      unsigned int M_step_end = ((i + 1) * N) / thread_num;

      M_step_start = (M_step_start % NB_COLS)
                       ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                       : M_step_start;
      M_step_end = (M_step_end % NB_COLS)
                     ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                     : M_step_end;

      nntr_gemm_q4_0_4x8_q8_0(K, (C + (M_step_start)), ldc,
                              ((char *)B + ((M_step_start)*B_step)), QA.data(),
                              M4 * 4, (M_step_end) - (M_step_start));
    });
  multi_future.wait();

  for (unsigned int pb = M4 * 4; pb < M; pb++) {
    BS::multi_future<void> loop_future =
      bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
        unsigned int M_step_start = (i * N) / thread_num;
        unsigned int M_step_end = ((i + 1) * N) / thread_num;

        M_step_start = (M_step_start % NB_COLS)
                         ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                         : M_step_start;
        M_step_end = (M_step_end % NB_COLS)
                       ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                       : M_step_end;

        nntr_gemv_q4_0_4x8_q8_0(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    loop_future.wait();
  }
}

template <>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  if (M == 1) { // GEMV
    __ggml_q4_0_4x8_q8_0_GEMM_GEMV(M, N, K, A, lda, B, ldb, C, ldc);
  } else { // GEMM
    __ggml_q4_0_4x8_q8_0_GEMM_GEMM(M, N, K, A, lda, B, ldb, C, ldc);
  }
}

template <typename T = float>
void __ggml_q4_0_4x8_q8_0_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const T *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<T *> C,
                               std::vector<unsigned int> ldcs) {
  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  int thread_num = bs_thread_pool.get_thread_count();

  int NB_COLS = 4;
  int B_step = sizeof(block_q4_0) * (K / QK4_0);
  int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;

  if (M == 1) {
    int qa_size = sizeof(block_q8_0) * blocks_per_4_rows;
    std::vector<char> QA = std::vector<char>(qa_size);
    auto qa_data = QA.data();
    quantize_row_q8_0(A, qa_data, K);
    if (std::all_of(Ns.begin(), Ns.end(),
                    [](unsigned int n) { return n <= 256; })) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];

        unsigned int M_step_start = 0;
        unsigned int M_step_end = N;
        M_step_start = (M_step_start % NB_COLS)
                         ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                         : M_step_start;
        M_step_end = (M_step_end % NB_COLS)
                       ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                       : M_step_end;

        nntr_gemv_q4_0_4x8_q8_0(K, (float *)(C + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA.data(), M, M_step_end - M_step_start);
      }
    } else {
      BS::multi_future<void> loop_future =
        bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
          for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
            unsigned int N = Ns[num_w];
            float *C = Cs[num_w];
            void *B = Bs[num_w];
            unsigned int M_step_start = (i * N) / thread_num;
            unsigned int M_step_end = ((i + 1) * N) / thread_num;

            M_step_start = (M_step_start % NB_COLS)
                             ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                             : M_step_start;
            M_step_end = (M_step_end % NB_COLS)
                           ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                           : M_step_end;

            nntr_gemv_q4_0_4x8_q8_0(K, (float *)(C + M_step_start), N,
                                    (void *)((char *)B + M_step_start * B_step),
                                    QA.data(), M, M_step_end - M_step_start);
          }
        });
      loop_future.wait();
    }
  } else {
    int n_threads = std::thread::hardware_concurrency() / 2;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;

    unsigned int M4 = ((M - M % 4) / 4);
    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);

    std::vector<char> QA = std::vector<char>(qa_size);

    for (unsigned int i = 0; i < M4; i++) {
      ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
    }

    for (unsigned int i = M4 * 4; i < M; i++) {
      quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

#pragma omp parallel for schedule(guided) num_threads(n_threads)
    for (int i = 0; i < n_threads; i++) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        unsigned int ldc = ldcs[num_w];

        float *C = Cs[num_w];
        void *B = Bs[num_w];

        unsigned int src0_start = (i * N) / n_threads;
        unsigned int src0_end = ((i + 1) * N) / n_threads;

        src0_start = (src0_start % NB_COLS)
                       ? src0_start + NB_COLS - (src0_start % NB_COLS)
                       : src0_start;

        src0_end = (src0_end % NB_COLS)
                     ? src0_end + NB_COLS - (src0_end % NB_COLS)
                     : src0_end;

        nntr_gemm_q4_0_4x8_q8_0(K, (float *)(C + src0_start), ldc,
                                (void *)((char *)B + src0_start * B_step),
                                QA.data(), M4 * 4, src0_end - src0_start);
      }
    }

    n_threads = 4;
#pragma omp parallel for schedule(guided) num_threads(n_threads)
    for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        unsigned int ldc = ldcs[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];

        for (int pb = M4 * 4; pb < static_cast<int>(M); pb++) {
          unsigned int M_step_start = (thread_idx * N) / n_threads;
          unsigned int M_step_end = ((thread_idx + 1) * N) / n_threads;
          M_step_start = (M_step_start % NB_COLS)
                           ? M_step_start + NB_COLS - (M_step_start % NB_COLS)
                           : M_step_start;
          M_step_end = (M_step_end % NB_COLS)
                         ? M_step_end + NB_COLS - (M_step_end % NB_COLS)
                         : M_step_end;

          nntr_gemv_q4_0_4x8_q8_0(
            K,
            (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
            N, (void *)((char *)B + M_step_start * B_step),
            QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
            M_step_end - M_step_start);
        }
      }
    }
  }
}

static inline void __ggml_q4_0_8x8_q8_0_GEMM_GEMV(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
  int qa_size = sizeof(block_q8_0) * blocks_per_row;
  std::vector<char> QA = std::vector<char>(qa_size);

  auto qa_data = QA.data();

  quantize_row_q8_0(A, qa_data, K);
  int B_step = sizeof(block_q4_0) * (K / QK4_0);

  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  int thread_num = bs_thread_pool.get_thread_count();
  BS::multi_future<void> loop_future =
    bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
      unsigned int M_step_start = (i * N) / thread_num;
      unsigned int M_step_end = ((i + 1) * N) / thread_num;

      M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8)
                                        : M_step_start;
      M_step_end =
        (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

      ggml_gemv_q4_0_8x8_q8_0(K, (float *)(C + M_step_start), N,
                              (void *)((char *)B + M_step_start * B_step),
                              QA.data(), M, M_step_end - M_step_start);
    });
  loop_future.wait();
}

static inline void __ggml_q4_0_8x8_q8_0_GEMM_GEMM(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  auto &bs_thread_pool = ThreadPoolManager::Global().getThreadPool();
  unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
  unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
  const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;
  unsigned int M4 = ((M - M % 4) / 4);
  int B_step = sizeof(block_q4_0) * (K / QK4_0);
  unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
  std::vector<char> QA = std::vector<char>(qa_size);

  // Quantize 4-divisible-M row portion with matrix-wise function
  for (unsigned int i = 0; i < M4; i++) {
    ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                               K);
  }
  // Quantize leftover 1 ~ 3 rows with row-wise function
  for (unsigned int i = M4 * 4; i < M; i++) {
    quantize_row_q8_0(
      (float *)A + i * K,
      (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
  }

  ///@todo Dynamic thread-number selection for GEMM problem size
  int thread_num = bs_thread_pool.get_thread_count();
  BS::multi_future<void> multi_future =
    bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
      unsigned int M_step_start = (i * N) / thread_num;
      unsigned int M_step_end = ((i + 1) * N) / thread_num;

      M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8)
                                        : M_step_start;
      M_step_end =
        (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

      ggml_gemm_q4_0_8x8_q8_0(K, (C + (M_step_start)), ldc,
                              ((char *)B + ((M_step_start)*B_step)), QA.data(),
                              M4 * 4, (M_step_end) - (M_step_start));
    });
  multi_future.wait();

  for (unsigned int pb = M4 * 4; pb < M; pb++) {
    BS::multi_future<void> loop_future =
      bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
        unsigned int M_step_start = (i * N) / thread_num;
        unsigned int M_step_end = ((i + 1) * N) / thread_num;

        M_step_start = (M_step_start % 8)
                         ? M_step_start + 8 - (M_step_start % 8)
                         : M_step_start;
        M_step_end =
          (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

        ggml_gemv_q4_0_8x8_q8_0(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    loop_future.wait();
  }
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

template <typename T = float>
void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const T *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<T *> C,
                               std::vector<unsigned int> ldcs) {
  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  int thread_num = bs_thread_pool.get_thread_count();

  int B_step = sizeof(block_q4_0) * (K / QK4_0);
  int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;

  if (M == 1) {
    int qa_size = sizeof(block_q8_0) * blocks_per_4_rows;
    std::vector<char> QA = std::vector<char>(qa_size);
    auto qa_data = QA.data();
    quantize_row_q8_0(A, qa_data, K);

    for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
      unsigned int N = Ns[num_w];
      float *C = Cs[num_w];
      void *B = Bs[num_w];

      if (N <= 256) {
        unsigned int M_step_start = 0;
        unsigned int M_step_end = N;
        M_step_start = (M_step_start % 8)
                         ? M_step_start + 8 - (M_step_start % 8)
                         : M_step_start;
        M_step_end =
          (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

        ggml_gemv_q4_0_8x8_q8_0(K, (float *)(C + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA.data(), M, M_step_end - M_step_start);
      }
    }

    BS::multi_future<void> loop_future =
      bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
        for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
          unsigned int N = Ns[num_w];
          float *C = Cs[num_w];
          void *B = Bs[num_w];
          unsigned int M_step_start = (i * N) / thread_num;
          unsigned int M_step_end = ((i + 1) * N) / thread_num;

          M_step_start = (M_step_start % 8)
                           ? M_step_start + 8 - (M_step_start % 8)
                           : M_step_start;
          M_step_end =
            (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

          ggml_gemv_q4_0_8x8_q8_0(K, (float *)(C + M_step_start), N,
                                  (void *)((char *)B + M_step_start * B_step),
                                  QA.data(), M, M_step_end - M_step_start);
        }
      });
    loop_future.wait();
  } else {
    int n_threads = std::thread::hardware_concurrency() / 2;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_0) * K) / QK8_0;

    unsigned int M4 = ((M - M % 4) / 4);
    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);

    std::vector<char> QA = std::vector<char>(qa_size);

    for (unsigned int i = 0; i < M4; i++) {
      ggml_quantize_mat_q8_0_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
    }

    for (unsigned int i = M4 * 4; i < M; i++) {
      quantize_row_q8_0(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

#pragma omp parallel for schedule(guided) num_threads(n_threads)
    for (int i = 0; i < n_threads; i++) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        unsigned int ldc = ldcs[num_w];

        float *C = Cs[num_w];
        void *B = Bs[num_w];

        unsigned int src0_start = (i * N) / n_threads;
        unsigned int src0_end = ((i + 1) * N) / n_threads;

        src0_start =
          (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;

        src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

        ggml_gemm_q4_0_8x8_q8_0(K, (float *)(C + src0_start), ldc,
                                (void *)((char *)B + src0_start * B_step),
                                QA.data(), M4 * 4, src0_end - src0_start);
      }
    }

    n_threads = 4;
#pragma omp parallel for schedule(guided) num_threads(n_threads)
    for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        unsigned int ldc = ldcs[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];

        for (int pb = M4 * 4; pb < static_cast<int>(M); pb++) {
          unsigned int M_step_start = (thread_idx * N) / n_threads;
          unsigned int M_step_end = ((thread_idx + 1) * N) / n_threads;
          M_step_start = (M_step_start % 8)
                           ? M_step_start + 8 - (M_step_start % 8)
                           : M_step_start;
          M_step_end =
            (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

          ggml_gemv_q4_0_8x8_q8_0(
            K,
            (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
            N, (void *)((char *)B + M_step_start * B_step),
            QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
            M_step_end - M_step_start);
        }
      }
    }
  }
}

static inline void __ggml_q4_K_8x8_q8_K_GEMM_GEMV(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  int B_step = sizeof(block_q4_K) * (K / QK_K);
  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int qa_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> QA = std::vector<char>(qa_size);
  auto qa_data = QA.data();
  quantize_row_q8_K(A, qa_data, K);

  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  int thread_num = bs_thread_pool.get_thread_count();
  BS::multi_future<void> loop_future =
    bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
      unsigned int M_step_start = (i * N) / thread_num;
      unsigned int M_step_end = ((i + 1) * N) / thread_num;

      M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8)
                                        : M_step_start;
      M_step_end =
        (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

      ggml_gemv_q4_K_8x8_q8_K(K, (float *)(C + M_step_start), N,
                              (void *)((char *)B + M_step_start * B_step),
                              QA.data(), M, M_step_end - M_step_start);
    });
  loop_future.wait();
}

static inline void __ggml_q4_K_8x8_q8_K_GEMM_GEMM(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const float *A, const unsigned int lda, const void *B, const unsigned int ldb,
  float *C, const unsigned int ldc) {
  auto &bs_thread_pool = ThreadPoolManager::Global().getThreadPool();
  unsigned int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
  unsigned int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
  const size_t qa_row_size = (sizeof(block_q8_K) * K) / QK_K;
  unsigned int M4 = ((M - M % 4) / 4);
  int B_step = sizeof(block_q4_K) * (K / QK_K);
  unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
  std::vector<char> QA = std::vector<char>(qa_size);

  // Quantize 4-divisible-M row portion with matrix-wise function
  for (unsigned int i = 0; i < M4; i++) {
    ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                               K);
  }
  // Quantize leftover 1 ~ 3 rows with row-wise function
  for (unsigned int i = M4 * 4; i < M; i++) {
    quantize_row_q8_K(
      (float *)A + i * K,
      (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
  }

  ///@todo Dynamic thread-number selection for GEMM problem size
  int thread_num = bs_thread_pool.get_thread_count();
  BS::multi_future<void> multi_future =
    bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
      unsigned int M_step_start = (i * N) / thread_num;
      unsigned int M_step_end = ((i + 1) * N) / thread_num;

      M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8)
                                        : M_step_start;
      M_step_end =
        (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

      ggml_gemm_q4_K_8x8_q8_K(K, (C + (M_step_start)), ldc,
                              ((char *)B + ((M_step_start)*B_step)), QA.data(),
                              M4 * 4, (M_step_end) - (M_step_start));
    });
  multi_future.wait();

  for (unsigned int pb = M4 * 4; pb < M; pb++) {
    BS::multi_future<void> loop_future =
      bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
        unsigned int M_step_start = (i * N) / thread_num;
        unsigned int M_step_end = ((i + 1) * N) / thread_num;

        M_step_start = (M_step_start % 8)
                         ? M_step_start + 8 - (M_step_start % 8)
                         : M_step_start;
        M_step_end =
          (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

        ggml_gemv_q4_K_8x8_q8_K(
          K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
          N, (void *)((char *)B + M_step_start * B_step),
          QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    loop_future.wait();
  }
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

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M,
                               std::vector<unsigned int> Ns,
                               const unsigned int K, const float *A,
                               const unsigned int lda, std::vector<void *> Bs,
                               std::vector<unsigned int> ldbs,
                               std::vector<float *> C,
                               std::vector<unsigned int> ldcs) {

  auto &bs_thread_pool = ThreadPoolManager::getInstance();
  int thread_num = bs_thread_pool.get_thread_count();

  int B_step = sizeof(block_q4_K) * (K / QK_K);
  int blocks_per_4_rows = (K + QK_K - 1) / QK_K;

  if (M == 1) {
    int qa_size = sizeof(block_q8_K) * blocks_per_4_rows;
    std::vector<char> QA = std::vector<char>(qa_size);
    auto qa_data = QA.data();
    quantize_row_q8_K(A, qa_data, K);
    if (std::all_of(Ns.begin(), Ns.end(),
                    [](unsigned int n) { return n <= 256; })) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];

        unsigned int M_step_start = 0;
        unsigned int M_step_end = N;
        M_step_start = (M_step_start % 8)
                         ? M_step_start + 8 - (M_step_start % 8)
                         : M_step_start;
        M_step_end =
          (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

        ggml_gemv_q4_K_8x8_q8_K(K, (float *)(C + M_step_start), N,
                                (void *)((char *)B + M_step_start * B_step),
                                QA.data(), M, M_step_end - M_step_start);
      }
    } else {
      BS::multi_future<void> loop_future =
        bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
          for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
            unsigned int N = Ns[num_w];
            float *C = Cs[num_w];
            void *B = Bs[num_w];
            unsigned int M_step_start = (i * N) / thread_num;
            unsigned int M_step_end = ((i + 1) * N) / thread_num;

            M_step_start = (M_step_start % 8)
                             ? M_step_start + 8 - (M_step_start % 8)
                             : M_step_start;
            M_step_end =
              (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

            ggml_gemv_q4_K_8x8_q8_K(K, (float *)(C + M_step_start), N,
                                    (void *)((char *)B + M_step_start * B_step),
                                    QA.data(), M, M_step_end - M_step_start);
          }
        });
      loop_future.wait();
    }
  } else {

    int n_threads = std::thread::hardware_concurrency() / 2;
    unsigned int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_K) * K) / QK_K;

    unsigned int M4 = ((M - M % 4) / 4);
    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);

    std::vector<char> QA = std::vector<char>(qa_size);

    for (unsigned int i = 0; i < M4; i++) {
      ggml_quantize_mat_q8_K_4x8(A + 4 * i * K, QA.data() + i * qa_4_rows_size,
                                 K);
    }

    for (unsigned int i = M4 * 4; i < M; i++) {
      quantize_row_q8_K(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

#pragma omp parallel for schedule(guided) num_threads(n_threads)
    for (int i = 0; i < n_threads; i++) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        unsigned int ldc = ldcs[num_w];

        float *C = Cs[num_w];
        void *B = Bs[num_w];

        unsigned int src0_start = (i * N) / n_threads;
        unsigned int src0_end = ((i + 1) * N) / n_threads;

        src0_start =
          (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;

        src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

        ggml_gemm_q4_K_8x8_q8_K(K, (float *)(C + src0_start), ldc,
                                (void *)((char *)B + src0_start * B_step),
                                QA.data(), M4 * 4, src0_end - src0_start);
      }
    }

    n_threads = 4;
#pragma omp parallel for schedule(guided) num_threads(n_threads)
    for (int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      for (unsigned int num_w = 0; num_w < Ns.size(); ++num_w) {
        unsigned int N = Ns[num_w];
        unsigned int ldc = ldcs[num_w];
        float *C = Cs[num_w];
        void *B = Bs[num_w];

        for (int pb = M4 * 4; pb < static_cast<int>(M); pb++) {
          unsigned int M_step_start = (thread_idx * N) / n_threads;
          unsigned int M_step_end = ((thread_idx + 1) * N) / n_threads;
          M_step_start = (M_step_start % 8)
                           ? M_step_start + 8 - (M_step_start % 8)
                           : M_step_start;
          M_step_end =
            (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

          ggml_gemv_q4_K_8x8_q8_K(
            K,
            (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
            N, (void *)((char *)B + M_step_start * B_step),
            QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
            M_step_end - M_step_start);
        }
      }
    }
  }
}

float __ggml_vec_dot_q6_K_q8_K(const unsigned int K,
                               const void *GGML_RESTRICT v_q6_K,
                               const void *GGML_RESTRICT v_q8_K) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ggml_vec_dot_q6_K_q8_K
  ggml_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_K, by, nrc);
  return result;
}

float __ggml_vec_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f) {
  // Quantization of activations
  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  quantize_row_q8_K(f, v_q8_activation.data(), K);

  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_activation.data());
}

float __ggml_vec_dot_q6_K(const unsigned int K,
                          const void *GGML_RESTRICT v_q6_K,
                          const float *GGML_RESTRICT activation) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ggml_vec_dot_q6_K_q8_K

  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  __ggml_quantize_row_q8_K(activation, v_q8_activation.data(), K);

  ggml_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_activation.data(), by,
                         nrc);
  return result;
}

template <>
void __ggml_gemm_q6_K(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C,
                      const unsigned int ldc) {
  static constexpr const int32_t bs = 1;
  static constexpr const int32_t bx = 1;
  static constexpr const int32_t by = 1;
  static constexpr const int32_t nrc = 1;

  const int32_t blocks_per_row = (K + QK_K - 1) / QK_K;
  const int32_t A_row_size = sizeof(block_q8_K) * blocks_per_row;
  const int32_t B_row_size = sizeof(block_q6_K) * blocks_per_row;

  auto &tp = ThreadPoolManager::getInstance();
  if (M == 1) {
    std::vector<char> quantized_A(A_row_size);
    quantize_row_q8_K(A, quantized_A.data(), K);
    const void *quantized_A_data = quantized_A.data();

    auto fut = tp.submit_loop(0, static_cast<int>(N), [&](int i) {
      const void *bptr = (const char *)B + i * B_row_size;
      ggml_vec_dot_q6_K_q8_K(K, &C[i], bs, bptr, bx, quantized_A_data, by, nrc);
    });
    fut.wait();
  } else {
    const int32_t A_total_size = A_row_size * static_cast<int32_t>(M);
    std::vector<char> quantized_A(A_total_size);

    for (int i = 0; i < static_cast<int>(M); ++i) {
      void *row_ptr = quantized_A.data() + i * A_row_size;
      quantize_row_q8_K(A + i * K, row_ptr, K);
    }

    auto fut = tp.submit_loop(0, static_cast<int>(M), [&](int i) {
      const void *a_row = quantized_A.data() + i * A_row_size;
      float *c_row = C + i * ldc;
      for (unsigned int j = 0; j < N; ++j) {
        const void *bptr = (const char *)B + j * B_row_size;
        ggml_vec_dot_q6_K_q8_K(K, &c_row[j], bs, bptr, bx, a_row, by, nrc);
      }
    });
    fut.wait();
  }
}

void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  ::dequantize_row_q4_K((const block_q4_K *)x_raw, y, k);
}

void __ggml_dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  ::dequantize_row_q6_K((const block_q6_K *)x, y, k);
}

template <>
void __ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  ::dequantize_row_q8_K((const block_q8_K *)x, y, k);
}

void __ggml_repack_q4_0_to_q4_0_4(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  ggml_repack_q4_0_to_q4_0_4_bl(W, 8, repacked_W, data_size, M, N);
}

void __ggml_repack_q4_0_to_q4_0_8(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  ggml_repack_q4_0_to_q4_0_8_bl(W, 8, repacked_W, data_size, M, N);
}

void __ggml_repack_q4_K_to_q4_K_8(void *W, void *repacked_W, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  ggml_repack_q4_K_to_q4_K_8_bl(W, 8, repacked_W, data_size, M, N);
}

} // namespace nntrainer
