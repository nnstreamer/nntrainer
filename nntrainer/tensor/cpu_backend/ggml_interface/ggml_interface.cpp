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

#include <bs_thread_pool_manager.hpp>
#include <ggml_interface.h>
#include "ggml_simd_quant.h"
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace nntrainer {

/**
 * @brief High-performance memory pool for quantization buffers
 */
class QuantizationBufferPool {
private:
  std::unordered_map<size_t, std::vector<std::unique_ptr<char[]>>> buffers_;
  std::mutex mutex_;
  
  static constexpr size_t CACHE_LINE_SIZE = 64;
  static constexpr size_t MAX_CACHED_BUFFERS = 8;

public:
  static QuantizationBufferPool& getInstance() {
    static QuantizationBufferPool instance;
    return instance;
  }

  std::unique_ptr<char[]> getBuffer(size_t size) {
    // Align size to cache line boundary for optimal performance
    size_t aligned_size = (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto& pool = buffers_[aligned_size];
    
    if (!pool.empty()) {
      auto buffer = std::move(pool.back());
      pool.pop_back();
      return buffer;
    }
    
    return std::make_unique<char[]>(aligned_size);
  }

  void returnBuffer(std::unique_ptr<char[]> buffer, size_t size) {
    if (!buffer) return;
    
    size_t aligned_size = (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto& pool = buffers_[aligned_size];
    
    if (pool.size() < MAX_CACHED_BUFFERS) {
      pool.emplace_back(std::move(buffer));
    }
    // If pool is full, buffer will be automatically destroyed
  }
};

/**
 * @brief RAII wrapper for pooled buffers
 */
class PooledBuffer {
private:
  std::unique_ptr<char[]> buffer_;
  size_t size_;
  QuantizationBufferPool& pool_;

public:
  PooledBuffer(size_t size) : size_(size), pool_(QuantizationBufferPool::getInstance()) {
    buffer_ = pool_.getBuffer(size);
  }

  ~PooledBuffer() {
    pool_.returnBuffer(std::move(buffer_), size_);
  }

  char* data() { return buffer_.get(); }
  const char* data() const { return buffer_.get(); }
  size_t size() const { return size_; }

  // Non-copyable, movable
  PooledBuffer(const PooledBuffer&) = delete;
  PooledBuffer& operator=(const PooledBuffer&) = delete;
  PooledBuffer(PooledBuffer&&) = default;
  PooledBuffer& operator=(PooledBuffer&&) = default;
};

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
  simd::quantize_row_q8_K_optimized(src, dst, k);
}

void __ggml_q4_0_8x8_q8_0_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  auto &bspool = ThreadPoolManager::getInstance();

  if (M == 1) { // GEMV
    // Adaptive thread count based on problem size
    const unsigned int min_work_per_thread = 64;
    const unsigned int n_threads = std::min(4u, std::max(1u, N / min_work_per_thread));
    unsigned int B_step = sizeof(block_q4_0) * (K / QK4_0);
    unsigned int blocks_per_row = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_size = sizeof(block_q8_0) * blocks_per_row;
    PooledBuffer QA(qa_size);
    ::quantize_row_q8_0(A, QA.data(), K);

    // Use BS thread pool for parallel GEMV
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);
    
    for (unsigned int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      futures.emplace_back(bspool.submit_task([=, &QA, &B, &C]() {
        unsigned int M_step_start = (thread_idx * N) / n_threads;
        unsigned int M_step_end = ((thread_idx + 1) * N) / n_threads;
        
        // Cache-line alignment for better performance
        M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8) : M_step_start;
        M_step_end = (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;
        
        ::ggml_gemv_q4_0_8x8_q8_0(K, (float *)((C) + M_step_start), N,
                                  (void *)((char *)B + M_step_start * B_step),
                                  QA.data(), M, M_step_end - M_step_start);
      }));
    }
    
    // Wait for all tasks to complete
    for (auto &future : futures) {
      future.wait();
    }
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK8_0 - 1) / QK8_0;
    unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * blocks_per_4_rows;
    unsigned int M4 = ((M + 3) / 4);

    unsigned int qa_size = qa_4_rows_size * M4;
    PooledBuffer QA(qa_size);

    // Quantization of activations
    /// @note Heuristic inspection conducted that applying multithreading on
    /// run-time quantization hurts model latency
    for (int i = 0; i < static_cast<int>(M4); i++) {
      ::ggml_quantize_mat_q8_0_4x8(A + 4 * i * K,
                                   QA.data() + i * qa_4_rows_size, K);
    }
    int delta = 8;
    int step_N = N / delta;
    int step_C = delta;
    int step_B = blocks_per_4_rows * sizeof(block_q4_0) * delta;
    
    // Adaptive thread count for GEMM
    const unsigned int optimal_threads = std::min(16u, 
      std::max(1u, static_cast<unsigned int>(step_N)));
    
    // Use BS thread pool for parallel GEMM
    BS::multi_future<void> multi_future = bspool.submit_loop(0, step_N, 
      [&](int i) {
        ::ggml_gemm_q4_0_8x8_q8_0(K, C + i * step_C, ldc, 
                                  (char *)B + i * step_B,
                                  QA.data(), M, delta);
      });
    multi_future.wait();
  }
}

void __ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const float *A,
                               const unsigned int lda, const void *B,
                               const unsigned int ldb, float *C,
                               const unsigned int ldc) {
  auto &bspool = ThreadPoolManager::getInstance();
  
  if (M == 1) { // GEMV
    const unsigned int n_threads = std::min(4u, std::max(1u, N / 64));
    unsigned int blocks_per_row = (K + QK_K - 1) / QK_K;
    unsigned int qa_size = sizeof(block_q8_K) * blocks_per_row;
    unsigned int B_step = sizeof(block_q4_K) * (K / QK_K);

    PooledBuffer QA(qa_size);

    simd::quantize_row_q8_K_optimized(A, QA.data(), K);

    // Use BS thread pool for parallel GEMV  
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);
    
    for (unsigned int thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      futures.emplace_back(bspool.submit_task([=, &QA, &B, &C]() {
        unsigned int M_step_start = (thread_idx * N) / n_threads;
        unsigned int M_step_end = ((thread_idx + 1) * N) / n_threads;
        
        M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8) : M_step_start;
        M_step_end = (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;
        
        ::ggml_gemv_q4_K_8x8_q8_K(K, (float *)((C) + M_step_start), N,
                                  (void *)((char *)B + M_step_start * B_step),
                                  QA.data(), M, M_step_end - M_step_start);
      }));
    }
    
    for (auto &future : futures) {
      future.wait();
    }
  } else if (M % 4 != 0) {
    const unsigned int n_threads = std::min(
      static_cast<unsigned int>(std::thread::hardware_concurrency()), 
      std::max(1u, N / 32));
    unsigned int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
    unsigned int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    const size_t qa_row_size = (sizeof(block_q8_K) * K) / QK_K;
    unsigned int M4 = ((M - M % 4) / 4);
    int B_step = sizeof(block_q4_K) * (K / QK_K);

    unsigned int qa_size = qa_4_rows_size * (((M >> 2) << 2) / 4 + 1);
    PooledBuffer QA(qa_size);

    // Quantize 4-divisible-M row portion with matrix-wise function
    for (unsigned int i = 0; i < M4; i++) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K,
                                   QA.data() + i * qa_4_rows_size, K);
    }
    // Quantize leftover 1 ~ 3 rows with row-wise function
    for (unsigned int i = M4 * 4; i < M; i++) {
      simd::quantize_row_q8_K_optimized(
        (float *)A + i * K,
        (QA.data() + (M4 * qa_4_rows_size) + (i - M4 * 4) * qa_row_size), K);
    }

    // Use BS thread pool for parallel execution
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);
    
    for (int i = 0; i < n_threads; i++) {
      futures.emplace_back(bspool.submit_task([=, &QA, &B, &C]() {
        unsigned int src0_start = (i * N) / n_threads;
        unsigned int src0_end = ((i + 1) * N) / n_threads;

        src0_start = (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;
        src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

        ::ggml_gemm_q4_K_8x8_q8_K(K, (float *)(C + src0_start), ldc,
                                  (void *)((char *)B + src0_start * B_step),
                                  QA.data(), M4 * 4, src0_end - src0_start);
      }));
    }
    
    // Wait for parallel GEMM to complete
    for (auto &future : futures) {
      future.wait();
    }

    // Compute leftover 1 ~ 3 rows with multithreaded GEMV
    const unsigned int gemv_threads = std::min(4u, std::max(1u, N / 64));
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      std::vector<std::future<void>> gemv_futures;
      gemv_futures.reserve(gemv_threads);
      
      for (int thread_idx = 0; thread_idx < gemv_threads; ++thread_idx) {
        gemv_futures.emplace_back(bspool.submit_task([=, &QA, &B, &C]() {
          unsigned int M_step_start = (thread_idx * N) / gemv_threads;
          unsigned int M_step_end = ((thread_idx + 1) * N) / gemv_threads;

          M_step_start = (M_step_start % 8) ? M_step_start + 8 - (M_step_start % 8) : M_step_start;
          M_step_end = (M_step_end % 8) ? M_step_end + 8 - (M_step_end % 8) : M_step_end;

          ::ggml_gemv_q4_K_8x8_q8_K(
            K, (float *)((C + ((pb - M4 * 4) * N) + (M4 * 4 * N)) + M_step_start),
            N, (void *)((char *)B + M_step_start * B_step),
            QA.data() + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
            M_step_end - M_step_start);
        }));
      }
      
      for (auto &future : gemv_futures) {
        future.wait();
      }
    }
  } else { // GEMM
    unsigned int blocks_per_4_rows = (K + QK_K - 1) / QK_K;
    unsigned int qa_4_rows_size = sizeof(block_q8_Kx4) * blocks_per_4_rows;
    unsigned int M4 = ((M + 3) / 4);
    unsigned int B_step = sizeof(block_q4_K) * (K / QK_K);
    
    const unsigned int thread_num = std::min(
      static_cast<unsigned int>(std::thread::hardware_concurrency()),
      std::max(1u, N / 32));

    unsigned int qa_size = qa_4_rows_size * M4;
    PooledBuffer QA(qa_size);

    // Quantization of activations
    /// @note Heuristic inspection conducted that applying multithreading on
    /// run-time quantization hurts model latency
    for (int i = 0; i < static_cast<int>(M4); i++) {
      ::ggml_quantize_mat_q8_K_4x8(A + 4 * i * K,
                                   QA.data() + i * qa_4_rows_size, K);
    }

    // Use BS thread pool for parallel GEMM
    std::vector<std::future<void>> gemm_futures;
    gemm_futures.reserve(thread_num);
    
    for (int i = 0; i < thread_num; i++) {
      gemm_futures.emplace_back(bspool.submit_task([=, &QA, &B, &C]() {
        unsigned int src0_start = (i * N) / thread_num;
        unsigned int src0_end = ((i + 1) * N) / thread_num;

        src0_start = (src0_start % 8) ? src0_start + 8 - (src0_start % 8) : src0_start;
        src0_end = (src0_end % 8) ? src0_end + 8 - (src0_end % 8) : src0_end;

        ::ggml_gemm_q4_K_8x8_q8_K(K, (float *)(C + src0_start), ldc,
                                  (void *)((char *)B + src0_start * B_step),
                                  QA.data(), M, src0_end - src0_start);
      }));
    }
    
    for (auto &future : gemm_futures) {
      future.wait();
    }
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
  PooledBuffer v_q8_activation(q8_K_activation_size);
  simd::quantize_row_q8_K_optimized(f, v_q8_activation.data(), K);

  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_activation.data());
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
    auto &bspool = ThreadPoolManager::getInstance();
    PooledBuffer quantized_A(A_row_size);
    simd::quantize_row_q8_K_optimized(A, quantized_A.data(), K);

    const void *const quantized_A_data = quantized_A.data();

    // Use BS thread pool for parallel dot product computation
    const unsigned int optimal_threads = std::min(16u, std::max(1u, static_cast<unsigned int>(N) / 32));
    BS::multi_future<void> multi_future = bspool.submit_loop(0, static_cast<int>(N), 
      [&](int32_t thread_job) {
        const int32_t B_row_data_offset = B_row_size * thread_job;
        const void *const B_data = (void *)((char *)B + B_row_data_offset);
        
        ::ggml_vec_dot_q6_K_q8_K(K, &C[thread_job], bs, B_data, bx,
                                 quantized_A_data, by, nrc);
      });
    multi_future.wait();
  } else { // GEMM
    const int32_t A_total_size = A_row_size * M;
    PooledBuffer quantized_A(A_total_size);

    for (int32_t thread_job = 0; thread_job < static_cast<int>(M);
         thread_job++) {
      const int32_t A_row_data_offset = A_row_size * thread_job;
      void *A_data = (void *)((char *)quantized_A.data() + A_row_data_offset);
      simd::quantize_row_q8_K_optimized(A + thread_job * K, A_data, K);

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
