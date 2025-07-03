// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend.cpp
 * @date	03 April 2025
 * @brief	This is unittest for cpu_backend standalone
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nntrainer_test_util.h"
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include "blas_kernels.h"

#include <chrono>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

template <typename T, bool random_init = false>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(src[i * N + j] - src2[i * N + j]));
      err_sum += std::abs(src[i * N + j] - src2[i * N + j]);
    }
  }
  // std::cout << "err_sum : " << err_sum << std::endl;
  return max_diff;
}

#define QK4_0 32
/**
 * @brief q4_0 block
 *
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;
/**
 * @brief q4_K block
 *
 */
typedef struct {
  union {
    struct {
      int16_t d;    // super-block scale for quantized scales
      int16_t dmin; // super-block scale for quantized mins
    };
    uint32_t dm;
  };
  uint8_t scales[12];  // scales and mins, quantized with 6 bits
  uint8_t qs[256 / 2]; // 4--bit quants
} block_q4_K_testonly;
/**
 * @brief q8_K block
 *
 */
typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;
/**
 * @brief q4_Kx8 block
 *
 */
struct block_q4_Kx8_testonly {
  int16_t d[8];       // super-block scale for quantized scales
  int16_t dmin[8];    // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

#define QK_K 256
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  uint16_t d;               // super-block scale
} block_q6_K_testonly;

/**
 * @brief Elementwise-addition unittest : Vanilla example for formulating a TC
 * in unittest_nntrainer_cpu_backend.cpp
 *
 */
TEST(nntrainer_cpu_backend_standalone, ele_add) {
  const unsigned int TEST_SIZE = 100;
  float alpha = 1.F;
  float beta = 0.F;
  unsigned int i_stride = 1;
  unsigned int o_stride = 1;

  std::vector<float> lhs = generate_random_vector<float>(TEST_SIZE);
  std::vector<float> rhs = generate_random_vector<float>(TEST_SIZE);
  std::vector<float> dst(TEST_SIZE);

  const float *lhs_ptr = (const float *)lhs.data();
  const float *rhs_ptr = (const float *)rhs.data();
  float *dst_ptr = (float *)dst.data();

  nntrainer::ele_add(TEST_SIZE, lhs_ptr, rhs_ptr, dst_ptr, alpha, beta,
                     i_stride, o_stride);

  for (unsigned int i = 0; i < TEST_SIZE; ++i) {
    EXPECT_EQ(dst[i], lhs[i] + rhs[i]);
  }
}

float compute_mse(const uint32_t M, const uint32_t N,
                  std::vector<float> &ref_dst, std::vector<float> &dst,
                  bool print = false) {
  auto mean_squared_error =
    mse<float, float>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

#ifdef ENABLE_GGML

TEST(nntrainer_cpu_backend_standalone, q4_K_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q4_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

TEST(nntrainer_cpu_backend_standalone, q6_K_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * ne0 / q6_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q6_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q6_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

float test_gemm_q4_0(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // needed to initialize f16 tables

  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  int64_t q4_0_num_blocks = (K * N) / q4_0_block_size;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight = std::vector<char>(q4_0_data_size);

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  char *q4_0_offline_qWeight_ptr = (char *)q4_0_offline_qWeight.data();
  nntrainer::quantize_q4_0(weights, (void *)q4_0_offline_qWeight_ptr, N, K,
                           nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> q4_0_repacked_qWeight = std::vector<char>(q4_0_data_size);
  nntrainer::repack_q4_0_to_q4_0_8(q4_0_repacked_qWeight.data(),
                                   q4_0_offline_qWeight_ptr, q4_0_data_size, N,
                                   K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_0: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

float test_gemm_q4_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * N / q4_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q4_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> repacked_qWeight = std::vector<char>(data_size);
  nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,
                                   data_size, N, K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                       N, dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

float test_gemm_q6_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * N / q6_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q6_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q6_K(M, N, K, activations, K, (void *)offline_qWeight_ptr, N,
                       dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q6_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

static void run_quant_test(const uint32_t M, const uint32_t K, const uint32_t N,
                           float &q4_0_mse, float &q4_k_mse, float &q6_k_mse,
                           bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] Quantization Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  // q4_0_mse =
  //   test_gemm_q4_0(M, K, N, weight.data(), activation.data(), ref_dst,
  //   print);
  q4_k_mse =
    test_gemm_q4_K(M, K, N, weight.data(), activation.data(), ref_dst, print);
  q6_k_mse =
    test_gemm_q6_K(M, K, N, weight.data(), activation.data(), ref_dst, print);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_256x1024x512) {
  const unsigned int M = 256;
  const unsigned int K = 1024;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  // ASSERT_LE(q4_0_mse, 2.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x3072x3072) {
  const unsigned int M = 457;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  // ASSERT_LE(q4_0_mse, 1.5f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x3072x3072) {
  const unsigned int M = 458;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  // ASSERT_LE(q4_0_mse, 1.5f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x3072x3072) {
  const unsigned int M = 459;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  // ASSERT_LE(q4_0_mse, 1.5f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  // ASSERT_LE(q4_0_mse, 2.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x1024) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 1024;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  // ASSERT_LE(q4_0_mse, 1.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x3072) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  // ASSERT_LE(q4_0_mse, 1.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

static void run_vec_dot_test(const uint32_t K, bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_time = (nanoseconds)0;
  nanoseconds q6_k_time = (nanoseconds)0;
  float ref_result, result;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> activation = generate_random_vector<float, false>(K);
    std::vector<float> weight = generate_random_vector<float, false>(K);

    {
      // GROUND TRUTH sdot for reference
      auto t1 = high_resolution_clock::now();
      ref_result = nntrainer::sdot(K, weight.data(), 1, activation.data(), 1);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_time += dt;
      }
    }

    // Quantization of weights
    int64_t num_blocks = K / 256;
    size_t q6_k_data_size = num_blocks * sizeof(block_q6_K_testonly);
    std::vector<char> q6_K_weight = std::vector<char>(q6_k_data_size);
    nntrainer::quantize_row_q6_K(weight.data(), q6_K_weight.data(), K);

    // Quantization of activations
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    int q8_K_activation_size = sizeof(block_q8_K_testonly) * blocks_per_row;
    std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
    nntrainer::quantize_row_q8_K(activation.data(), v_q8_activation.data(), K);

    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      result =
        nntrainer::dot_q6_K_q8_K(K, q6_K_weight.data(), v_q8_activation.data());
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        q6_k_time += dt;
      }
    }
    EXPECT_NEAR(result, ref_result, 0.25 * K / 256);
  }
  if (print) {
    std::cout << "[INFO] dot_q6_K_q8_K: TEST CNT: " << TEST_CNT << ", K: " << K
              << ", Average ref_time: " << ref_time.count() / TEST_CNT
              << " ns, Average q6_k_time: " << q6_k_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_1024) {
  const uint32_t K = 1024;
  run_vec_dot_test(K);
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_2560) {
  const uint32_t K = 2560;
  run_vec_dot_test(K);
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_10240) {
  const uint32_t K = 10240;
  run_vec_dot_test(K);
}

#endif

static void run_ele_mul_test(const unsigned int N, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  // Z = X ⊙ alpha * Y + beta * Z
  const int TEST_CNT = 20;
  nanoseconds ref_mul_time = (nanoseconds)0;
  nanoseconds mul_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Y = generate_random_vector<float, false>(
      std::max<size_t>(1, (size_t)N * i_stride));
    std::vector<float> Z =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Z_ref = Z;
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_ele_mul(N, X.data(), Y.data(), Z_ref.data(), alpha,
                                    beta, i_stride, o_stride);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_mul_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::ele_mul(N, X.data(), Y.data(), Z.data(), alpha, beta, i_stride,
                         o_stride);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        mul_time += dt;
      }
    }

    auto mean_squared_error = compute_mse(1, N, Z_ref, Z, false);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }

  std::cout << "[INFO] ele_mul: TEST CNT: " << TEST_CNT << ", N: " << N
            << ", i_stride: " << i_stride
            << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
            << " ns, Average mul_time: " << mul_time.count() / TEST_CNT
            << " ns " << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_0) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 0;
  const unsigned int o_stride = 1;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_1) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 1;
  const unsigned int o_stride = 1;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_mul_3072_istr_16_ostr_16) {
  const unsigned int N = 3072;
  const float alpha = 3.f;
  const float beta = 2.f;
  const unsigned int i_stride = 16;
  const unsigned int o_stride = 16;
  run_ele_mul_test(N, alpha, beta, i_stride, o_stride);
}

static void run_ele_add_test(const unsigned int N, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  // Z = X ⊙ alpha * Y + beta * Z
  const int TEST_CNT = 20;
  nanoseconds ref_add_time = (nanoseconds)0;
  nanoseconds add_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Y = generate_random_vector<float, false>(
      std::max<size_t>(1, (size_t)N * i_stride));
    std::vector<float> Z =
      generate_random_vector<float, false>((size_t)N * o_stride);
    std::vector<float> Z_ref = Z;
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_ele_add(N, X.data(), Y.data(), Z_ref.data(), alpha,
                                    beta, i_stride, o_stride);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_add_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::ele_add(N, X.data(), Y.data(), Z.data(), alpha, beta, i_stride,
                         o_stride);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        add_time += dt;
      }
    }
    auto mean_squared_error = compute_mse(1, N, Z_ref, Z, false);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }

  std::cout << "[INFO] ele_add: TEST CNT: " << TEST_CNT << ", N: " << N
            << ", i_stride: " << i_stride
            << ", Average ref_time: " << ref_add_time.count() / TEST_CNT
            << " ns, Average add_time: " << add_time.count() / TEST_CNT
            << " ns " << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_0) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 0;
  const unsigned int o_stride = 1;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_1) {
  const unsigned int N = 3072;
  const float alpha = 1.f;
  const float beta = 0.f;
  const unsigned int i_stride = 1;
  const unsigned int o_stride = 1;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

template <bool is_q_8_1_weights = false>
static void run_q_6_K_test(const uint32_t M, const uint32_t K,
                           const uint32_t N) {

  auto debug_print_beg_end = [M, K, N](const float *const data,
                                       const uint32_t count = 5) {
    std::cout << "[";
    for (unsigned int i = 0; i < count; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "][";
    for (unsigned int i = M * N - count; i < M * N; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "]";
  };

  static constexpr uint32_t run_count = 100;

  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  std::vector<float> weight = generate_random_vector<float, false>(N * K);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> cpu_q6_dst(M * N, 0.0f);
  // std::vector<float> gpu_q6_dst(M * N, 0.0f);

  void *gpu_q6_dst =
    nntrainer::blas_cc->context_inst_.createSVMRegion(M * N * sizeof(float));

  const auto data_size = sizeof(block_q6_K_testonly) * N * K / 256;
  std::vector<char> q6_weight = std::vector<char>(data_size);
  // char *q6_weight_ptr = (char *)q6_weight.data();

  void *q6_weight_ptr =
    nntrainer::blas_cc->context_inst_.createSVMRegion(data_size);

  nntrainer::blas_cc->command_queue_inst_.enqueueSVMMap(q6_weight_ptr,
                                                        data_size, false);

  float *weights_f32_ptr = weight.data();
  // float *activations_f32_ptr = activation.data();

  float *activations_f32_ptr =
    (float *)nntrainer::blas_cc->context_inst_.createSVMRegion(M * K *
                                                               sizeof(float));

  nntrainer::blas_cc->command_queue_inst_.enqueueSVMMap(
    activations_f32_ptr, M * K * sizeof(float), false);

  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  // F32-F32 GEMM
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  nntrainer::quantize_q6_K(weights_f32_ptr, (void *)q6_weight_ptr, N, K,
                           nullptr);

  auto t1 = high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q6_K(M, N, K, activations_f32_ptr, K, (void *)q6_weight_ptr,
                         N, cpu_q6_dst.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);

  auto t3 = high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    if constexpr (is_q_8_1_weights) {
      nntrainer::sgemv_q6_k_cl(q6_weight_ptr, activations_f32_ptr,
                               (float *)gpu_q6_dst, K, N);
    } else {
      nntrainer::sgemv_q6_k_cl(q6_weight_ptr, activations_f32_ptr,
                               (float *)gpu_q6_dst, K, N);
    }
  }
  auto t4 = high_resolution_clock::now();
  auto gpu_dt = duration_cast<nanoseconds>(t4 - t3);

  // Compute raports
  {
    uint32_t first_zero_index = UINT32_MAX;
    int zeros = 0;
    int nans = 0;

    for (uint32_t i = 0; i < M * N; ++i) {
      if (((float *)gpu_q6_dst)[i] == 0) {
        zeros++;
        if (first_zero_index == UINT32_MAX) {
          first_zero_index = i;
        }
      }

      if (std::isnan(((float *)gpu_q6_dst)[i])) {
        nans++;
      }
    }

    // const auto mean_squared_error_dst_gpu =
    //   compute_mse(M, N, ref_dst, gpu_q6_dst, false);
    // const auto mean_squared_error_dst =
    //   compute_mse(M, N, ref_dst, cpu_q6_dst, false);

    const auto data_size_mb = data_size / (1024 * 1024.0f);

    std::cout << "--- Raport (is_q_8_1_weights = " << is_q_8_1_weights
              << ") : " << M << " x " << K << " x " << N << std::endl;
    std::cout << " - q6_K data size : " << data_size_mb << " [MB]" << std::endl;
    std::cout << " - time : CPU = "
              << dt.count() / (1000.f * 1000.0f * run_count) << " ms"
              << std::endl;
    std::cout << " - time : GPU = "
              << gpu_dt.count() / (1000.f * 1000.0f * run_count) << " ms"
              << std::endl;
    std::cout << " - sample : CPU = ";
    debug_print_beg_end(cpu_q6_dst.data());
    std::cout << std::endl;
    std::cout << " - sample : GPU = ";
    debug_print_beg_end((float *)gpu_q6_dst);
    std::cout << std::endl;
    std::cout << " - zeros : " << zeros << " / " << M * N << " [ "
              << zeros * 100.0f / float(M * N) << " %] - first at [ "
              << first_zero_index << " ]" << std::endl;
    std::cout << " - nans : " << nans << " / " << M * N << " [ "
              << nans * 100.0f / float(M * N) << " %]" << std::endl;
    // std::cout << " - MSE : CPU = " << mean_squared_error_dst << std::endl;
    // std::cout << " - MSE : GPU = " << mean_squared_error_dst_gpu <<
    // std::endl;
  }
}

#define DECLARE_q_6_K_test_M_K_N(M, K, N, is_q_8_1_weights)                    \
  TEST(nntrainer_cpu_backend_standalone,                                       \
       q_6_K_test_##M##_##K##_##N##_is_q_8_1_weights_##is_q_8_1_weights) {     \
    run_q_6_K_test<is_q_8_1_weights>(M, K, N);                                 \
  }

// DECLARE_q_6_K_test_M_K_N(1, 512, 32);
// DECLARE_q_6_K_test_M_K_N(1, 768, 32);
// DECLARE_q_6_K_test_M_K_N(1, 1024, 32);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 32);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 32);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 32);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 32);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 32);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 32);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 32);
//
// DECLARE_q_6_K_test_M_K_N(1, 512, 64);
// DECLARE_q_6_K_test_M_K_N(1, 768, 64);
// DECLARE_q_6_K_test_M_K_N(1, 1024, 64);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 64);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 64);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 64);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 64);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 64);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 64);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 64);
//
// DECLARE_q_6_K_test_M_K_N(1, 512, 128);
// DECLARE_q_6_K_test_M_K_N(1, 768, 128);
// DECLARE_q_6_K_test_M_K_N(1, 1024, 128);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 128);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 128);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 128);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 128);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 128);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 128);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 128);
//
// DECLARE_q_6_K_test_M_K_N(1, 128, 512);
// DECLARE_q_6_K_test_M_K_N(1, 128, 768);
// DECLARE_q_6_K_test_M_K_N(1, 128, 1024);
// DECLARE_q_6_K_test_M_K_N(1, 128, 2048);
// DECLARE_q_6_K_test_M_K_N(1, 128, 4096);
// DECLARE_q_6_K_test_M_K_N(1, 128, 8192);
// DECLARE_q_6_K_test_M_K_N(1, 128, 16384);
// DECLARE_q_6_K_test_M_K_N(1, 128, 32768);
// DECLARE_q_6_K_test_M_K_N(1, 128, 65536);
// DECLARE_q_6_K_test_M_K_N(1, 128, 131072);
//
// DECLARE_q_6_K_test_M_K_N(1, 256, 512);
// DECLARE_q_6_K_test_M_K_N(1, 256, 768);
// DECLARE_q_6_K_test_M_K_N(1, 256, 1024);
// DECLARE_q_6_K_test_M_K_N(1, 256, 2048);
// DECLARE_q_6_K_test_M_K_N(1, 256, 4096);
// DECLARE_q_6_K_test_M_K_N(1, 256, 8192);
// DECLARE_q_6_K_test_M_K_N(1, 256, 16384);
// DECLARE_q_6_K_test_M_K_N(1, 256, 32768);
// DECLARE_q_6_K_test_M_K_N(1, 256, 65536);
// DECLARE_q_6_K_test_M_K_N(1, 256, 131072);
//
// DECLARE_q_6_K_test_M_K_N(1, 512, 256);
// DECLARE_q_6_K_test_M_K_N(1, 768, 256);
// DECLARE_q_6_K_test_M_K_N(1, 1024, 256);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 256);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 256);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 256);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 256);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 256);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 256);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 256);
//
// // DECLARE_q_6_K_test_M_K_N(1, 384, 512);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 768);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 1024);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 2048);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 4096);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 8192);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 16384);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 32768);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 65536);
// // DECLARE_q_6_K_test_M_K_N(1, 384, 131072);
//
// DECLARE_q_6_K_test_M_K_N(1, 512, 384);
// DECLARE_q_6_K_test_M_K_N(1, 768, 384);
// DECLARE_q_6_K_test_M_K_N(1, 1024, 384);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 384);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 384);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 384);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 384);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 384);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 384);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 384);
//
// DECLARE_q_6_K_test_M_K_N(1, 512, 320);
// DECLARE_q_6_K_test_M_K_N(1, 768, 320);
// DECLARE_q_6_K_test_M_K_N(1, 1024, 320);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 320);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 320);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 320);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 320);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 320);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 320);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 320);
//
// DECLARE_q_6_K_test_M_K_N(1, 512, 288);
// DECLARE_q_6_K_test_M_K_N(1, 768, 288);
// DECLARE_q_6_K_test_M_K_N(1, 1024, 288);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 288);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 288);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 288);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 288);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 288);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 288);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 288);
//
// DECLARE_q_6_K_test_M_K_N(1, 512, 768);
// DECLARE_q_6_K_test_M_K_N(1, 512, 1024);
// DECLARE_q_6_K_test_M_K_N(1, 512, 2048);
// DECLARE_q_6_K_test_M_K_N(1, 512, 4096);
// DECLARE_q_6_K_test_M_K_N(1, 512, 8192);
// DECLARE_q_6_K_test_M_K_N(1, 512, 16384);
// DECLARE_q_6_K_test_M_K_N(1, 512, 32768);
// DECLARE_q_6_K_test_M_K_N(1, 512, 65536);
// DECLARE_q_6_K_test_M_K_N(1, 512, 131072);
//
// DECLARE_q_6_K_test_M_K_N(1, 768, 512);
// DECLARE_q_6_K_test_M_K_N(1, 1024, 512);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 512);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 512);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 512);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 512);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 512);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 512);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 512);
//
// DECLARE_q_6_K_test_M_K_N(1, 768, 1024);
// DECLARE_q_6_K_test_M_K_N(1, 768, 2048);
// DECLARE_q_6_K_test_M_K_N(1, 768, 4096);
// DECLARE_q_6_K_test_M_K_N(1, 768, 8192);
// DECLARE_q_6_K_test_M_K_N(1, 768, 16384);
// DECLARE_q_6_K_test_M_K_N(1, 768, 32768);
// DECLARE_q_6_K_test_M_K_N(1, 768, 65536);
// DECLARE_q_6_K_test_M_K_N(1, 768, 131072);
//
// DECLARE_q_6_K_test_M_K_N(1, 1024, 768);
// DECLARE_q_6_K_test_M_K_N(1, 2048, 768);
// DECLARE_q_6_K_test_M_K_N(1, 4096, 768);
// DECLARE_q_6_K_test_M_K_N(1, 8192, 768);
// DECLARE_q_6_K_test_M_K_N(1, 16384, 768);
// DECLARE_q_6_K_test_M_K_N(1, 32768, 768);
// DECLARE_q_6_K_test_M_K_N(1, 65536, 768);
// DECLARE_q_6_K_test_M_K_N(1, 131072, 768);

DECLARE_q_6_K_test_M_K_N(1, 3072, 105900, false);
DECLARE_q_6_K_test_M_K_N(1, 3072, 105900, true);

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
