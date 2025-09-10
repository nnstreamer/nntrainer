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
#include <fp16.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

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

#if defined(ENABLE_FP16) || defined(__AVX2__)
static inline std::vector<uint16_t>
convert_vector_f32_to_f16_as_uint16(std::vector<float> f32_vec) {
  std::vector<uint16_t> vec(f32_vec.size());
  for (size_t i = 0; i < f32_vec.size(); i++) {
    vec[i] = nntrainer::compute_fp32_to_fp16(f32_vec[i]);
  }
  return vec;
}

static inline std::vector<float>
convert_vector_f16_as_uint16_to_f32(std::vector<uint16_t> uint16_vec) {
  std::vector<float> vec(uint16_vec.size());
  for (size_t i = 0; i < uint16_vec.size(); i++) {
    vec[i] = nntrainer::compute_fp16_to_fp32(uint16_vec[i]);
  }
  return vec;
}
#endif

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

template <typename T = float>
float compute_mse(const uint32_t M, const uint32_t N, std::vector<T> &ref_dst,
                  std::vector<T> &dst, bool print = false) {
  auto mean_squared_error = mse<T, T>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity<T, T>(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff<T>(ref_dst.data(), dst.data(), M, N);

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

TEST(nntrainer_cpu_backend_standalone, q4_0_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q4_0_block_size = QK4_0;
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t num_blocks = (K * N) / q4_0_block_size;
  size_t data_size = num_blocks * q4_0_type_size;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q4_0(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_0(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

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
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(), q4_0_offline_qWeight_ptr,
                         q4_0_data_size, N, K);

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
  nntrainer::repack_q4_K(repacked_qWeight.data(), offline_qWeight_ptr,
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
  q4_0_mse =
    test_gemm_q4_0(M, K, N, weight.data(), activation.data(), ref_dst, print);
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
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x3072x3072) {
  const unsigned int M = 457;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x3072x3072) {
  const unsigned int M = 458;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x3072x3072) {
  const unsigned int M = 459;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x1024) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 1024;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x3072) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q4_k_mse, q4_0_mse);
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
                             unsigned int i_stride, unsigned int o_stride,
                             bool print = false) {
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

  if (print) {
    std::cout << "[INFO] ele_mul: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", i_stride: " << i_stride
              << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
              << " ns, Average mul_time: " << mul_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
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
                             unsigned int i_stride, unsigned int o_stride,
                             bool print = false) {
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
  if (print) {
    std::cout << "[INFO] ele_add: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", i_stride: " << i_stride
              << ", Average ref_time: " << ref_add_time.count() / TEST_CNT
              << " ns, Average add_time: " << add_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
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

TEST(nntrainer_cpu_backend_standalone, ele_add_3072_istr_16_ostrid_16) {
  const unsigned int N = 3072;
  const float alpha = 3.f;
  const float beta = 2.f;
  const unsigned int i_stride = 16;
  const unsigned int o_stride = 16;
  run_ele_add_test(N, alpha, beta, i_stride, o_stride);
}

TEST(nntrainer_cpu_backend_standalone, softmax_row_inplace) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  std::vector<float> qk_out = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,
    1.973170,  1.937003,  -6.879627, -1.083345, -6.880110, -8.000502,
    -8.838327, -0.815022, 7.323523,  -3.325828, 2.022300,  -7.142663,
    4.161452,  3.017770,  -9.588310, -8.871769, 9.398197,  4.439975,
    6.648853,  8.771055,  -5.753218, -9.984425, -6.363501, 9.844232};
  std::vector<float> ref_out = {
    0.986697, 0.999999, 0.405184, 0.000021, 0.043301, 0.040031,
    0.487615, 0.999879, 0.000016, 0.000018, 0.012472, 0.000001,
    0.000000, 0.005194, 0.633859, 0.000005, 0.512170, 0.000114,
    0.999957, 0.001083, 0.000831, 0.000000, 0.594816, 0.994785,
    0.322840, 0.959963, 0.000215, 0.000007, 0.000027, 0.998899};

  nntrainer::softmax_row_inplace(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, softmax_row) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  std::vector<float> qk_out = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,
    1.973170,  1.937003,  -6.879627, -1.083345, -6.880110, -8.000502,
    -8.838327, -0.815022, 7.323523,  -3.325828, 2.022300,  -7.142663,
    4.161452,  3.017770,  -9.588310, -8.871769, 9.398197,  4.439975,
    6.648853,  8.771055,  -5.753218, -9.984425, -6.363501, 9.844232};
  std::vector<float> ref_out = {
    0.986697, 0.999999, 0.405184, 0.000021, 0.043301, 0.040031,
    0.487615, 0.999879, 0.000016, 0.000018, 0.012472, 0.000001,
    0.000000, 0.005194, 0.633859, 0.000005, 0.512170, 0.000114,
    0.999957, 0.001083, 0.000831, 0.000000, 0.594816, 0.994785,
    0.322840, 0.959963, 0.000215, 0.000007, 0.000027, 0.998899};

  nntrainer::softmax_row(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0001f);
  }
}

#if defined(ENABLE_FP16) || defined(__AVX2__)
TEST(nntrainer_cpu_backend_standalone, compute_kcaches) {
  int num_rows = 1;
  int N = 2;
  int head_dim = 10;
  int group_size = 4;
  int tile_size = 16;
  size_t in_size = N * group_size * head_dim;
  size_t kcache_size = num_rows * N * head_dim;
  size_t output_size = num_rows * N * group_size;
  std::vector<float> in = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,  1.973170,
    1.937003,  -6.879627, -1.083345, -6.880110, -8.000502, -8.838327, -0.815022,
    7.323523,  -3.325828, 2.022300,  -7.142663, 4.161452,  3.017770,  -9.588310,
    -8.871769, 9.398197,  4.439975,  6.648853,  8.771055,  -5.753218, -9.984425,
    -6.363501, 9.844232,  -6.331910, 2.349631,  -3.915155, 2.233063,  0.495129,
    -9.858674, -1.361099, -9.538752, -4.175417, 0.495493,  2.237058,  -2.002780,
    -7.210123, -9.066687, -4.157107, 9.475111,  -2.672763, -5.344573, -0.878600,
    -8.187871, 5.703520,  2.367721,  -6.006524, -2.350760, 0.284688,  9.664618,
    1.848291,  -0.664742, -9.070992, 7.198808,  2.150897,  3.606151,  -6.589518,
    -0.990015, -8.698968, -9.734701, 8.977711,  8.844034,  9.312640,  1.265764,
    6.167947,  -2.291670, -3.907725, -9.680675, -8.046557, -5.382123, 3.684660,
    -5.179491, -1.196950, 3.665271};
  std::vector<uint16_t> kcache = {3751, 7967, 9507, 1842, 7322, 7799, 5990,
                                  5972, 1568, 4463, 1568, 1008, 590,  4597,
                                  8663, 3343, 6015, 1437, 7083, 6512};
  std::vector<float> ref_out = {0.089252,  -0.072949, 0.058948,  -0.045583,
                                -0.025812, -0.002068, -0.014971, -0.028027};
  std::vector<float> output(output_size);

  nntrainer::compute_kcaches<uint16_t>(in.data(), kcache.data(), output.data(),
                                       num_rows, N, head_dim, group_size,
                                       tile_size);

  for (size_t i = 0; i < output_size; i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_null) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;
  bool only_convert_to_fp16 = false;

  std::vector<float> inout = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,  0.197317,
    0.193700,  -0.687963, -0.108334, -0.688011, -0.800050, -0.883833, -0.081502,
    0.732352,  -0.332583, 0.202230,  -0.714266, 0.416145,  0.301777,  -0.958831,
    -0.887177, 0.939820,  0.443998,  0.664885,  0.877105,  -0.575322, -0.998442,
    -0.636350, 0.984423,  -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,
    -0.985867, -0.136110, -0.953875, -0.417542, 0.049549};
  std::vector<float> cos_ = {-0.508061, 0.998221,  -0.733869, 0.164825,
                             0.297530,  -0.591437, -0.165907, -0.999895,
                             -0.163179, 0.249787};
  std::vector<float> sin_ = {0.809128,  -0.390829, -0.112627, 0.377992,
                             -0.994569, -0.641599, -0.927266, -0.401338,
                             0.244335,  -0.414953};
  std::vector<float> ref_out = {
    0.684172,  0.279348,  -0.761074, -0.073549, 0.866425,  -0.544224, 0.154785,
    -0.480342, 0.010582,  0.098163,  0.146526,  -1.030422, 0.547092,  -0.252752,
    -0.243571, -0.162197, -0.216517, 0.636452,  -0.236000, 0.120334,  0.999478,
    -0.793769, -0.733800, -0.011226, 0.247067,  -1.151284, -0.030760, 0.615511,
    0.205859,  0.266457,  -0.454117, 0.581280,  0.181472,  0.204634,  -0.646542,
    0.020328,  0.556058,  1.354487,  -0.087349, -0.396113};

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(), nullptr,
                                      cos_.data(), sin_.data(),
                                      only_convert_to_fp16);

  for (size_t i = 0; i < inout.size(); i++) {
    EXPECT_NEAR(ref_out[i], inout[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_uint16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;
  bool only_convert_to_fp16 = false;

  std::vector<float> inout = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,  0.197317,
    0.193700,  -0.687963, -0.108334, -0.688011, -0.800050, -0.883833, -0.081502,
    0.732352,  -0.332583, 0.202230,  -0.714266, 0.416145,  0.301777,  -0.958831,
    -0.887177, 0.939820,  0.443998,  0.664885,  0.877105,  -0.575322, -0.998442,
    -0.636350, 0.984423,  -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,
    -0.985867, -0.136110, -0.953875, -0.417542, 0.049549};
  std::vector<float> cos_ = {-0.508061, 0.998221,  -0.733869, 0.164825,
                             0.297530,  -0.591437, -0.165907, -0.999895,
                             -0.163179, 0.249787};
  std::vector<float> sin_ = {0.809128,  -0.390829, -0.112627, 0.377992,
                             -0.994569, -0.641599, -0.927266, -0.401338,
                             0.244335,  -0.414953};
  std::vector<float> ref_out = {
    0.684172,  0.279348,  -0.761074, -0.073549, 0.866425,  -0.544224, 0.154785,
    -0.480342, 0.010582,  0.098163,  0.146526,  -1.030422, 0.547092,  -0.252752,
    -0.243571, -0.162197, -0.216517, 0.636452,  -0.236000, 0.120334,  0.999478,
    -0.793769, -0.733800, -0.011226, 0.247067,  -1.151284, -0.030760, 0.615511,
    0.205859,  0.266457,  -0.454117, 0.581280,  0.181472,  0.204634,  -0.646542,
    0.020328,  0.556058,  1.354487,  -0.087349, -0.396113};
  std::vector<uint16_t> output(width);

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(),
                                      output.data(), cos_.data(), sin_.data(),
                                      only_convert_to_fp16);

  std::vector<float> f32_output = convert_vector_f16_as_uint16_to_f32(output);
  for (size_t i = 0; i < f32_output.size(); i++) {
    EXPECT_NEAR(ref_out[i], f32_output[i], 0.0003f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_fp16vcache_fp32_transposed) {
  int row_num = 1;
  int num_cache_head = 2;
  int gqa_size = 2;
  int head_dim = 9;
  int max_iter = 2;
  size_t in_size = max_iter * max_iter * num_cache_head * gqa_size;
  size_t vcache_size = max_iter * num_cache_head * head_dim;
  size_t output_size = num_cache_head * gqa_size * head_dim;

  std::vector<float> in = {0.092400,  -0.414656, -0.271581, -0.300455,
                           0.317950,  -0.176271, 0.138504,  -0.361754,
                           -0.024215, -0.968064, 0.360932,  0.384471};
  std::vector<float> f32_vcache = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,
    0.197317,  0.193700,  -0.687963, -0.108334, -0.688011, -0.800050,
    -0.883833, -0.081502, 0.732352,  -0.332583, 0.202230,  -0.714266,
    0.416145,  0.301777,  -0.958831, -0.887177, 0.939820,  0.443998,
    0.664885,  0.877105,  -0.575322, -0.998442, -0.636350, 0.984423,
    -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,  -0.985867};
  std::vector<uint16_t> vcache =
    convert_vector_f32_to_f16_as_uint16(f32_vcache);
  std::vector<float> ref_out = {
    0.109160,  0.150761,  -0.221623, -0.340605, 0.341716,  0.192903,
    0.229677,  0.296728,  -0.246453, 0.030694,  -0.299190, -0.204716,
    0.418990,  -0.358029, -0.310309, -0.199024, -0.234911, 0.386668,
    -0.108879, 0.098724,  0.353685,  0.152306,  0.054675,  -0.253151,
    0.121229,  -0.048077, 0.057462,  0.393775,  0.436868,  -0.115650,
    0.494638,  -0.060525, -0.078396, 0.019140,  -0.078680, 0.571263};
  std::vector<float> output(output_size);

  nntrainer::compute_fp16vcache_fp32_transposed(
    row_num, in.data(), vcache.data(), output.data(), num_cache_head, gqa_size,
    head_dim);

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}
#endif

#if defined(ENABLE_FP16) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
TEST(nntrainer_cpu_backend_standalone, softmax_row_inplace_fp16) {
  size_t start_row = 0;
  size_t end_row = 3;
  size_t num_heads = 10;
  size_t qk_out_size = num_heads * end_row;
  // auto qk_out = generate_random_vector<float>(qk_out_size, -10, 10);
  std::vector<__fp16> qk_out = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,
    1.973170,  1.937003,  -6.879627, -1.083345, -6.880110, -8.000502,
    -8.838327, -0.815022, 7.323523,  -3.325828, 2.022300,  -7.142663,
    4.161452,  3.017770,  -9.588310, -8.871769, 9.398197,  4.439975,
    6.648853,  8.771055,  -5.753218, -9.984425, -6.363501, 9.844232};
  std::vector<__fp16> ref_out = {
    0.986697, 0.999999, 0.405184, 0.000021, 0.043301, 0.040031,
    0.487615, 0.999879, 0.000016, 0.000018, 0.012472, 0.000001,
    0.000000, 0.005194, 0.633859, 0.000005, 0.512170, 0.000114,
    0.999957, 0.001083, 0.000831, 0.000000, 0.594816, 0.994785,
    0.322840, 0.959963, 0.000215, 0.000007, 0.000027, 0.998899};

  nntrainer::softmax_row_inplace(qk_out.data(), start_row, end_row, num_heads);

  for (size_t i = 0; i < qk_out_size; i++) {
    EXPECT_NEAR(ref_out[i], qk_out[i], 0.0005f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_kcaches_fp16) {
  int num_rows = 1;
  int num_cache_head = 2;
  int head_dim = 10;
  int group_size = 4;
  int tile_size = 16;
  int tile_offset = 0;
  size_t in_size = num_cache_head * group_size * head_dim;
  size_t kcache_size = num_rows * num_cache_head * head_dim;
  size_t output_size = num_rows * num_cache_head * group_size;
  std::vector<__fp16> in = {
    -2.509198, 5.930860,  9.014286,  -6.331305, 4.639878,  5.593820,  1.973170,
    1.937003,  -6.879627, -1.083345, -6.880110, -8.000502, -8.838327, -0.815022,
    7.323523,  -3.325828, 2.022300,  -7.142663, 4.161452,  3.017770,  -9.588310,
    -8.871769, 9.398197,  4.439975,  6.648853,  8.771055,  -5.753218, -9.984425,
    -6.363501, 9.844232,  -6.331910, 2.349631,  -3.915155, 2.233063,  0.495129,
    -9.858674, -1.361099, -9.538752, -4.175417, 0.495493,  2.237058,  -2.002780,
    -7.210123, -9.066687, -4.157107, 9.475111,  -2.672763, -5.344573, -0.878600,
    -8.187871, 5.703520,  2.367721,  -6.006524, -2.350760, 0.284688,  9.664618,
    1.848291,  -0.664742, -9.070992, 7.198808,  2.150897,  3.606151,  -6.589518,
    -0.990015, -8.698968, -9.734701, 8.977711,  8.844034,  9.312640,  1.265764,
    6.167947,  -2.291670, -3.907725, -9.680675, -8.046557, -5.382123, 3.684660,
    -5.179491, -1.196950, 3.665271};
  std::vector<__fp16> kcache = {
    0.000406, 0.006954, 0.020065, 0.000110, 0.004494, 0.006313, 0.001806,
    0.001789, 0.000093, 0.000663, 0.000093, 0.000060, 0.000035, 0.000727,
    0.011406, 0.000309, 0.001830, 0.000086, 0.003744, 0.002655};
  std::vector<__fp16> ref_out = {0.089252,  -0.072949, 0.058948,  -0.045583,
                                 -0.025812, -0.002068, -0.014971, -0.028027};
  std::vector<__fp16> output(output_size);
  for (int n = 0; n < num_cache_head; ++n) {
    const __fp16 *in_ptr = in.data() + n * group_size * head_dim;
    const __fp16 *kcache_ptr = kcache.data() + n * head_dim;
    __fp16 *out_ptr = output.data() + n * group_size;
    nntrainer::compute_kcaches(in_ptr, kcache_ptr, out_ptr, num_rows,
                               num_cache_head, head_dim, group_size,
                               tile_offset, tile_size);
  }

  for (size_t i = 0; i < output_size; i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_out_null_fp16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;

  std::vector<__fp16> inout = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,  0.197317,
    0.193700,  -0.687963, -0.108334, -0.688011, -0.800050, -0.883833, -0.081502,
    0.732352,  -0.332583, 0.202230,  -0.714266, 0.416145,  0.301777,  -0.958831,
    -0.887177, 0.939820,  0.443998,  0.664885,  0.877105,  -0.575322, -0.998442,
    -0.636350, 0.984423,  -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,
    -0.985867, -0.136110, -0.953875, -0.417542, 0.049549};
  std::vector<__fp16> cos_ = {-0.508061, 0.998221,  -0.733869, 0.164825,
                              0.297530,  -0.591437, -0.165907, -0.999895,
                              -0.163179, 0.249787};
  std::vector<__fp16> sin_ = {0.809128,  -0.390829, -0.112627, 0.377992,
                              -0.994569, -0.641599, -0.927266, -0.401338,
                              0.244335,  -0.414953};
  std::vector<__fp16> ref_out = {
    0.684172,  0.279348,  -0.761074, -0.073549, 0.866425,  -0.544224, 0.154785,
    -0.480342, 0.010582,  0.098163,  0.146526,  -1.030422, 0.547092,  -0.252752,
    -0.243571, -0.162197, -0.216517, 0.636452,  -0.236000, 0.120334,  0.999478,
    -0.793769, -0.733800, -0.011226, 0.247067,  -1.151284, -0.030760, 0.615511,
    0.205859,  0.266457,  -0.454117, 0.581280,  0.181472,  0.204634,  -0.646542,
    0.020328,  0.556058,  1.354487,  -0.087349, -0.396113};

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(), nullptr,
                                      cos_.data(), sin_.data());

  for (size_t i = 0; i < inout.size(); i++) {
    EXPECT_NEAR(ref_out[i], inout[i], 0.001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_rotary_emb_value_fp16) {
  unsigned int width = 40;
  unsigned int dim = 20;
  unsigned int half_ = 10;

  std::vector<__fp16> inout = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,  0.197317,
    0.193700,  -0.687963, -0.108334, -0.688011, -0.800050, -0.883833, -0.081502,
    0.732352,  -0.332583, 0.202230,  -0.714266, 0.416145,  0.301777,  -0.958831,
    -0.887177, 0.939820,  0.443998,  0.664885,  0.877105,  -0.575322, -0.998442,
    -0.636350, 0.984423,  -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,
    -0.985867, -0.136110, -0.953875, -0.417542, 0.049549};
  std::vector<__fp16> cos_ = {-0.508061, 0.998221,  -0.733869, 0.164825,
                              0.297530,  -0.591437, -0.165907, -0.999895,
                              -0.163179, 0.249787};
  std::vector<__fp16> sin_ = {0.809128,  -0.390829, -0.112627, 0.377992,
                              -0.994569, -0.641599, -0.927266, -0.401338,
                              0.244335,  -0.414953};
  std::vector<__fp16> ref_out = {
    0.684172,  0.279348,  -0.761074, -0.073549, 0.866425,  -0.544224, 0.154785,
    -0.480342, 0.010582,  0.098163,  0.146526,  -1.030422, 0.547092,  -0.252752,
    -0.243571, -0.162197, -0.216517, 0.636452,  -0.236000, 0.120334,  0.999478,
    -0.793769, -0.733800, -0.011226, 0.247067,  -1.151284, -0.030760, 0.615511,
    0.205859,  0.266457,  -0.454117, 0.581280,  0.181472,  0.204634,  -0.646542,
    0.020328,  0.556058,  1.354487,  -0.087349, -0.396113};
  std::vector<__fp16> output(width);

  nntrainer::compute_rotary_emb_value(width, dim, half_, inout.data(),
                                      output.data(), cos_.data(), sin_.data());

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.001f);
  }
}

TEST(nntrainer_cpu_backend_standalone, compute_fp16vcache_transposed_fp16) {
  int row_num = 1;
  int num_cache_head = 2;
  int gqa_size = 2;
  int head_dim = 9;
  int max_iter = 2;
  int chunk_size = 9;
  size_t in_size = max_iter * max_iter * num_cache_head * gqa_size;
  size_t vcache_size = max_iter * num_cache_head * head_dim;
  size_t output_size = num_cache_head * gqa_size * head_dim;

  std::vector<__fp16> in = {0.092400,  -0.414656, -0.271581, -0.300455,
                            0.317950,  -0.176271, 0.138504,  -0.361754,
                            -0.024215, -0.968064, 0.360932,  0.384471};
  std::vector<__fp16> vcache = {
    -0.250920, 0.593086,  0.901429,  -0.633130, 0.463988,  0.559382,
    0.197317,  0.193700,  -0.687963, -0.108334, -0.688011, -0.800050,
    -0.883833, -0.081502, 0.732352,  -0.332583, 0.202230,  -0.714266,
    0.416145,  0.301777,  -0.958831, -0.887177, 0.939820,  0.443998,
    0.664885,  0.877105,  -0.575322, -0.998442, -0.636350, 0.984423,
    -0.633191, 0.234963,  -0.391515, 0.223306,  0.049513,  -0.985867};
  std::vector<__fp16> ref_out = {
    0.109160,  0.150761,  -0.221623, -0.340605, 0.341716,  0.192903,
    0.229677,  0.296728,  -0.246453, 0.030694,  -0.299190, -0.204716,
    0.418990,  -0.358029, -0.310309, -0.199024, -0.234911, 0.386668,
    -0.108879, 0.098724,  0.353685,  0.152306,  0.054675,  -0.253151,
    0.121229,  -0.048077, 0.057462,  0.393775,  0.436868,  -0.115650,
    0.494638,  -0.060525, -0.078396, 0.019140,  -0.078680, 0.571263};
  std::vector<__fp16> output(output_size);

  for (int n = 0; n < num_cache_head; ++n) {
    int chunk_size = head_dim;
    const _FP16 *in_ptr = in.data() + n * gqa_size;
    const _FP16 *vcache_ptr = vcache.data() + n * head_dim;
    _FP16 *out_ptr = output.data() + n * gqa_size * head_dim;
    nntrainer::compute_fp16vcache_transposed(row_num, in_ptr, vcache_ptr,
                                             out_ptr, num_cache_head, gqa_size,
                                             head_dim, chunk_size);
  }

  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(ref_out[i], output[i], 0.0004f);
  }
}
#endif

static void run_clamp_test(const unsigned int N, float lower_bound,
                           float upper_bound, bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_mul_time = (nanoseconds)0;
  nanoseconds mul_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> X = generate_random_vector<float, false>(N, -10, 10);
    std::vector<float> Y = generate_random_vector<float, false>(N);
    std::vector<float> Y_ref = generate_random_vector<float, false>(N);
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::__fallback_clamp(X.data(), Y_ref.data(), N, lower_bound,
                                  upper_bound);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_mul_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::clamp(X.data(), Y.data(), N, lower_bound, upper_bound);
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        mul_time += dt;
      }
    }

    auto mean_squared_error = compute_mse(1, N, Y_ref, Y, print);
    ASSERT_LE(mean_squared_error, 0.00001f);
  }

  if (print) {
    std::cout << "[INFO] clamp: TEST CNT: " << TEST_CNT << ", N: " << N
              << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
              << " ns, Average mul_time: " << mul_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, clamp_3072_0_1) {
  const unsigned int N = 3072;
  float lower_bound = 0.F;
  float upper_bound = 1.F;
  run_clamp_test(N, lower_bound, upper_bound, false);
}
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
