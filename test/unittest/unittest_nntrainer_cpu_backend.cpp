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

#ifdef ENABLE_GGML

TEST(nntrainer_cpu_backend_standalone, q4_K_quantization) {
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

static float compute_mse(const uint32_t M, const uint32_t N,
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

static float test_gemm_q4_0(const uint32_t M, const uint32_t K,
                            const uint32_t N, const float *weights,
                            const float *activations,
                            std::vector<float> &ref_dst, bool print = false) {
  // needed to initialize f16 tables
  nntrainer::init_backend();

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
    std::cout << "[INFO] gemm_q4_0: " << dt.count() << " ns " << std::endl;
  }

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst);
  return mean_squared_error;
}

static float test_gemm_q4_K(const uint32_t M, const uint32_t K,
                            const uint32_t N, const float *weights,
                            const float *activations,
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
    std::cout << "[INFO] gemm_q4_K: " << dt.count() << " ns " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst);
  return mean_squared_error;
}

static void run_quant_test(const uint32_t M, const uint32_t K, const uint32_t N,
                           float &q0_k_mse, float &q4_k_mse,
                           bool print = false) {
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
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns " << std::endl;
  }
  q0_k_mse = test_gemm_q4_0(M, K, N, weight.data(), activation.data(), ref_dst);
  q4_k_mse = test_gemm_q4_K(M, K, N, weight.data(), activation.data(), ref_dst);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_256x1024x512) {
  const unsigned int M = 256;
  const unsigned int K = 1024;
  const unsigned int N = 512;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_512x768x1024) {
  const unsigned int M = 512;
  const unsigned int K = 768;
  const unsigned int N = 1024;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x1536x1536) {
  const unsigned int M = 1024;
  const unsigned int K = 1536;
  const unsigned int N = 1536;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x1536x5760) {
  const unsigned int M = 1024;
  const unsigned int K = 1536;
  const unsigned int N = 5760;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x3072x3072) {
  const unsigned int M = 457;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  // ASSERT_LE(q0_k_mse, 1.5f);
  ASSERT_LE(q4_k_mse, 1.5f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x3072x3072) {
  const unsigned int M = 458;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  // ASSERT_LE(q0_k_mse, 1.5f);
  ASSERT_LE(q4_k_mse, 1.5f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x3072x3072) {
  const unsigned int M = 459;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  // ASSERT_LE(q0_k_mse, 1.5f);
  ASSERT_LE(q4_k_mse, 1.5f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x512x768) {
  const unsigned int M = 1;
  const unsigned int K = 512;
  const unsigned int N = 768;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x512) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 512;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x1024) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 1024;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x1536x1536) {
  const unsigned int M = 1;
  const unsigned int K = 1536;
  const unsigned int N = 1536;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x1536x5760) {
  const unsigned int M = 1;
  const unsigned int K = 1536;
  const unsigned int N = 5760;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  ASSERT_LE(q4_k_mse, 1.0f);
}

static void run_vec_dot_test(const uint32_t K) {
  const int TEST_CNT = 100;
  nanoseconds ref_time = (nanoseconds)0;
  nanoseconds q6_k_time = (nanoseconds)0;
  float ref_result, result;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> activation = generate_random_vector<float, true>(K);
    std::vector<float> weight = generate_random_vector<float, true>(K);

    {
      // GROUND TRUTH sdot for reference
      auto t1 = high_resolution_clock::now();
      ref_result = nntrainer::sdot(K, weight.data(), 1, activation.data(), 1);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      // std::cout << "[INFO] sdot : " << dt.count() << " ns " << std::endl;
      if (i >= 0) { // skip the first run
        ref_time += dt;
      }
    }

    // Quantization of weights
    int64_t num_blocks = K / 256;
    size_t q6_k_data_size = num_blocks * sizeof(block_q6_K_testonly);
    std::vector<char> q6_K_weight = std::vector<char>(q6_k_data_size);
    nntrainer::quantize_row_q6_K(weight.data(), q6_K_weight.data(), K);

    std::vector<float> weight_dequant(K, 0.0f);
    nntrainer::dequantize_row_q6_K(q6_K_weight.data(), weight_dequant.data(),
                                   K);

    const auto weights_quant_dequant_cos_sim =
      cosine_similarity(weight.data(), weight_dequant.data(), K);
    const auto weights_quant_dequant_max_diff =
      find_max_diff(weight.data(), weight_dequant.data(), 1, K);

    EXPECT_NEAR(weights_quant_dequant_cos_sim, 1.0f, 0.001f);
    EXPECT_NEAR(weights_quant_dequant_max_diff, 0.0f, 0.025f);

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
      // std::cout << "[INFO] dot_q6_K_q8_K : " << dt.count() << " ns " <<
      // std::endl;
      if (i >= 0) { // skip the first run
        q6_k_time += dt;
      }
    }
    // printf("res: %f  res: %f, diff: %f\n", result, ref_result, result -
    // ref_result);
    EXPECT_NEAR(result, ref_result, 0.25 * K / 256);
  }

  std::cout << "[INFO] dot_q6_K_q8_K: TEST CNT: " << TEST_CNT << ", K: " << K
            << ", Average ref_time: " << ref_time.count() / TEST_CNT
            << " ns, Average q6_k_time: " << q6_k_time.count() / TEST_CNT
            << " ns " << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_256) {
  const uint32_t K = 256;
  run_vec_dot_test(K);
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

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_25600) {
  const uint32_t K = 25600;
  run_vec_dot_test(K);
}

#endif

int main(int argc, char **argv) {
  int result = -1;
#ifdef ENABLE_GGML
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
#else
  result = 0;
#endif
  return result;
}
