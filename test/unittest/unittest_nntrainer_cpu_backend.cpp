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
#include <fstream>
#include <filesystem>

#if defined(_WIN32)
#include <Sysinfoapi.h>
#endif

bool is_cpu_fully_utilized(int cpu_id) {
    std::ifstream cur_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) + "/cpufreq/scaling_cur_freq");
    std::ifstream max_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) + "/cpufreq/cpuinfo_max_freq");

    int cur_freq = 0, max_freq = 0;
    if (cur_file >> cur_freq && max_file >> max_freq) {
        float utilization_ratio = float(cur_freq) / max_freq;
        std::cout << "CPU utilization_ratio : " << utilization_ratio << "\n";
        return utilization_ratio > 0.95f; // 95%+ means likely fully utilized
    }

    return false;
}

void print_cpu_frequency() {
    for (int i = 0; i < 8; ++i) { // Change 8 to match your number of cores
        is_cpu_fully_utilized(i);
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/scaling_cur_freq";
        std::ifstream freq_file(path);
        if (freq_file.is_open()) {
            int freq_khz;
            freq_file >> freq_khz;
            std::cout << "CPU" << i << " Frequency: " << freq_khz / 1000.0 << " MHz\n";
        }
    }
}

void print_cpu_temperature() {
    for (const auto& entry : std::filesystem::directory_iterator("/sys/class/thermal/")) {
        std::string temp_path = entry.path().string() + "/temp";
        std::ifstream temp_file(temp_path);
        if (temp_file.is_open()) {
            int temp_millic;
            temp_file >> temp_millic;
            std::cout << entry.path().filename() << " Temp: " << temp_millic / 1000.0 << " °C\n";
        }
    }
}

void print_memory_usage() {
  std::ifstream statm("/proc/self/statm");
  if (statm.is_open()) {
    long pages_total, resident;
    statm >> pages_total >> resident;

#if defined(_WIN32)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    auto page_size = sysInfo.dwAllocationGranularity;
#else
    auto page_size = sysconf(_SC_PAGE_SIZE);
#endif
    long page_size_kb = page_size / 1024;

    std::cout << "Virtual Memory: " << pages_total * page_size_kb << " KB\n";
    std::cout << "Resident Set Size (RSS): " << resident * page_size_kb
              << " KB\n";
  }
}

template <typename T>
void transpose_matrix_util(T* src, T* dst, unsigned int M, unsigned int N){
  for (unsigned int i = 0; i < M; ++i){
    for (unsigned int j = 0; j < N; ++j){
      dst[j * M + i] = src[i * N + j];
    }
  }
}

template <typename T>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  std::mt19937 gen(42);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

template <typename T>
inline std::vector<T> generate_random_positive_vector(size_t size,
                                                      float min_val = 0.F,
                                                      float max_val = 0.5F) {
  std::random_device rd;
  std::mt19937 gen(42);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

template <typename T>
static inline std::vector<T> generate_homogeneous_vector(size_t size, T value) {
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = value;
  }
  return vec;
}

template <typename T> static inline void print_matrix(T *src, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << src[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
inline void print_matrix_partially(T *src, int M, int N, int partial_m = 5,
                                   int partial_n = 5, int partial_len = 5) {
  for (int k = 0; k < partial_len; ++k) {
    std::cout << src[partial_m * N + partial_n + k] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void print_vector_partially(T *src, int init_idx = 5,
                                   int partial_len = 5) {
  for (int k = 0; k < partial_len; ++k) {
    std::cout << src[init_idx + k] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
static inline void
print_matrix_partially_n(const std::string &name, const T *src, int M, int N,
                         int partial_m = 5, int partial_n = 5) {
  std::cout << name << ":" << std::endl;
  std::cout << "--------------------------" << std::endl;
  for (int i = 0; i < partial_m; ++i) {
    for (int j = 0; j < partial_n; ++j) {
      std::cout << src[i * N + j] << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "--------------------------" << std::endl;
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
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;

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

typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;

/**
 * @brief
 */
struct block_q4_Kx8_testonly {
  int16_t d[8];       // super-block scale for quantized scales
  int16_t dmin[8];    // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

static inline void print_q4_k_block_partially(void *block) {
  block_q4_K_testonly *b = (block_q4_K_testonly *)block;
  std::cout << "d : " << b->d << std::endl;
  std::cout << "dmin : " << b->dmin << std::endl;
  // std::cout << "qs : ";
  // for (int i = 0; i < 256/2; ++i){
  //     uint8_t packed_val = b->qs[i];
  //     uint8_t val1 = packed_val & 0x0F;
  //     uint8_t val2 = (packed_val >> 4) & 0x0F;
  //     std::cout << (int)val1 << " " << (int)val2 << " ";
  // }
  std::cout << "qs 5~8 : ";
  for (int i = 5; i < 8; ++i) {
    uint8_t packed_val = b->qs[i];
    uint8_t val1 = packed_val & 0x0F;
    uint8_t val2 = (packed_val >> 4) & 0x0F;
    std::cout << (int)val1 << " " << (int)val2 << " ";
  }
  std::cout << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_ele_add) {
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
                         std::vector<float> &ref_dst, std::vector<float> &dst) {
  auto mean_squared_error =
    mse<float, float>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  std::cout << "[INFO]            MSE: " << mean_squared_error
            << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
            << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  return mean_squared_error;
}

float test_gemm_q4_0(const uint32_t M, const uint32_t K,
                            const uint32_t N, const float *weights,
                            const float *activations,
                            std::vector<float> &ref_dst) {
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
  const int TC = 1;
  double gflops = 2 * M * N * K;
  std::vector<float> dst(M * N);
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  for (int tc = 0; tc < TC; ++tc){
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  }
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "[INFO] gemm_q4_0: " << dt.count() / TC << " ns " << " , gflops : " << 1e+9 * gflops / (dt.count() / TC)<< std::endl;

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst);
  return mean_squared_error;
}

static float test_gemm_q4_K(const uint32_t M, const uint32_t K,
                            const uint32_t N, const float *weights,
                            const float *activations,
                            std::vector<float> &ref_dst) {
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
  const int TC = 1;
  double gflops = 2 * M * N * K;
  std::vector<float> dst(M * N);
  // nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
  //                   N, dst.data(), N);
  // nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
  //                   N, dst.data(), N);
  // nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
  //                   N, dst.data(), N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  for (int tc = 0; tc < TC; ++tc){
    nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                       N, dst.data(), N);
  }
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  // auto dt_ms = duration_cast<milliseconds>(t2 - t1);
  std::cout << "[INFO] gemm_q4_K: " << dt.count() / TC << " ns , " << std::endl;
  ///@note Needs validation!

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst);
  
  std::vector<float> dst_trans(M * N);
  transpose_matrix_util<float>(dst.data(), dst_trans.data(), M, N);
  auto mean_squared_error_transpose = compute_mse(M, N, ref_dst, dst_trans);

  return mean_squared_error;
}

static void run_quant_test(const uint32_t M, const uint32_t K, const uint32_t N,
                           float &q0_k_mse, float &q4_k_mse) {
  std::cout << "[INFO] Quantization Test (M:" << M << ", K:" << K << ", N:" << N
            << ")" << std::endl;
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  // std::vector<float> activation = generate_homogeneous_vector<float>(M *
  // K, 2.0f); std::vector<float> weight = generate_homogeneous_vector<float>(N
  // * K, 1.0F);
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);

  // GROUND TRUTH TRANSB SGEMM for reference
const int TC = 1;
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc){
     nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "[INFO] sgemm : " << dt.count() / TC << " ns " << std::endl;

  // q0_k_mse = test_gemm_q4_0(M, K, N, weight.data(), activation.data(), ref_dst);
  q4_k_mse = test_gemm_q4_K(M, K, N, weight.data(), activation.data(), ref_dst);
}


// TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x3072) {
//   const unsigned int M = 1;
//   const unsigned int K = 3072;
//   const unsigned int N = 3072;
//   float q0_k_mse, q4_k_mse;
//   run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
//   // ASSERT_LE(q0_k_mse, 1.0f);
//   ASSERT_LE(q4_k_mse, 1.0f);
// }

// TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x8192) {
//   const unsigned int M = 1;
//   const unsigned int K = 3072;
//   const unsigned int N = 8192;
//   float q0_k_mse, q4_k_mse;
//   run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
//   // ASSERT_LE(q0_k_mse, 1.0f);
//   ASSERT_LE(q4_k_mse, 1.0f);
// }

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_256x3072x3072) {
  const unsigned int M = 256;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  // ASSERT_LE(q0_k_mse, 1.0f);
  ASSERT_LE(q4_k_mse, 1.0f);
}

// TEST(nntrainer_cpu_backend_standalone, quant_GEMM_256x3072x3072) {
//   const unsigned int M = 256;
//   const unsigned int K = 3072;
//   const unsigned int N = 3072;
//   float q0_k_mse, q4_k_mse;
//   run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
//   // ASSERT_LE(q0_k_mse, 1.0f);
//   ASSERT_LE(q4_k_mse, 1.0f);
// }

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_512x3072x3072) {
  const unsigned int M = 512;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  // ASSERT_LE(q0_k_mse, 2.0f);
  ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  // ASSERT_LE(q0_k_mse, 2.0f);
  ASSERT_LE(q4_k_mse, 2.0f);
}

#endif

/**
 * Room for optimization
 *
 *   1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
 *        - PRO : We can save the time for repacking the weight
 *        - CON : We need to save the weight in two different formats (q4_K and
 *   q4_K_8x8), and such kernel works for specific HWs.
 *    2. Pre-allocation of runtime quantized activation buffer
 *   3. ???
 */

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
