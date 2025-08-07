/**
 * Copyright (C) 2024 Daekyoung Jung <daekyoung.jung@gmail.com>
 *
 * @file	unittest_cl_helper.cpp
 * @date	07 August 2025
 * @brief	unittest for blas_kernel_helper.cpp
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Daekyoung Jung <daekyoung.jung@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "blas_kernel_helper.h"
#include "blas_kernels.h"
#include "nntrainer_test_util.h"
#include <cl_context.h>

#include <gtest/gtest.h>

#include <cstring>

typedef unsigned short us;
typedef unsigned long long ull;

using namespace nntrainer;
using namespace std::chrono;

typedef void (*convert_q4_0x8)(const void *, unsigned short *, unsigned char *,
                               int, int);

#ifdef ENABLE_GGML
template <typename T, bool random_init = false>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

static void run_q4_0x8_convert_test2(const uint32_t M, const uint32_t K,
                                     const uint32_t N, convert_q4_0x8 func) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  auto debug_print_beg_end = [M, K, N](const float *const data,
                                       const uint32_t count = 12) {
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

  static constexpr uint32_t run_count = 16;

  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  std::vector<float> weight = generate_random_vector<float, false>(N * K);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> cpu_q4_dst(M * N, 0.0f);

  struct block_q4_0 {
    uint16_t d[1];
    uint8_t qs[16];
  };

  int64_t block_size = 32;
  int64_t q4_0_tile_size = sizeof(block_q4_0);
  int64_t n_blk = (K * N) / block_size;
  size_t data_size = q4_0_tile_size * n_blk;

  // Generate result from SGEMM
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  // Initialize data
  void *gpu_q4_dst =
    blas_cc->context_inst_.createSVMRegion(M * N * sizeof(float));
  void *weight_q4_0 = blas_cc->context_inst_.createSVMRegion(data_size);
  void *weight_q4_0_8 = blas_cc->context_inst_.createSVMRegion(data_size);

  blas_cc->command_queue_inst_.enqueueSVMMap(weight_q4_0, data_size, false);

  float *weight_f32 = weight.data();

  float *acti_f32 =
    (float *)blas_cc->context_inst_.createSVMRegion(M * K * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(acti_f32, M * K * sizeof(float),
                                             false);

  for (unsigned int i = 0; i < M * K; ++i) {
    acti_f32[i] = activation[i];
  }

  /// Quantize weight data
  nntrainer::quantize_q4_0(weight_f32, weight_q4_0, N, K, nullptr);

  // GPU Q4_0 GEMM
  void *src0_q_ref =
    blas_cc->context_inst_.createSVMRegion(n_blk * sizeof(uint8_t) * 16);
  void *src0_d_ref =
    blas_cc->context_inst_.createSVMRegion(n_blk * sizeof(uint16_t));

  void *src0_q =
    blas_cc->context_inst_.createSVMRegion(n_blk * sizeof(uint8_t) * 16);
  void *src0_d =
    blas_cc->context_inst_.createSVMRegion(n_blk * sizeof(uint16_t));

  nntrainer::repack_q4_0_to_q4_0_8(weight_q4_0_8, weight_q4_0, data_size, N, K);

  // CPU Q4_0 GEMM
  auto t1 = high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0(M, N, K, acti_f32, K, weight_q4_0_8, N,
                         cpu_q4_dst.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<milliseconds>(t2 - t1);

  flatten_block_q4_0_cl(weight_q4_0, src0_q_ref, src0_d_ref, n_blk);

  auto t3 = high_resolution_clock::now();
  func(weight_q4_0_8, (unsigned short *)src0_d, (unsigned char *)src0_q,
       n_blk / 8, K);
  auto t4 = high_resolution_clock::now();

  auto t5 = high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl2(src0_q, src0_d, acti_f32, (float *)gpu_q4_dst, M,
                             N, K);
  }
  auto t6 = high_resolution_clock::now();

  for (int i = 0; i < n_blk; ++i) {
    unsigned short *ref = (unsigned short *)src0_d_ref;
    unsigned short *sample = (unsigned short *)src0_d;

    ASSERT_EQ(ref[i], sample[i]) << "i: " << i << std::endl;
  }
  for (int i = 0; i < n_blk; ++i) {
    for (int j = 0; j < 16; ++j) {
      ASSERT_EQ(((unsigned char *)src0_q_ref)[i * 16 + j],
                ((unsigned char *)src0_q)[i * 16 + j])
        << "i: " << i << ", j: " << j << std::endl;
    }
  }

  blas_cc->context_inst_.releaseSVMRegion(src0_q);
  blas_cc->context_inst_.releaseSVMRegion(src0_d);
  blas_cc->context_inst_.releaseSVMRegion(src0_q_ref);
  blas_cc->context_inst_.releaseSVMRegion(src0_d_ref);
  auto convert_dt = duration_cast<microseconds>(t4 - t3);
  auto gpu_dt = duration_cast<milliseconds>(t6 - t5);

  // Compute reports
  {
    uint32_t first_zero_index = UINT32_MAX;
    int zeros = 0;
    int nans = 0;

    for (uint32_t i = 0; i < M * N; ++i) {
      if (((float *)gpu_q4_dst)[i] == 0) {
        zeros++;
        if (first_zero_index == UINT32_MAX) {
          first_zero_index = i;
        }
      }

      if (std::isnan(((float *)gpu_q4_dst)[i])) {
        nans++;
      }
    }

    const auto data_size_mb = data_size / (1024 * 1024.0f);

    std::cout << "Q4_0 GEMM : " << M << " x " << K << " x " << N << std::endl;
    std::cout << " - time : CPU = " << dt.count() / (run_count * 1.0f) << " ms"
              << std::endl;
    std::cout << " - time : Convert = "
              << convert_dt.count() / (run_count * 1.0f) << " us" << std::endl;
    std::cout << " - time : GPU = " << gpu_dt.count() / (run_count * 1.0f)
              << " ms" << std::endl;
    std::cout << " - sample : REF = ";
    debug_print_beg_end(ref_dst.data());
    std::cout << std::endl;
    std::cout << " - sample : CPU = ";
    debug_print_beg_end(cpu_q4_dst.data());
    std::cout << std::endl;
    std::cout << " - sample : GPU = ";
    debug_print_beg_end((float *)gpu_q4_dst);
    std::cout << std::endl;
    std::cout << " - zeros : " << zeros << " / " << M * N << " [ "
              << zeros * 100.0f / float(M * N) << " %] - first at [ "
              << first_zero_index << " ]" << std::endl;
    std::cout << " - nans : " << nans << " / " << M * N << " [ "
              << nans * 100.0f / float(M * N) << " %]" << std::endl;
  }

  // Release data
  blas_cc->context_inst_.releaseSVMRegion(gpu_q4_dst);
  blas_cc->context_inst_.releaseSVMRegion(weight_q4_0);
  blas_cc->context_inst_.releaseSVMRegion(weight_q4_0_8);
  blas_cc->context_inst_.releaseSVMRegion(acti_f32);
}

#define DECLARE_q4_0x8_convert_test2_M_K_N(M, K, N, FUNC)                      \
  TEST(nntrainer_cl_helper,                                                    \
       run_q4_0x8_convert_test2_##M##_##K##_##N##_##FUNC) {                    \
    run_q4_0x8_convert_test2(M, K, N, FUNC);                                   \
  }

DECLARE_q4_0x8_convert_test2_M_K_N(28, 512, 256, convert_q4_0x8_st);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 512, 256, convert_q4_0x8_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 1024, 256, convert_q4_0x8_st);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 1024, 256, convert_q4_0x8_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 512, 512, convert_q4_0x8_st);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 512, 512, convert_q4_0x8_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 256, convert_q4_0x8_st);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 256, convert_q4_0x8_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 8192, convert_q4_0x8_st);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 8192, convert_q4_0x8_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 8192, 3072, convert_q4_0x8_st);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 8192, 3072, convert_q4_0x8_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 3072, convert_q4_0x8_st);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 3072, convert_q4_0x8_omp);

#endif

GTEST_API_ int main(int argc, char **argv) {
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
