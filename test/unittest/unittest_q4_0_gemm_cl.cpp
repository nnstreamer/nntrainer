/**
 * Copyright (C) 2025 Daekyoung Jung <daekyoung.jung@gmail.com>
 *
 * @file        unittest_q4_0_gemm_cl.cpp
 * @date        03 June 2020
 * @brief       Unit test utility for tensor.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Daekyoung Jung <daekyoung.jung@gmail.com>
 * @bug         No known bugs
 */
#include "blas_kernels.h"
#include "cl_context.h"
#include "generate_random_vector.h"
#include "nntrainer_test_util.h"
#include "q4_0x8_tool.h"

#include <gtest/gtest.h>

#include <chrono>

#define QK4_0 32
#define GGML_ASSERT assert

using namespace std::chrono;
using namespace nntrainer;

/**
 * @brief struct for q4_0x8
 *
 */
struct block_q4_0x8 {
  unsigned short d[8];
  unsigned char qs[128];
};

/**
 * @brief struct for q4_0
 *
 */
struct block_q4_0 {
  unsigned short d[1];
  unsigned char qs[16];
};

typedef void (*convert_q4_0x8)(const void *, unsigned short *, unsigned char *,
                               int, int);

void convert_block_q4_0_noshuffle(unsigned char *dst, unsigned char *qs,
                                  int N) {
  for (int g_id = 0; g_id < N / 16; ++g_id, qs += 16, dst += 16) {
    for (int i = 0; i < 8; ++i) {
      unsigned char x0 = qs[2 * i + 0];
      unsigned char x1 = qs[2 * i + 1];
      dst[i + 0] =
        (unsigned char)(x0 & 0x0F) | (unsigned char)((x1 & 0x0F) << 4);
      dst[i + QK4_0 / 4] =
        (unsigned char)((x0 & 0xF0) >> 4) | (unsigned char)(x1 & 0xF0);
    }
  }
}

static void run_q4_0x8_convert_shuffle_test(const uint32_t M, const uint32_t K,
                                            const uint32_t N,
                                            convert_q4_0x8 func) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 16;

  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  std::vector<float> weight = generate_random_vector<float, false>(N * K);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> cpu_q4_dst(M * N, 0.0f);

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

  auto t1 = high_resolution_clock::now();
  flatten_block_q4_0_cl(weight_q4_0, src0_q_ref, src0_d_ref, n_blk);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<microseconds>(t2 - t1);

  auto t3 = high_resolution_clock::now();
  func(weight_q4_0_8, (unsigned short *)src0_d, (unsigned char *)src0_q,
       n_blk / 8, K);
  auto t4 = high_resolution_clock::now();

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

    std::cout << " - time : flatten_block_q4_0_cl = "
              << dt.count() / (run_count * 1.0f) << " us" << std::endl;
    std::cout << " - time : convert_q4_0x8_shuffle = "
              << convert_dt.count() / (run_count * 1.0f) << " us" << std::endl;
  }
}

#define DECLARE_q4_0x8_convert_test2_M_K_N(M, K, N, FUNC)                      \
  TEST(nntrainer_cl_helper,                                                    \
       run_q4_0x8_convert_test2_##M##_##K##_##N##_##FUNC) {                    \
    run_q4_0x8_convert_shuffle_test(M, K, N, FUNC);                            \
  }

DECLARE_q4_0x8_convert_test2_M_K_N(28, 512, 256, convert_q4_0x8_shuffle);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 512, 256, convert_q4_0x8_shuffle_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 1024, 256, convert_q4_0x8_shuffle);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 1024, 256, convert_q4_0x8_shuffle_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 512, 512, convert_q4_0x8_shuffle);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 512, 512, convert_q4_0x8_shuffle_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 256, convert_q4_0x8_shuffle);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 256, convert_q4_0x8_shuffle_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 8192, convert_q4_0x8_shuffle);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 8192, convert_q4_0x8_shuffle_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 8192, 3072, convert_q4_0x8_shuffle);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 8192, 3072, convert_q4_0x8_shuffle_omp);

DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 3072, convert_q4_0x8_shuffle);
DECLARE_q4_0x8_convert_test2_M_K_N(28, 3072, 3072, convert_q4_0x8_shuffle_omp);

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
