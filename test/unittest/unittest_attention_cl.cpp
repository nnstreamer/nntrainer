// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	unittest_attention_kernels_cl.cpp
 * @date	28 August 2024
 * @brief	Test setup for blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <fstream>
#include <gtest/gtest.h>
#include <type_traits>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <attention_kernels.h>
#include <cl_context.h>
#include <layer_context.h>
#include <tensor.h>

#include <algorithm>

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

TEST(attention_kernels, attention_cl_fp16) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  // static constexpr size_t m = 3072;
  // static constexpr size_t n = 3072;
  // static constexpr size_t d_k = 256;
  // static constexpr size_t d_v = 3072;

  static constexpr size_t m = 64;
  static constexpr size_t n = 48;
  static constexpr size_t d_k = 32;
  static constexpr size_t d_v = 16;

  static constexpr size_t q_width = m;
  static constexpr size_t q_height = d_k;
  static constexpr size_t q_elements = q_width * q_height;
  static constexpr size_t q_size = q_elements * sizeof(_FP16);

  static constexpr size_t k_width = n;
  static constexpr size_t k_height = d_k;
  static constexpr size_t k_elements = k_width * k_height;
  static constexpr size_t k_size = k_elements * sizeof(_FP16);

  static constexpr size_t sm_width = m;
  static constexpr size_t sm_height = n;
  static constexpr size_t sm_elements = sm_width * sm_height;
  static constexpr size_t sm_size = sm_elements * sizeof(_FP16);

  static constexpr size_t v_width = n;
  static constexpr size_t v_height = d_v;
  static constexpr size_t v_elements = v_width * v_height;
  static constexpr size_t v_size = v_elements * sizeof(_FP16);

  static constexpr size_t o_width = m;
  static constexpr size_t o_height = d_v;
  static constexpr size_t o_elements = o_width * o_height;
  static constexpr size_t o_size = o_elements * sizeof(_FP16);

  std::vector<_FP16> Q(q_elements, static_cast<_FP16>(1.0f));
  std::vector<_FP16> K(k_elements, static_cast<_FP16>(2.0f));
  std::vector<_FP16> S(sm_elements, static_cast<_FP16>(0.0f));
  std::vector<_FP16> V(v_elements, static_cast<_FP16>(3.0f));
  std::vector<_FP16> O(o_elements, static_cast<_FP16>(0.0f));

  void *Q_svm = blas_cc->context_inst_.createSVMRegion(q_size);
  void *K_svm = blas_cc->context_inst_.createSVMRegion(k_size);
  void *S_svm = blas_cc->context_inst_.createSVMRegion(sm_size);
  void *V_svm = blas_cc->context_inst_.createSVMRegion(v_size);
  void *O_svm = blas_cc->context_inst_.createSVMRegion(o_size);

  blas_cc->command_queue_inst_.enqueueSVMMap(Q_svm, q_size, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(K_svm, k_size, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(V_svm, v_size, false);

  std::memcpy(Q_svm, Q.data(), q_size);
  std::memcpy(K_svm, K.data(), k_size);
  std::memcpy(V_svm, V.data(), v_size);

  blas_cc->command_queue_inst_.enqueueSVMUnmap(Q_svm);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(K_svm);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(V_svm);

  nntrainer::attention_cl_fp16(Q_svm, K_svm, S_svm, V_svm, O_svm, m, n,
                                     d_k, d_v);

  blas_cc->command_queue_inst_.enqueueSVMMap(Q_svm, q_size, true);
  blas_cc->command_queue_inst_.enqueueSVMMap(K_svm, k_size, true);
  blas_cc->command_queue_inst_.enqueueSVMMap(S_svm, sm_size, true);
  blas_cc->command_queue_inst_.enqueueSVMMap(V_svm, v_size, true);
  blas_cc->command_queue_inst_.enqueueSVMMap(O_svm, o_size, true);

#if 0
  std::cout << "Q\n";
  for (uint row = 0; row < m; row++) {
    for (uint column = 0; column < d_k; column++) {
      std::printf("%.4f ", static_cast<float>(reinterpret_cast<_FP16 *>(
                             Q_svm)[row * d_k + column]));
    }
    std::printf("\n");
  }

  std::cout << "K\n";
  for (uint row = 0; row < n; row++) {
    for (uint column = 0; column < d_k; column++) {
      std::printf("%.4f ", static_cast<float>(reinterpret_cast<_FP16 *>(
                             K_svm)[row * d_k + column]));
    }
    std::printf("\n");
  }

  std::cout << "S\n";
  for (uint row = 0; row < m; row++) {
    for (uint column = 0; column < n; column++) {
      std::printf("%.4f ", static_cast<float>(reinterpret_cast<_FP16 *>(
                             S_svm)[row * n + column]));
    }
    std::printf("\n");
  }

  std::cout << "V\n";
  for (uint row = 0; row < n; row++) {
    for (uint column = 0; column < d_v; column++) {
      std::printf("%.4f ", static_cast<float>(reinterpret_cast<_FP16 *>(
                             V_svm)[row * d_v + column]));
    }
    std::printf("\n");
  }
#endif

  std::cout << "O\n";
  for (uint row = 0; row < m; row++) {
    for (uint column = 0; column < d_v; column++) {
      std::printf("%.4f ", static_cast<float>(reinterpret_cast<_FP16 *>(
                             O_svm)[row * d_v + column]));
    }
    std::printf("\n");
  }
}

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
