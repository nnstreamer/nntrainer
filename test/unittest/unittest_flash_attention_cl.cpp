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

TEST(attention_kernels, flash_attention_cl_fp16) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr size_t q_width = 1;
  static constexpr size_t q_height = 3072;
  static constexpr size_t q_channel = 3072;
  static constexpr size_t q_batch = 1;
  //
  static constexpr size_t q_elements = q_width * q_height * q_channel * q_batch;
  static constexpr size_t q_size = q_elements * sizeof(_FP16);

  static constexpr size_t k_width = 1;
  static constexpr size_t k_height = 256;
  static constexpr size_t k_channel = 3072;
  static constexpr size_t k_batch = 1;
  //
  static constexpr size_t k_elements = k_width * k_height * k_channel * k_batch;
  static constexpr size_t k_size = k_elements * sizeof(_FP16);

  static constexpr size_t v_width = 1;
  static constexpr size_t v_height = 256;
  static constexpr size_t v_channel = 3072;
  static constexpr size_t v_batch = 1;
  //
  static constexpr size_t v_elements = v_width * v_height * v_channel * v_batch;
  static constexpr size_t v_size = v_elements * sizeof(_FP16);

  static constexpr size_t o_width = 1;
  static constexpr size_t o_height = 3072;
  static constexpr size_t o_channel = 3072;
  static constexpr size_t o_batch = 1;
  //
  static constexpr size_t o_elements = o_width * o_height * o_channel * o_batch;
  static constexpr size_t o_size = o_elements * sizeof(_FP16);

 // auto random_init = false;
 // std::random_device rd;
 // auto init_val = random_init ? rd() : 42;
 // std::mt19937 gen = random_init ? std::mt19937(init_val) : std::mt19937(rd());
//
 // std::uniform_real_distribution<float> dist(min_val, max_val);

  auto random_fp16 = [](const _FP16 number) -> _FP16 {
    return static_cast<_FP16>(1.0f);
  };

  std::vector<_FP16> Q(q_elements, static_cast<_FP16>(0.0f));
  std::vector<_FP16> K(k_elements, static_cast<_FP16>(0.0f));
  std::vector<_FP16> V(v_elements, static_cast<_FP16>(0.0f));
  std::vector<_FP16> O(o_elements, static_cast<_FP16>(0.0f));

  std::transform(std::begin(Q), std::end(Q), std::begin(Q), random_fp16);
  std::transform(std::begin(K), std::end(K), std::begin(K), random_fp16);
  std::transform(std::begin(V), std::end(V), std::begin(V), random_fp16);

  // for (size_t i = 0; i < elements; i++) {
  //   Q[i] = static_cast<_FP16>(5.0f - 10.0f * rand() / RAND_MAX);
  //   K[i] = static_cast<_FP16>(5.0f - 10.0f * rand() / RAND_MAX);
  //   V[i] = static_cast<_FP16>(5.0f - 10.0f * rand() / RAND_MAX);
  // }

  void *Q_svm = blas_cc->context_inst_.createSVMRegion(q_size);
  void *K_svm = blas_cc->context_inst_.createSVMRegion(k_size);
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

  nntrainer::flash_attention_cl_fp16(
    // clang-format off
    Q_svm, K_svm, V_svm, O_svm,
    q_width, q_height, q_channel, q_batch,
    k_width, k_height, k_channel, k_batch,
    v_width, v_height, v_channel, v_batch,
    o_width, o_height, o_channel, o_batch
    // clang-format on
  );

  blas_cc->command_queue_inst_.enqueueSVMMap(O_svm, o_size, true);

  for (size_t i = 0; i < 8; i++) {
    std::printf(
      "[%zu] = %.6f | [%zu] = %.6f\n",
      // clang-format off
                i,
                static_cast<float>(reinterpret_cast<_FP16*>(O_svm)[i]), 
                o_elements - 1 - i,
                static_cast<float>(reinterpret_cast<_FP16*>(O_svm)[o_elements - 1 - i])
      // clang-format on
    );
  }

  for (size_t i = 0; i < o_elements; i++) {
    if (static_cast<float>(reinterpret_cast<_FP16 *>(O_svm)[i]) == 0.0f) {
      std::printf("First zero at : %zu\n", i);
      break;
    }
  }

  for (size_t i = 0; i < o_elements; i++) {
    if (static_cast<float>(
          reinterpret_cast<_FP16 *>(O_svm)[o_elements - 1 - i]) != 0.0f) {
      std::printf("First zero at : %zu\n", i);
      break;
    }
  }

  size_t zeros = 0;
  for (size_t i = 0; i < o_elements; i++) {
    if (static_cast<float>(reinterpret_cast<_FP16 *>(O_svm)[i]) == 0.0f) {
      zeros++;
    }
  }

  std::printf("Cout of zeros : %zu / %zu = %.3f\n", zeros, o_elements,
              100.0f * static_cast<float>(zeros) /
                static_cast<float>(o_elements));

  std::printf("Cout of non : %zu / %zu = %.3f\n", o_elements - zeros,
              o_elements,
              100.0f * static_cast<float>(o_elements - zeros) /
                static_cast<float>(o_elements));
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
