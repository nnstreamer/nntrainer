// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file        unittest_nntrainer_tensor_neon_fp16.cpp
 * @date        03 August 2023
 * @brief       Unit test utility for tensor with NEON __fp16 support for ARM.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Debadri Samaddar <s.debadri@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

TEST(nntrainer_Tensor, add_i) {
  int batch = 1;
  int channel = 1;
  int height = 2;
  int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_copy(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_copy, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // NEON fp16
  int result = input.add_i(input_copy);

  // fp32
  result = input_fp32.add_i(input_fp32);

  float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
                                   input_fp32.getData<float>(), input.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    input.getData<__fp16>(), input_fp32.getData<float>(), input.size());
  
  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
