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

TEST(nntrainer_Tensor, dot) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  // conditions for fp16 sdot call:
  // this->(batch * channel * height) = arg->(width) = 1;

  size_t width = 23;

  __fp16 a_data[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                     12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);
  __fp16 b_data[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                     12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_2(
    nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp16), b_data);

  float a_data_fp32[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                         12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);
  float b_data_fp32[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                         12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_fp32_2(
    nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp32), b_data_fp32);

  nntrainer::Tensor result_neon;
  nntrainer::Tensor result_fp32;

  // NEON fp16
  result_neon = input.dot(input_2, false, false);

  // fp32
  result_fp32 = input_fp32.dot(input_fp32_2, false, false);

  float mseErrorNeon =
    mse<__fp16>(result_neon.getData<__fp16>(), result_fp32.getData<float>(),
                result_neon.size());

  double cosSimNeon =
    cosine_similarity<__fp16>(result_neon.getData<__fp16>(),
                              result_fp32.getData<float>(), result_neon.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, l2norm) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t width = 23;

  __fp16 a_data[] = {0,   1.2, 2, 3.4, 4.1, 5.3, 2.9, 2.1, 1.4, 1.6, 0, 2.7,
                     2.3, 1,   2, 1.1, 3.1, 1.1, 2.8, 3.2, 2,   3.6, 1};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);

  float a_data_fp32[] = {0,   1.2, 2, 3.4, 4.1, 5.3, 2.9, 2.1, 1.4, 1.6, 0, 2.7,
                         2.3, 1,   2, 1.1, 3.1, 1.1, 2.8, 3.2, 2,   3.6, 1};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);

  __fp16 result_neon;
  float result_fp32;

  // NEON fp16
  result_neon = input.l2norm();

  // fp32
  result_fp32 = input_fp32.l2norm();

  // absolute error
  const float epsilon = 1e-2;

  EXPECT_NEAR(result_neon, result_fp32, epsilon);
}

TEST(nntrainer_Tensor, multiply_i) {
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
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // NEON fp16
  int result = input.multiply_i(0.1);

  // fp32
  result = input_fp32.multiply_i(0.1);

  float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
                                   input_fp32.getData<float>(), input.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    input.getData<__fp16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, copy) {
  int batch = 1;
  int channel = 1;
  int height = 2;
  int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  nntrainer::Tensor output;
  nntrainer::Tensor output_fp32;

  // NEON fp16
  output.copy(input);

  // fp32
  output_fp32.copy(input_fp32);

  float mseErrorNeon = mse<__fp16>(output.getData<__fp16>(),
                                   output_fp32.getData<float>(), output.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    output.getData<__fp16>(), output_fp32.getData<float>(), output.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, max_abs) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t width = 25;

  __fp16 a_data[] = {0,   1.2, 2,   3.4, 4.1, 5.3, 2.9, 2.1, 1.4,
                     1.6, 0,   2.7, 2.3, 1,   2,   1.1, 3.1, 1.1,
                     2.8, 3.2, 2,   3.6, 1,   2.8, 7.9};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);

  float a_data_fp32[] = {0,   1.2, 2,   3.4, 4.1, 5.3, 2.9, 2.1, 1.4,
                         1.6, 0,   2.7, 2.3, 1,   2,   1.1, 3.1, 1.1,
                         2.8, 3.2, 2,   3.6, 1,   2.8, 7.9};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);

  __fp16 result_neon;
  float result_fp32;

  // NEON fp16
  result_neon = input.max_abs();

  // fp32
  result_fp32 = input_fp32.max_abs();

  // absolute error
  const float epsilon = 1e-2;

  EXPECT_NEAR(result_neon, result_fp32, epsilon);
}

TEST(nntrainer_Tensor, sum_sgemv_transpose) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_copy(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_copy, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result0_fp32 = input_fp32.sum(0);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, sum_sgemv) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_copy(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_copy, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  nntrainer::Tensor result0 = input.sum_by_batch();
  nntrainer::Tensor result0_fp32 = input_fp32.sum_by_batch();

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
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
