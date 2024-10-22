// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	unittest_blas_kernels_cl.cpp
 * @date	6 June 2024
 * @brief	Test setup for blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <fstream>
#include <gtest/gtest.h>
#include <type_traits>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <blas_kernel_interface.h>
#include <cl_context.h>
#include <layer_context.h>
#include <tensor.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

static void setUpGpuContext() { auto &ac = nntrainer::ClContext::Global(); }

TEST(blas_kernels, dotCL_sgemv_M_1_1) {
  setUpGpuContext();
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 2048;
  int width_b = 768;

  bool transA = false;
  bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 768;
  int width_b = 2048;

  bool transA = false;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_1) {
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 2048;

  int height_b = 768;
  int width_b = 1;

  bool transA = true;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_2) {
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 2048;

  int height_b = 2048;
  int width_b = 1;

  bool transA = false;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_n) {

  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 768;
  int width_b = 2048;

  bool transA = true;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  EXPECT_THROW(dotCl(A_fp32, B_fp32, transA, transB), std::runtime_error);
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
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // fp16
  multiplyCl(input, 0.1);

  // fp32
  multiplyCl(input_fp32, 0.1);

  float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
                                   input_fp32.getData<float>(), input.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    input.getData<__fp16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_1024_noTrans) {
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 768;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_2048_transB) {
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 2048;
  int width_b = 768;

  bool transA = false;
  bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_1024_transA) {
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 50;

  int height_b = 768;
  int width_b = 1024;

  bool transA = true;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_2048_transAB) {
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 50;

  int height_b = 2048;
  int width_b = 768;

  bool transA = true;
  bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
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
