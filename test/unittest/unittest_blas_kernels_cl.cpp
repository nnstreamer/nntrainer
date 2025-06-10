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
#include <random>
#include <type_traits>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <tensor.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

TEST(blas_kernels, dotCL_sgemv_M_1_1) {
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
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

TEST(blas_kernels, dotCL_sgemv_N_1_M_1_1) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 1;
  int width_b = 768;

  bool transA = false;
  bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  auto gen_data = [](nntrainer::Tensor x) {
    auto ptr = x.getData();
    for (unsigned int i = 0; i < x.size(); ++i) {
      ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  };

  gen_data(A_fp32), gen_data(B_fp32);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float err = mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-5 * width;

  EXPECT_IN_RANGE(err, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_M_1_2) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 1;
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-5 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_noTrans) {
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transB) {
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_transA) {
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transAB) {
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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, addition_i) {

  int batch = 12;
  int channel = 1;
  int height = 26;
  int width = 26;

  int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch_b, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor C_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor D_fp32(batch_b, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);
  GEN_TEST_INPUT(C_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(D_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  A_fp32.add_i(B_fp32);
  add_i_cl(C_fp32, D_fp32);

  float mseError =
    mse<float>(A_fp32.getData<float>(), C_fp32.getData<float>(), A_fp32.size());

  double cosSim = cosine_similarity<float>(
    A_fp32.getData<float>(), C_fp32.getData<float>(), A_fp32.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, l2norm) {
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 768;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  GEN_TEST_INPUT(B_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  float gpu_result = nrm2Cl(A_fp32);
  float cpu_result = B_fp32.l2norm();

  EXPECT_FLOAT_EQ(gpu_result, cpu_result);
}

TEST(blas_kernels, absolute_sum) {
  int batch = 1;
  int channel = 1;
  int height = 32;
  int width = 32;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  float cpu_result = 0.0f;
  for (size_t i = 0; i < A_fp32.size(); ++i) {
    cpu_result += fabs(A_fp32.getData<float>()[i]);
  }

  float gpu_result = asumCl(A_fp32);

  EXPECT_FLOAT_EQ(cpu_result, gpu_result);
}

#ifdef ENABLE_FP16

TEST(blas_kernels, dotCL_sgemv_M_1_1_fp16) {
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

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp16, B_fp16, transA, transB);
  nntrainer::Tensor C_fp16 = A_fp16.dot(B_fp16, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2_fp16) {
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

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp16, B_fp16, transA, transB);
  nntrainer::Tensor C_fp16 = A_fp16.dot(B_fp16, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_1_fp16) {
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

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp16, B_fp16, transA, transB);
  nntrainer::Tensor C_fp16 = A_fp16.dot(B_fp16, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_2_fp16) {
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

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = dotCl(A_fp16, B_fp16, transA, transB);
  nntrainer::Tensor C_fp16 = A_fp16.dot(B_fp16, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_n_fp16) {

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

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp16, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  EXPECT_THROW(dotCl(A_fp16, B_fp16, transA, transB), std::runtime_error);
}

TEST(blas_kernels, sgemv_3072_20120_noTrans) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 3072;

  int height_b = 3072;
  int width_b = 20120;

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

  float mseError =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSim = cosine_similarity<float>(C.getData<float>(),
                                           C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, multiply_i) {
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

  float mseError = mse<_FP16>(input.getData<_FP16>(),
                              input_fp32.getData<float>(), input.size());

  double cosSim = cosine_similarity<_FP16>(
    input.getData<_FP16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_noTrans_fp16) {
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

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);

  nntrainer::Tensor C = dotCl(A, B, transA, transB);
  nntrainer::Tensor C_fp16 = A.dot(B, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transB_fp16) {
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

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);

  nntrainer::Tensor C = dotCl(A, B, transA, transB);
  nntrainer::Tensor C_fp16 = A.dot(B, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_transA_fp16) {
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

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);

  nntrainer::Tensor C = dotCl(A, B, transA, transB);
  nntrainer::Tensor C_fp16 = A.dot(B, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp16.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp16.getData<_FP16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transAB_fp16) {
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

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);

  nntrainer::Tensor C = dotCl(A, B, transA, transB);
  nntrainer::Tensor C_fp32 = A.dot(B, transA, transB);

  float mseError =
    mse<_FP16>(C.getData<_FP16>(), C_fp32.getData<_FP16>(), C.size());

  double cosSim = cosine_similarity<_FP16>(C.getData<_FP16>(),
                                           C_fp32.getData<_FP16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, addition_i_fp16) {

  int batch = 12;
  int channel = 1;
  int height = 26;
  int width = 26;

  int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch_b, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor C_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor D_fp16(batch_b, channel, height, width, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp16, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);
  GEN_TEST_INPUT(C_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(D_fp16, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  A_fp16.add_i(B_fp16);
  add_i_cl(C_fp16, D_fp16);

  float mseError =
    mse<_FP16>(A_fp16.getData<_FP16>(), C_fp16.getData<_FP16>(), A_fp16.size());

  double cosSim = cosine_similarity<_FP16>(
    A_fp16.getData<_FP16>(), C_fp16.getData<_FP16>(), A_fp16.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

#endif

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

static void run_q_6_K_test(const uint32_t M, const uint32_t K,
                           const uint32_t N) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  auto debug_print_beg_end = [M, K, N](const float *const data,
                                       const uint32_t count = 5) {
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

  static constexpr uint32_t run_count = 100;

  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  std::vector<float> weight = generate_random_vector<float, false>(N * K);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> cpu_q6_dst(M * N, 0.0f);

  const auto data_size = 210 * N * (K / 256);
  std::vector<char> q6_weight = std::vector<char>(data_size);

  // Generate result from SGEMM
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  // Initialize data
  void *gpu_q6_dst =
    blas_cc->context_inst_.createSVMRegion(M * N * sizeof(float));

  void *q6_weight_ptr = blas_cc->context_inst_.createSVMRegion(data_size);

  blas_cc->command_queue_inst_.enqueueSVMMap(q6_weight_ptr, data_size, false);

  float *weights_f32_ptr = weight.data();

  float *activations_f32_ptr =
    (float *)blas_cc->context_inst_.createSVMRegion(M * K * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);

  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  /// Quantize weight data
  nntrainer::quantize_q6_K(weights_f32_ptr, q6_weight_ptr, N, K, nullptr);

  // CPU Q6_K GEMV
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q6_K(M, N, K, activations_f32_ptr, K, q6_weight_ptr, N,
                         cpu_q6_dst.data(), N);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  // GPU Q6_K GEMV
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::sgemv_q6_k_cl(q6_weight_ptr, activations_f32_ptr,
                             (float *)gpu_q6_dst, K, N);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  // Compute raports
  {
    uint32_t first_zero_index = UINT32_MAX;
    int zeros = 0;
    int nans = 0;

    for (uint32_t i = 0; i < M * N; ++i) {
      if (((float *)gpu_q6_dst)[i] == 0) {
        zeros++;
        if (first_zero_index == UINT32_MAX) {
          first_zero_index = i;
        }
      }

      if (std::isnan(((float *)gpu_q6_dst)[i])) {
        nans++;
      }
    }

    // const auto mean_squared_error_dst_gpu =
    //   compute_mse(M, N, ref_dst, gpu_q6_dst, false);
    // const auto mean_squared_error_dst =
    //   compute_mse(M, N, ref_dst, cpu_q6_dst, false);

    const auto data_size_mb = data_size / (1024 * 1024.0f);

    std::cout << "Q6_K GEMV : " << M << " x " << K << " x " << N << std::endl;
    std::cout << " - q6_K data size : " << data_size_mb << " [MB]" << std::endl;
    std::cout << " - time : CPU = " << dt.count() / (run_count * 1.0f) << " ms"
              << std::endl;
    std::cout << " - time : GPU = " << gpu_dt.count() / (run_count * 1.0f)
              << " ms" << std::endl;
    std::cout << " - sample : CPU = ";
    debug_print_beg_end(cpu_q6_dst.data());
    std::cout << std::endl;
    std::cout << " - sample : GPU = ";
    debug_print_beg_end((float *)gpu_q6_dst);
    std::cout << std::endl;
    std::cout << " - zeros : " << zeros << " / " << M * N << " [ "
              << zeros * 100.0f / float(M * N) << " %] - first at [ "
              << first_zero_index << " ]" << std::endl;
    std::cout << " - nans : " << nans << " / " << M * N << " [ "
              << nans * 100.0f / float(M * N) << " %]" << std::endl;
    // std::cout << " - MSE : CPU = " << mean_squared_error_dst << std::endl;
    // std::cout << " - MSE : GPU = " << mean_squared_error_dst_gpu <<
    // std::endl;
  }
}

#define DECLARE_q_6_K_test_M_K_N(M, K, N)                                      \
  TEST(nntrainer_cpu_backend_standalone, q_6_K_test_##M##_##K##_##N) {         \
    run_q_6_K_test(M, K, N);                                                   \
  }

DECLARE_q_6_K_test_M_K_N(1, 3072, 105900);

#endif // ENABLE_GGML

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
