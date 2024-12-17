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
#include <rmsnorm_layer_cl.h>
#include <tensor.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

static void setUpGpuContext() { auto &ac = nntrainer::ClContext::Global(); }

// else if m == 1, and either trans true
TEST(blas_kernels, fused_M_1_1) {
  // //if M == 1 and trans, Batch: 1, Channel: 1, Height: 1, Width: 2048
  setUpGpuContext();
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 2048;
  int width_b = 768;

  int batch_res = 1;
  int channel_res = 1;
  int height_res = 1;
  int width_res = height_b;

  bool transA = false;
  bool transB = true;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);
  GEN_TEST_BIAS(bias_fp32, 1);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  add_i_cl(C, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(C, outRMS_fp32, gamma_fp32, epsilon);

  bool disable_bias_value = true;
  nntrainer::Tensor res_fp32 =
    fusedProcess(A_fp32, B_fp32, bias_fp32, disable_bias_value, gamma_fp32,
                 epsilon, transA, transB);

  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  // printf("Size of res_fp32: %zu, Size of outRMS_fp32: %zu\n",
  // res_fp32.size(), outRMS_fp32.size()); std::cout << "\res_fp32 and
  // outRMS_fp32 before rotary embedding:" << std::endl; for (unsigned int i =
  // 0; i < res_fp32.size(); ++i) {
  //   std::cout << "Element " << i << " -> " << *(res_fp32.getData<float>() +
  //   i)
  //   <<"\t"<<*(outRMS_fp32.getData<float>() + i)<< std::endl;
  // }

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-6);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_1) {
  // if M == 1 and trans, Batch: 1, Channel: 1, Height: 1, Width: 2048
  // setUpGpuContext();
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
  // printf("if M == 1 and trans, Batch: %zu, Channel: %zu, Height: %zu, Width:
  // %zu\n", C_fp32.batch(), C_fp32.channel(), C_fp32.height(), C_fp32.width());

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

// else if m == 1, and trans false
TEST(blas_kernels, fused_M_1_2) {
  // //if M == 1 and trans false, Batch: 1, Channel: 1, Height: 1, Width: 2048
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 768;
  int width_b = 2048;

  int batch_res = 1;
  int channel_res = 1;
  int height_res = 1;
  int width_res = width_b;

  bool transA = false;
  bool transB = false;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  GEN_TEST_INPUT_RES(bias_fp32,
                     ((i * (batch_res * height_res * channel_res) +
                       j * (batch_res * height_res) + k * (width_res) + l + 1) %
                      MOD) *
                       alpha);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  add_i_cl(C, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(C, outRMS_fp32, gamma_fp32, epsilon);

  bool disable_bias_value = true;
  nntrainer::Tensor res_fp32 =
    fusedProcess(A_fp32, B_fp32, bias_fp32, disable_bias_value, gamma_fp32,
                 epsilon, transA, transB);

  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-6);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2) {
  // if M == 1 and !trans, Batch: 1, Channel: 1, Height: 1, Width: 2048
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

// else if n == 1, and trans true
TEST(blas_kernels, fused_N_1_1) {
  // if N == 1 and trans, Batch: 1, Channel: 1, Height: 2048, Width: 1
  // setUpGpuContext();
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 2048;

  int height_b = 768;
  int width_b = 1;

  int batch_res = 1;
  int channel_res = 1;
  int height_res = width;
  int width_res = 1;

  bool transA = true;
  bool transB = false;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  GEN_TEST_BIAS(bias_fp32, 1);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  bool disable_bias_value = true;
  nntrainer::Tensor res_fp32 =
    fusedProcess(A_fp32, B_fp32, bias_fp32, disable_bias_value, gamma_fp32,
                 epsilon, transA, transB);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  add_i_cl(C, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(C, outRMS_fp32, gamma_fp32, epsilon);

  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  // printf("Size of res_fp32: %zu, Size of outRMS_fp32: %zu\n",
  // res_fp32.size(), outRMS_fp32.size()); std::cout << "\res_fp32 and
  // outRMS_fp32 after everything" << std::endl; for (unsigned int i = 0; i <
  // res_fp32.size(); ++i) {
  //   std::cout << "Element " << i << " -> " << *(res_fp32.getData<float>() +
  //   i)
  //   <<"\t"<<*(outRMS_fp32.getData<float>() + i)<< std::endl;
  // }

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-6);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
}

// if N == 1 and trans, Batch: 1, Channel: 1, Height: 2048, Width: 1
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

// // else if n == 1, and trans false
TEST(blas_kernels, fused_N_1_2) {
  // if N == 1 and !trans, Batch: 1, Channel: 1, Height: 768, Width: 1
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 2048;

  int height_b = 2048;
  int width_b = 1;

  int batch_res = 1;
  int channel_res = 1;
  int height_res = height;
  int width_res = 1;

  bool transA = false;
  bool transB = false;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  GEN_TEST_INPUT_RES(bias_fp32,
                     ((i * (batch_res * height_res * channel_res) +
                       j * (batch_res * height_res) + k * (width_res) + l + 1) %
                      MOD) *
                       alpha);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  bool disable_bias_value = true;
  nntrainer::Tensor res_fp32 =
    fusedProcess(A_fp32, B_fp32, bias_fp32, disable_bias_value, gamma_fp32,
                 epsilon, transA, transB);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  add_i_cl(C, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(C, outRMS_fp32, gamma_fp32, epsilon);

  // nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-6);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
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
  // if N == 1 and !trans, Batch: 1, Channel: 1, Height: 768, Width: 1
  // printf("if N == 1 and !trans, Batch: %zu, Channel: %zu, Height: %zu, Width:
  // %zu\n", C_fp32.batch(), C_fp32.channel(), C_fp32.height(), C_fp32.width());

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

TEST(nntrainer_Tensor, fused_gemm_50_768_1024_noTrans) {
  // if N == 1 and !transA & !transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 1024
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 768;
  int width_b = 1024;

  // int batch = 1;
  // int channel = 1;
  // int height = 2;
  // int width = 3;

  // int height_b = 3;
  // int width_b = 4;

  int batch_res = batch;
  int channel_res = channel;
  int height_res = height;
  int width_res = width_b;

  bool transA = false;
  bool transB = false;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  GEN_TEST_BIAS(bias_fp32, 1);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  // auto t1 = std::chrono::high_resolution_clock::now();
  // nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  // // for(int i=0;i<999;i++)
  // //   C = dotCl(A_fp32, B_fp32, transA, transB);
  // auto t2 = std::chrono::high_resolution_clock::now();
  // auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(t2 -
  // t1); std::cout<<"Timing SGEMV dotCl: "<<ms_int.count()/1000<<std::endl;

  bool disable_bias_value = true;
  nntrainer::Tensor res_fp32 =
    fusedProcess(A_fp32, B_fp32, bias_fp32, disable_bias_value, gamma_fp32,
                 epsilon, transA, transB);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  add_i_cl(C, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(C, outRMS_fp32, gamma_fp32, epsilon);

  // nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-9);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_1024_noTrans) {
  // if N == 1 and !transA & !transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 1024
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
  // printf("if N == 1 and !transA & !transB, Batch: %zu, Channel: %zu, Height:
  // %zu, Width: %zu\n", C.batch(), C.channel(), C.height(), C.width());

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, fused_gemm_50_768_2048_transB) {
  // if N == 1 and !transA & transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 2048
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 2048;
  int width_b = 768;

  int batch_res = batch;
  int channel_res = channel;
  int height_res = height;
  int width_res = height_b;

  bool transA = false;
  bool transB = true;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  GEN_TEST_BIAS(bias_fp32, 1);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  bool disable_bias_value = true;

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  add_i_cl(C, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(C, outRMS_fp32, gamma_fp32, epsilon);

  nntrainer::Tensor res_fp32 =
    fusedProcess(A_fp32, B_fp32, bias_fp32, disable_bias_value, gamma_fp32,
                 epsilon, transA, transB);
  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-9);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_2048_transB) {
  // if N == 1 and !transA & transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 2048
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 2048;
  int width_b = 768;

  int batch_res = batch;
  int channel_res = channel;
  int height_res = height;
  int width_res = height_b;

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
  // printf("if N == 1 and !transA & transB, Batch: %zu, Channel: %zu, Height:
  // %zu, Width: %zu\n", C.batch(), C.channel(), C.height(), C.width());

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, fused_ResultThere) {
  // if N == 1 and transA & !transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 1024
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 3;
  int width = 2;

  int height_b = 3;
  int width_b = 4;

  int batch_res = batch;
  int channel_res = channel;
  int height_res = width;
  int width_res = width_b;

  bool transA = true;
  bool transB = false;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor res_fp32(batch_res, channel_res, height_res, width_res,
                             t_type_nchw_fp32);
  nntrainer::Tensor out_fp32(batch_res, channel_res, height_res, width_res,
                             t_type_nchw_fp32);
  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  GEN_TEST_INPUT_RES(res_fp32, 2);

  GEN_TEST_INPUT_RES(out_fp32, 2);

  GEN_TEST_BIAS(bias_fp32, 1);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  bool disable_bias_value = true;
  fusedProcess(A_fp32, B_fp32, res_fp32, bias_fp32, disable_bias_value,
               gamma_fp32, epsilon, transA, transB);

  dotCl(A_fp32, B_fp32, out_fp32, transA, transB);
  add_i_cl(out_fp32, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(out_fp32, outRMS_fp32, gamma_fp32, epsilon);

  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-9);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
}
TEST(nntrainer_Tensor, dot_resultThere) {
  // if N == 1 and transA & !transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 1024
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 50;

  int height_b = 768;
  int width_b = 1024;

  int batch_res = batch;
  int channel_res = channel;
  int height_res = width;
  int width_res = width_b;

  bool transA = true;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor C(batch_res, channel_res, height_res, width_res,
                      t_type_nchw_fp32);
  nntrainer::Tensor out_fp32(batch_res, channel_res, height_res, width_res,
                             t_type_nchw_fp32);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  dotCl(A_fp32, B_fp32, C, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, out_fp32, transA, transB);
  // printf("if N == 1 and transA & !transB, Batch: %zu, Channel: %zu, Height:
  // %zu, Width: %zu\n", C.batch(), C.channel(), C.height(), C.width());

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, fused_gemm_50_768_1024_transA) {
  // if N == 1 and transA & !transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 1024
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 50;

  int height_b = 768;
  int width_b = 1024;

  int batch_res = batch;
  int channel_res = channel;
  int height_res = width;
  int width_res = width_b;

  bool transA = true;
  bool transB = false;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  GEN_TEST_BIAS(bias_fp32, 1);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  bool disable_bias_value = true;
  nntrainer::Tensor res_fp32 =
    fusedProcess(A_fp32, B_fp32, bias_fp32, disable_bias_value, gamma_fp32,
                 epsilon, transA, transB);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  add_i_cl(C, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(C, outRMS_fp32, gamma_fp32, epsilon);

  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-9);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
}
TEST(nntrainer_Tensor, dot_gemm_50_768_1024_transA) {
  // if N == 1 and transA & !transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 1024
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 50;

  int height_b = 768;
  int width_b = 1024;

  // int batch_res = batch;
  // int channel_res = channel;
  // int height_res = width;
  // int width_res = width_b;

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
  // printf("if N == 1 and transA & !transB, Batch: %zu, Channel: %zu, Height:
  // %zu, Width: %zu\n", C.batch(), C.channel(), C.height(), C.width());

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, fused_gemm_50_768_2048_transAB) {
  // if N == 1 and transA & transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 2048
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 50;

  int height_b = 2048;
  int width_b = 768;

  int batch_res = batch;
  int channel_res = channel;
  int height_res = width;
  int width_res = height_b;

  bool transA = true;
  bool transB = true;

  const float epsilon = 1e-3 * width;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);
  nntrainer::Tensor bias_fp32(batch_res, channel_res, height_res, width_res,
                              t_type_nchw_fp32);
  nntrainer::Tensor outRMS_fp32(batch_res, channel_res, height_res, width_res,
                                t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width_res, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  GEN_TEST_BIAS(bias_fp32, 1);
  GEN_TEST_INPUT_GAMMA(gamma_fp32, 1);

  nntrainer::Tensor C = dotCl(A_fp32, B_fp32, transA, transB);
  add_i_cl(C, bias_fp32);
  RMSNormLayerCl obj;
  obj.rmsnormProcess(C, outRMS_fp32, gamma_fp32, epsilon);

  bool disable_bias_value = true;
  nntrainer::Tensor res_fp32 =
    fusedProcess(A_fp32, B_fp32, bias_fp32, disable_bias_value, gamma_fp32,
                 epsilon, transA, transB);

  float mseErrorNeon = mse<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    res_fp32.getData<float>(), outRMS_fp32.getData<float>(), res_fp32.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-9);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.999999, 1);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_2048_transAB) {
  // if N == 1 and transA & transB, Batch: 1, Channel: 1, Height: 50, Width:
  // 2048
  /// @note GEMM : A X B = C

  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 50;

  int height_b = 2048;
  int width_b = 768;

  int batch_res = batch;
  int channel_res = channel;
  int height_res = width;
  int width_res = height_b;

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
  // printf("if N == 1 and transA & transB, Batch: %zu, Channel: %zu, Height:
  // %zu, Width: %zu\n", C.batch(), C.channel(), C.height(), C.width());

  float mseErrorNeon =
    mse<float>(C.getData<float>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<float>(
    C.getData<float>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
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

  float mseErrorNeon =
    mse<float>(A_fp32.getData<float>(), C_fp32.getData<float>(), A_fp32.size());

  double cosSimNeon = cosine_similarity<float>(
    A_fp32.getData<float>(), C_fp32.getData<float>(), A_fp32.size());

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
