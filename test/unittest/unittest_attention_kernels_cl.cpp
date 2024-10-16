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
#include <attention_kernel_interface.h>
#include <cl_context.h>
#include <layer_context.h>
#include <tensor.h>

#include "testing_rotary_emb.cpp"

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

static void setUpGpuContext() {
  auto &ac = nntrainer::ClContext::Global();
  ac.initAttentionClKernels();
}

TEST(attention_kernels, rotary_emb_kernel_FP32) {
  setUpGpuContext();

  int batch = 1;
  int channel = 1;
  int height = 4;
  int width = 4;

  unsigned int dim = 2;
  unsigned int from = 4;
  unsigned int max_timestep = 4;

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

  B_fp32.copy(A_fp32);

  apply_rotary_emb_cl(A_fp32, dim, from, max_timestep);
  apply_rotary_emb_tensor(B_fp32, dim, from, max_timestep);

  float mseErrorNeon_fp32 =
    mse<float>(A_fp32.getData<float>(), B_fp32.getData<float>(), A_fp32.size());

  double cosSimNeon_fp32 = cosine_similarity<float>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), A_fp32.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon_fp32, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon_fp32, 0.99, 1);
}

TEST(attention_kernels, rotary_emb_kernel_FP32_case2) {
  setUpGpuContext();

  int batch = 4;
  int channel = 4;
  int height = 8;
  int width = 8;

  unsigned int dim = 2;
  unsigned int from = 2;
  unsigned int max_timestep = 4;

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

  B_fp32.copy(A_fp32);

  apply_rotary_emb_cl(A_fp32, dim, from, max_timestep);
  apply_rotary_emb_tensor(B_fp32, dim, from, max_timestep);

  float mseErrorNeon_fp32 =
    mse<float>(A_fp32.getData<float>(), B_fp32.getData<float>(), A_fp32.size());

  double cosSimNeon_fp32 = cosine_similarity<float>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), A_fp32.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon_fp32, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon_fp32, 0.99, 1);
}

TEST(attention_kernels, rotary_emb_kernel_FP16) {
  setUpGpuContext();

  int batch = 1;
  int channel = 1;
  int height = 4;
  int width = 4;

  unsigned int dim = 2;
  unsigned int from = 4;
  unsigned int max_timestep = 4;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height, width, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, i * (batch * height * channel) * alpha +
                           j * (batch * height) * alpha + k * (width)*alpha +
                           l + 1);

  B_fp16.copy(A_fp16);

  apply_rotary_emb_cl(A_fp16, dim, from, max_timestep);
  apply_rotary_emb_tensor(B_fp16, dim, from, max_timestep);

  float mseErrorNeon_fp16 = mse<__fp16>(
    A_fp16.getData<__fp16>(), B_fp16.getData<__fp16>(), A_fp16.size());

  double cosSimNeon_fp16 = cosine_similarity<__fp16>(
    A_fp16.getData<__fp16>(), B_fp16.getData<__fp16>(), A_fp16.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon_fp16, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon_fp16, 0.99, 1);
}

TEST(attention_kernels, rotary_emb_kernel_FP16_case2) {
  setUpGpuContext();

  int batch = 4;
  int channel = 4;
  int height = 8;
  int width = 8;

  unsigned int dim = 4;
  unsigned int from = 4;
  unsigned int max_timestep = 8;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height, width, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, i * (batch * height * channel) * alpha +
                           j * (batch * height) * alpha + k * (width)*alpha +
                           l + 1);

  B_fp16.copy(A_fp16);

  apply_rotary_emb_cl(A_fp16, dim, from, max_timestep);
  apply_rotary_emb_tensor(B_fp16, dim, from, max_timestep);

  float mseErrorNeon_fp16 = mse<__fp16>(
    A_fp16.getData<__fp16>(), B_fp16.getData<__fp16>(), A_fp16.size());

  double cosSimNeon_fp16 = cosine_similarity<__fp16>(
    A_fp16.getData<__fp16>(), B_fp16.getData<__fp16>(), A_fp16.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon_fp16, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon_fp16, 0.99, 1);
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
