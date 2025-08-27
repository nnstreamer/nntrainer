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

#include <cstring>
#include <gtest/gtest.h>
#include <utility>

#include "fallback_internal.h"
#include "nntrainer_test_util.h"
#include "swiglu_cl.h"
#include "tensor_dim.h"
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

// -----
// Functions
// -----

static std::pair<float, float>
dotCL_sgemv_test_func(const int batch, const int channel, const int height,
                      const int width, const int height_b, const int width_b,
                      const bool transA, const bool transB, const float alpha,
                      const int MOD) {
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

  return {mseError, static_cast<float>(cosSim)};
}

#ifdef ENABLE_FP16
static std::pair<float, float> dotCL_sgemv_fp16_test_func(
  const int batch, const int channel, const int height, const int width,
  const int height_b, const int width_b, const bool transA, const bool transB,
  const float alpha, const int MOD) {
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

  return {mseError, static_cast<float>(cosSim)};
}

static std::pair<float, float>
dot_gemm_fp16(const int batch, const int channel, const int height,
              const int width, const int height_b, const int width_b,
              const bool transA, const bool transB, const float alpha,
              const int MOD) {
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

  return {mseError, static_cast<float>(cosSim)};
}
#endif

void *allocateSVM(size_t size_bytes) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  void *ptr = blas_cc->context_inst_.createSVMRegion(size_bytes);

  if (ptr == nullptr) {
    throw std::runtime_error(
      "Failed to allocated SVM for the OpenCL BLAS unit test.");
  }

  return ptr;
}

void freeSVM(void *ptr) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  blas_cc->context_inst_.releaseSVMRegion(ptr);
  ptr = nullptr;
}

auto debug_print_data = [](const float *const data, const unsigned int len,
                           const uint32_t count = 5) {
  std::cout << "[";
  for (unsigned int i = 0; i < count; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << "][";
  for (unsigned int i = len - count; i < len; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << "]";
};

// -----
// Tests
// -----

TEST(blas_kernels, dotCL_sgemv_M_1_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_1) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 768;
  const int width_b = 1;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_2) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 2048;
  const int width_b = 1;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_n) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = true;
  const bool transB = false;

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
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 1;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

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
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 1;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-5 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_noTrans) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transB) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_transA) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transAB) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = true;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_test_func(batch, channel, height, width, height_b, width_b,
                          transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE(cosSim, 0.99, 1);
}

TEST(blas_kernels, addition_i) {
  const int batch = 1;
  const int channel = 1;
  const int height = 64;
  const int width = 3072;

  const int batch_b = 1;

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
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 768;

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
  const int batch = 1;
  const int channel = 1;
  const int height = 32;
  const int width = 32;

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

TEST(blas_kernels, rmsnorm_fp32_67_3072) {
  const int batch = 1;
  const int channel = 1;
  const int height = 67;
  const int width = 3072;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor in_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor gamma_fp32(1, 1, 1, width, t_type_nchw_fp32);
  nntrainer::Tensor out_cl_fp32(batch, channel, height, width,
                                t_type_nchw_fp32);
  nntrainer::Tensor out_ref_fp32(batch, channel, height, width,
                                 t_type_nchw_fp32);

  /// Initialize CPU input data
  GEN_TEST_INPUT(in_fp32, ((i * (batch * height * channel) +
                            j * (batch * height) + k * (width) + l + 1) %
                           MOD) *
                            alpha);
  for (int l = 0; l < width; ++l) {
    float val = ((l + 1) % MOD) * alpha;
    gamma_fp32.setValue(0, 0, 0, l, val);
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  /// Initialize GPU input/ouput data
  void *in_fp32_svm = allocateSVM(in_fp32.size() * sizeof(float));
  void *gamma_fp32_svm = allocateSVM(gamma_fp32.size() * sizeof(float));
  void *out_fp32_svm = allocateSVM(out_cl_fp32.size() * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(
    in_fp32_svm, in_fp32.size() * sizeof(float), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(
    gamma_fp32_svm, gamma_fp32.size() * sizeof(float), false);

  std::memcpy(in_fp32_svm, in_fp32.getData<float>(),
              in_fp32.size() * sizeof(float));
  std::memcpy(gamma_fp32_svm, gamma_fp32.getData<float>(),
              gamma_fp32.size() * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMUnmap(in_fp32_svm);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(gamma_fp32_svm);

  static constexpr uint32_t run_count = 500;
  static constexpr float kEpsilon = 0.001f;

  auto t1_cl = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    rmsnorm_cl((float *)in_fp32_svm, (float *)gamma_fp32_svm,
               (float *)out_fp32_svm, kEpsilon,
               in_fp32.batch() * in_fp32.channel() * in_fp32.height(),
               in_fp32.width(), true);
  }
  auto t2_cl = std::chrono::high_resolution_clock::now();

  auto t1_ref = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    std::function<float(float)> f = [](float x) { return 1 / std::sqrt(x); };
    auto t = in_fp32.multiply(in_fp32).average(3).add(kEpsilon);
    t.apply_i(f);
    in_fp32.multiply(t, out_ref_fp32);
    out_ref_fp32.multiply_i(gamma_fp32);
  }
  auto t2_ref = std::chrono::high_resolution_clock::now();

  auto dt_cl =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_cl - t1_cl);
  auto dt_ref =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_ref - t1_ref);

  std::cout << "RMSNorm time : GPU = " << dt_cl.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  std::cout << "RMSNorm time : CPU = " << dt_ref.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  blas_cc->command_queue_inst_.enqueueSVMMap(
    out_fp32_svm, out_cl_fp32.size() * sizeof(float), false);

  float mseError = mse<float>((float *)out_fp32_svm,
                              out_ref_fp32.getData<float>(), height * width);

  double cosSim = cosine_similarity<float>(
    (float *)out_fp32_svm, out_ref_fp32.getData<float>(), height * width);

  const float epsilon = 1e-3 * width;

  freeSVM(out_fp32_svm);
  freeSVM(in_fp32_svm);
  freeSVM(gamma_fp32_svm);

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, swiglu_layer_fp32_67_3072) {
  const int batch = 1;
  const int channel = 1;
  const int height = 67;
  const int width = 3072;

  const int dim = width * height;

  const int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  /// Initialize CPU input/output
  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch_b, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor out_ref_fp32(batch, channel, height, width,
                                 t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp32, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  /// Initialize GPU input/output
  void *gpu_in1 = allocateSVM(dim * sizeof(float));
  void *gpu_in2 = allocateSVM(dim * sizeof(float));
  void *gpu_dst = allocateSVM(dim * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(gpu_in1, dim * sizeof(float),
                                             false);
  blas_cc->command_queue_inst_.enqueueSVMMap(gpu_in2, dim * sizeof(float),
                                             false);

  std::memcpy(gpu_in1, A_fp32.getData(), dim * sizeof(float));
  std::memcpy(gpu_in2, B_fp32.getData(), dim * sizeof(float));

  static constexpr uint32_t run_count = 500;

  SwiGLULayerCl layer;

  auto t1_cl = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    layer.swiglu_cl((float *)gpu_in1, (float *)gpu_in2, (float *)gpu_dst, width,
                    height, true);
  }
  auto t2_cl = std::chrono::high_resolution_clock::now();

  auto t1_ref = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::swiglu(height * width, out_ref_fp32.getData(), A_fp32.getData(),
                      B_fp32.getData());
  }
  auto t2_ref = std::chrono::high_resolution_clock::now();

  auto dt_cl =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_cl - t1_cl);
  auto dt_ref =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_ref - t1_ref);

  std::cout << "Swiglu time : GPU = " << dt_cl.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  std::cout << "Swiglu time : CPU = " << dt_ref.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  float mseError =
    mse<float>((float *)gpu_dst, out_ref_fp32.getData<float>(), height * width);

  double cosSim = cosine_similarity<float>(
    (float *)gpu_dst, out_ref_fp32.getData<float>(), height * width);

  const float epsilon = 1e-3 * width;

  freeSVM(gpu_in1);
  freeSVM(gpu_in2);
  freeSVM(gpu_dst);

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

#ifdef ENABLE_FP16

TEST(blas_kernels, dotCL_sgemv_M_1_1_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_M_1_2_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_1_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 768;
  const int width_b = 1;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_N_1_2_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 2048;

  const int height_b = 2048;
  const int width_b = 1;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dotCL_sgemv_fp16_test_func(batch, channel, height, width, height_b, width_b,
                               transA, transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dotCL_sgemv_n_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 1;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 2048;

  const bool transA = true;
  const bool transB = false;

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
  const int batch = 1;
  const int channel = 1;
  const int height = 2;
  const int width = 11;

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
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = false;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transB_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 50;
  const int width = 768;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = false;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_1024_transA_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 768;
  const int width_b = 1024;

  const bool transA = true;
  const bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, dot_gemm_50_768_2048_transAB_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 768;
  const int width = 50;

  const int height_b = 2048;
  const int width_b = 768;

  const bool transA = true;
  const bool transB = true;

  const float alpha = 1e-1;
  const int MOD = 10;

  const auto [mseError, cosSim] =
    dot_gemm_fp16(batch, channel, height, width, height_b, width_b, transA,
                  transB, alpha, MOD);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}

TEST(blas_kernels, addition_i_fp16) {
  const int batch = 12;
  const int channel = 1;
  const int height = 26;
  const int width = 26;

  const int batch_b = 1;

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

  static constexpr uint32_t run_count = 10;

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
  void *gpu_q6_dst = allocateSVM(M * N * sizeof(float));

  void *q6_weight_ptr = allocateSVM(data_size);

  blas_cc->command_queue_inst_.enqueueSVMMap(q6_weight_ptr, data_size, false);

  float *weights_f32_ptr = weight.data();

  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));

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

    std::cout << "Q6_K GEMV : " << M << " x " << K << " x " << N << std::endl;
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

    freeSVM(gpu_q6_dst);
    freeSVM(q6_weight_ptr);
    freeSVM(activations_f32_ptr);
  }
}

#define DECLARE_q_6_K_test_M_K_N(M, K, N)                                      \
  TEST(nntrainer_cpu_backend_standalone, q_6_K_test_##M##_##K##_##N) {         \
    run_q_6_K_test(M, K, N);                                                   \
  }

DECLARE_q_6_K_test_M_K_N(1, 3072, 105900);

#if 1
static void run_q4_0_test(const uint32_t M, const uint32_t K,
                          const uint32_t N) {
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

  static constexpr uint32_t run_count = 200;

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
  int64_t num_blocks = (K * N) / block_size;
  size_t data_size = q4_0_tile_size * num_blocks;

  // Generate result from SGEMM
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  // Initialize data
  void *gpu_q4_dst = allocateSVM(M * N * sizeof(float));

  void *q4_weight_ptr = allocateSVM(data_size);
  void *q4_weight_repack_ptr = allocateSVM(data_size);

  blas_cc->command_queue_inst_.enqueueSVMMap(q4_weight_ptr, data_size, false);

  float *weights_f32_ptr = weight.data();

  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);

  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  /// Quantize weight data
  nntrainer::quantize_q4_0(weights_f32_ptr, q4_weight_ptr, N, K, nullptr);
  nntrainer::repack_q4_0(q4_weight_repack_ptr, q4_weight_ptr, data_size, N, K);

  // CPU Q4_0 GEMM
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0(M, N, K, activations_f32_ptr, K, q4_weight_repack_ptr,
                         N, cpu_q4_dst.data(), N);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  // GPU Q4_0 GEMM
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl(q4_weight_repack_ptr, activations_f32_ptr,
                            (float *)gpu_q4_dst, M, N, K);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  // Compute raports
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

    freeSVM(gpu_q4_dst);
    freeSVM(q4_weight_ptr);
    freeSVM(q4_weight_repack_ptr);
    freeSVM(activations_f32_ptr);
  }
}

#define DECLARE_q4_0_test_M_K_N(M, K, N)                                       \
  TEST(nntrainer_blas_kernel, q4_0_test_##M##_##K##_##N) {                     \
    run_q4_0_test(M, K, N);                                                    \
  }

DECLARE_q4_0_test_M_K_N(68, 3072, 256);
DECLARE_q4_0_test_M_K_N(68, 3072, 8192);
DECLARE_q4_0_test_M_K_N(68, 8192, 3072);
DECLARE_q4_0_test_M_K_N(68, 3072, 3072);

DECLARE_q4_0_test_M_K_N(28, 3072, 256);
DECLARE_q4_0_test_M_K_N(28, 3072, 8192);
DECLARE_q4_0_test_M_K_N(28, 8192, 3072);
DECLARE_q4_0_test_M_K_N(28, 3072, 3072);

#endif

TEST(nntrainer_blas_kernel, q4_0_async_test) {

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;

  const int M = 68;
  const int K = 3072;
  const int N0 = 3072, N1 = 256;

  // Initialize Activation
  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));
  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);
  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  // Initialize Weight
  struct block_q4_0 {
    uint16_t d[1];
    uint8_t qs[16];
  };
  int64_t block_size = 32;
  int64_t q4_0_tile_size = sizeof(block_q4_0);
  size_t data_size_n0 = q4_0_tile_size * (K * N0) / block_size;
  size_t data_size_n1 = q4_0_tile_size * (K * N1) / block_size;

  std::vector<float> weight0 = generate_random_vector<float, true>(N0 * K);
  std::vector<float> weight1 = generate_random_vector<float, true>(N1 * K);
  std::vector<float> weight2 = generate_random_vector<float, true>(N1 * K);

  // weight 0 (3072 x 3072)
  void *w0 = allocateSVM(data_size_n0);
  void *wq0 = allocateSVM(data_size_n0);
  blas_cc->command_queue_inst_.enqueueSVMMap(w0, data_size_n0, false);
  float *weights_f32_ptr = weight0.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w0, N0, K, nullptr);
  nntrainer::repack_q4_0(wq0, w0, data_size_n0, N0, K);

  // weight 1 (3072 x 256)
  void *w1 = allocateSVM(data_size_n1);
  void *wq1 = allocateSVM(data_size_n1);
  blas_cc->command_queue_inst_.enqueueSVMMap(w1, data_size_n1, false);
  weights_f32_ptr = weight1.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w1, N1, K, nullptr);
  nntrainer::repack_q4_0(wq1, w1, data_size_n1, N1, K);

  // weight 2 (3072 x 256)
  void *w2 = allocateSVM(data_size_n1);
  void *wq2 = allocateSVM(data_size_n1);
  blas_cc->command_queue_inst_.enqueueSVMMap(w2, data_size_n1, false);
  weights_f32_ptr = weight2.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w2, N1, K, nullptr);
  nntrainer::repack_q4_0(wq2, w2, data_size_n1, N1, K);

  // Initialize Output data
  float *out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *out1 = (float *)allocateSVM(M * N1 * sizeof(float));
  float *out2 = (float *)allocateSVM(M * N1 * sizeof(float));

  // In-order kernel execution
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl(wq0, activations_f32_ptr, (float *)out0, M, N0, K);
    nntrainer::gemm_q4_0_cl(wq1, activations_f32_ptr, (float *)out1, M, N1, K);
    nntrainer::gemm_q4_0_cl(wq2, activations_f32_ptr, (float *)out2, M, N1, K);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  float *async_out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *async_out1 = (float *)allocateSVM(M * N1 * sizeof(float));
  float *async_out2 = (float *)allocateSVM(M * N1 * sizeof(float));

  std::vector<void *> weight_vec = {wq0, wq1, wq2};
  std::vector<float *> out_vec = {async_out0, async_out1, async_out2};
  std::vector<unsigned int> n_vec = {N0, N1, N1};

  // Async
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_async_cl(weight_vec, activations_f32_ptr, out_vec, M,
                                  n_vec, K);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "Q4_0 GEMM : " << M << " x " << K << " x " << N1 << std::endl;
  std::cout << " - time : Orig = " << dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;
  std::cout << " - time : Async = " << gpu_dt.count() / (run_count * 1.0f)
            << " ms" << std::endl;

  // Free allocated SVM
  freeSVM(activations_f32_ptr);
  freeSVM(w0);
  freeSVM(wq0);
  freeSVM(w1);
  freeSVM(wq1);
  freeSVM(w2);
  freeSVM(wq2);
  freeSVM(out0);
  freeSVM(out1);
  freeSVM(out2);
  freeSVM(async_out0);
  freeSVM(async_out1);
  freeSVM(async_out2);
}

TEST(nntrainer_blas_kernel, q4_0_async_test2) {

  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  static constexpr uint32_t run_count = 200;

  const int M = 68;
  const int K = 3072;
  const int N0 = 8192;

  // Initialize Activation
  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  float *activations_f32_ptr = (float *)allocateSVM(M * K * sizeof(float));
  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);
  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  // Initialize Weight
  struct block_q4_0 {
    uint16_t d[1];
    uint8_t qs[16];
  };
  int64_t block_size = 32;
  int64_t q4_0_tile_size = sizeof(block_q4_0);
  size_t data_size_n0 = q4_0_tile_size * (K * N0) / block_size;

  std::vector<float> weight0 = generate_random_vector<float, true>(N0 * K);
  std::vector<float> weight1 = generate_random_vector<float, true>(N0 * K);

  // weight 0 (3072 x 8192)
  void *w0 = allocateSVM(data_size_n0);
  void *wq0 = allocateSVM(data_size_n0);
  blas_cc->command_queue_inst_.enqueueSVMMap(w0, data_size_n0, false);
  float *weights_f32_ptr = weight0.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w0, N0, K, nullptr);
  nntrainer::repack_q4_0(wq0, w0, data_size_n0, N0, K);

  // weight 1 (3072 x 8192)
  void *w1 = allocateSVM(data_size_n0);
  void *wq1 = allocateSVM(data_size_n0);
  blas_cc->command_queue_inst_.enqueueSVMMap(w1, data_size_n0, false);
  weights_f32_ptr = weight1.data();
  nntrainer::quantize_q4_0(weights_f32_ptr, w1, N0, K, nullptr);
  nntrainer::repack_q4_0(wq1, w1, data_size_n0, N0, K);

  // Initialize Output data
  float *out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *out1 = (float *)allocateSVM(M * N0 * sizeof(float));

  // CPU
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_cl(wq0, activations_f32_ptr, (float *)out0, M, N0, K);
    nntrainer::gemm_q4_0_cl(wq1, activations_f32_ptr, (float *)out1, M, N0, K);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  float *async_out0 = (float *)allocateSVM(M * N0 * sizeof(float));
  float *async_out1 = (float *)allocateSVM(M * N0 * sizeof(float));

  std::vector<void *> weight_vec = {wq0, wq1};
  std::vector<float *> out_vec = {async_out0, async_out1};
  std::vector<unsigned int> n_vec = {N0, N0};

  // Async
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_0_async_cl(weight_vec, activations_f32_ptr, out_vec, M,
                                  n_vec, K);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  std::cout << "Q4_0 GEMM : " << M << " x " << K << " x " << N0 << std::endl;
  std::cout << " - time : Orig = " << dt.count() / (run_count * 1.0f) << " ms"
            << std::endl;
  std::cout << " - time : Async = " << gpu_dt.count() / (run_count * 1.0f)
            << " ms" << std::endl;

  // Free allocated SVM
  freeSVM(activations_f32_ptr);
  freeSVM(w0);
  freeSVM(wq0);
  freeSVM(w1);
  freeSVM(wq1);
  freeSVM(out0);
  freeSVM(out1);
  freeSVM(async_out0);
  freeSVM(async_out1);
}

#endif // ENABLE_GGML

#ifdef ENABLE_FP16
TEST(blas_kernels, swiglu_layer_fp16) {
  const int batch = 1;
  const int channel = 1;
  const int height = 64;
  const int width = 3072;

  const int batch_b = 1;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch_b, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor out_cl_fp16(batch_b, channel, height, width,
                                t_type_nchw_fp16);
  nntrainer::Tensor out_ref_fp16(batch, channel, height, width,
                                 t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_C(B_fp16, ((i * (batch_b * height * channel) +
                             j * (batch_b * height) + k * (width) + l + 1) %
                            MOD) *
                             alpha);

  static constexpr uint32_t run_count = 500;

  SwiGLULayerCl layer;

  auto t1_cl = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    layer.swiglu_cl_fp16(A_fp16.getData<_FP16>(), B_fp16.getData<_FP16>(),
                         out_cl_fp16.getData<_FP16>(), width, height);
  }
  auto t2_cl = std::chrono::high_resolution_clock::now();

  auto t1_ref = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::swiglu(height * width, out_ref_fp16.getData<_FP16>(),
                      A_fp16.getData<_FP16>(), B_fp16.getData<_FP16>());
  }
  auto t2_ref = std::chrono::high_resolution_clock::now();

  auto dt_cl =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_cl - t1_cl);
  auto dt_ref =
    std::chrono::duration_cast<std::chrono::microseconds>(t2_ref - t1_ref);

  std::cout << "Swiglu time : GPU = " << dt_cl.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  std::cout << "Swiglu time : CPU = " << dt_ref.count() / (run_count * 1000.0f)
            << " ms" << std::endl;

  float mseError = mse<_FP16>(out_cl_fp16.getData<_FP16>(),
                              out_ref_fp16.getData<_FP16>(), height * width);

  double cosSim =
    cosine_similarity<_FP16>(out_cl_fp16.getData<_FP16>(),
                             out_ref_fp16.getData<_FP16>(), height * width);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseError, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSim, 0.99, 1);
}
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
