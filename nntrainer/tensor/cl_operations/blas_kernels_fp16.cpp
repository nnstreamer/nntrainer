// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels_fp16.cpp
 * @date	29 May 2024
 * @brief	Common blas OpenCL fp16 kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "blas_kernels_templates.h"

namespace nntrainer {

void sgemv_cl(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_sgemv_fp16_ptr;
  if (TransA) {
    kernel_sgemv_fp16_ptr =
      blas_cc->registerClKernel(getHgemvClKernel(), "sgemv_cl_fp16");
  } else {
    kernel_sgemv_fp16_ptr = blas_cc->registerClKernel(getHgemvClNoTransKernel(),
                                                      "sgemv_cl_noTrans_fp16");
  }

  if (!kernel_sgemv_fp16_ptr) {
    return;
  }

  sgemv_cl_internal<_FP16>(kernel_sgemv_fp16_ptr, matAdata, vecXdata, vecYdata,
                           dim1, dim2, lda);
}

_FP16 dot_cl(const _FP16 *vecAdata, const _FP16 *vecXdata, unsigned int dim1) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_dot_fp16_ptr =
    blas_cc->registerClKernel(getDotClKernelFP16(), "dot_cl_fp16");

  if (!kernel_dot_fp16_ptr) {
    return {};
  }

  return dot_cl_internal<_FP16>(kernel_dot_fp16_ptr, vecAdata, vecXdata, dim1);
}

void sgemm_cl(bool TransA, bool TransB, const _FP16 *A, const _FP16 *B,
              _FP16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc) {
  std::string kernel_func_;
  std::string sgemm_cl_kernel_fp16_;
  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans_fp16";
    sgemm_cl_kernel_fp16_ = getHgemmClNoTransKernel();
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA_fp16";
    sgemm_cl_kernel_fp16_ = getHgemmClTransAKernel();
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB_fp16";
    sgemm_cl_kernel_fp16_ = getHgemmClTransBKernel();
  } else {
    kernel_func_ = "sgemm_cl_transAB_fp16";
    sgemm_cl_kernel_fp16_ = getHgemmClTransABKernel();
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_sgemm_fp16_ptr =
    blas_cc->registerClKernel(sgemm_cl_kernel_fp16_, kernel_func_);
  if (!kernel_sgemm_fp16_ptr) {
    return;
  }

  sgemm_cl_internal<_FP16>(kernel_sgemm_fp16_ptr, TransA, TransB, A, B, C, M, N,
                           K, lda, ldb, ldc);
}

void addition_cl(const _FP16 *input, _FP16 *res, unsigned int size_input,
                 unsigned int size_res) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_addition_fp16_ptr =
    blas_cc->registerClKernel(getAdditionClKernelFP16(), "addition_cl_fp16");
  if (!kernel_addition_fp16_ptr) {
    return;
  }

  addition_cl_internal<_FP16>(kernel_addition_fp16_ptr, input, res, size_input,
                              size_res);
}

void sscal_cl(_FP16 *X, const unsigned int N, const float alpha) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_sscal_fp16_ptr =
    blas_cc->registerClKernel(getHscalClKernel(), "sscal_cl_fp16");

  if (!kernel_sscal_fp16_ptr) {
    return;
  }

  sscal_cl_internal<_FP16>(kernel_sscal_fp16_ptr, X, N, alpha);
}

void transpose_cl_axis(const _FP16 *in, _FP16 *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_transpose_fp_16_ptr;
  switch (axis) {
  case 0:
    kernel_transpose_fp_16_ptr = blas_cc->registerClKernel(
      getTransposeClAxis0KernelFP16(), "transpose_cl_fp16_axis0");
    break;
  case 1:
    kernel_transpose_fp_16_ptr = blas_cc->registerClKernel(
      getTransposeClAxis1KernelFP16(), "transpose_cl_fp16_axis1");
    break;
  case 2:
    kernel_transpose_fp_16_ptr = blas_cc->registerClKernel(
      getTransposeClAxis2KernelFP16(), "transpose_cl_fp16_axis2");
    break;
  default:
    throw std::invalid_argument("failed to register CL kernel");
    break;
  }

  if (!kernel_transpose_fp_16_ptr) {
    return;
  }

  transpose_cl_axis_internal<_FP16>(kernel_transpose_fp_16_ptr, in, res,
                                    input_batch_size, input_channels,
                                    input_height, input_width, axis);
}
} // namespace nntrainer
