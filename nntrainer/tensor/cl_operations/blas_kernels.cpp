// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.cpp
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "blas_kernels_templates.h"

namespace nntrainer {

void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {
  ClContext::SharedPtrClKernel kernel_sgemv_ptr;

  if (TransA) {
    kernel_sgemv_ptr =
      blas_cc->registerClKernel(getSgemvClKernel(), "sgemv_cl");
  } else {
    kernel_sgemv_ptr =
      blas_cc->registerClKernel(getSgemvClNoTransKernel(), "sgemv_cl_noTrans");
  }

  if (!kernel_sgemv_ptr) {
    return;
  }

  sgemv_cl_internal<float>(kernel_sgemv_ptr, matAdata, vecXdata, vecYdata, dim1,
                           dim2, lda);
}

float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1) {
  ClContext::SharedPtrClKernel kernel_dot_ptr =
    blas_cc->registerClKernel(getDotClKernel(), "dot_cl");

  if (!kernel_dot_ptr) {
    return {};
  }

  return dot_cl_internal<float>(kernel_dot_ptr, vecAdata, vecXdata, dim1);
}

void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc) {
  std::string kernel_func_;
  std::string sgemm_cl_kernel_;

  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans";
    sgemm_cl_kernel_ = getSgemmClNoTransKernel();
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA";
    sgemm_cl_kernel_ = getSgemmClTransAKernel();
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB";
    sgemm_cl_kernel_ = getSgemmClTransBKernel();
  } else {
    kernel_func_ = "sgemm_cl_transAB";
    sgemm_cl_kernel_ = getSgemmClTransABKernel();
  }

  ClContext::SharedPtrClKernel kernel_sgemm_ptr =
    blas_cc->registerClKernel(sgemm_cl_kernel_, kernel_func_);
  if (!kernel_sgemm_ptr) {
    return;
  }

  sgemm_cl_internal<float>(kernel_sgemm_ptr, TransA, TransB, A, B, C, M, N, K,
                           lda, ldb, ldc);
}

void addition_cl(const float *input, float *res, unsigned int size_input,
                 unsigned int size_res) {
  ClContext::SharedPtrClKernel kernel_addition_ptr =
    blas_cc->registerClKernel(getAdditionClKernel(), "addition_cl");

  if (!kernel_addition_ptr) {
    return;
  }

  addition_cl_internal<float>(kernel_addition_ptr, input, res, size_input,
                              size_res);
}

void sscal_cl(float *X, const unsigned int N, const float alpha) {
  ClContext::SharedPtrClKernel kernel_ptr =
    blas_cc->registerClKernel(getSscalClKernel(), "sscal_cl");

  if (!kernel_ptr) {
    return;
  }

  sscal_cl_internal<float>(kernel_ptr, X, N, alpha);
}

void transpose_cl_axis(const float *in, float *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis) {
  ClContext::SharedPtrClKernel kernel_transpose_ptr;
  switch (axis) {
  case 0:
    kernel_transpose_ptr = blas_cc->registerClKernel(
      getTransposeClKernelAxis0(), "transpose_cl_axis0");
    break;
  case 1:
    kernel_transpose_ptr = blas_cc->registerClKernel(
      getTransposeClKernelAxis1(), "transpose_cl_axis1");
    break;
  case 2:
    kernel_transpose_ptr = blas_cc->registerClKernel(
      getTransposeClKernelAxis2(), "transpose_cl_axis2");
    break;
  default:
    throw std::invalid_argument("failed to register CL kernel");
    break;
  }
  if (!kernel_transpose_ptr) {
    return;
  }

  transpose_cl_axis_internal<float>(kernel_transpose_ptr, in, res,
                                    input_batch_size, input_channels,
                                    input_height, input_width, axis);
}
} // namespace nntrainer
