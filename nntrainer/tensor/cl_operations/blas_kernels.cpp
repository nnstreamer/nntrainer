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

void sgemv_q6_k_cl(void *matAdata, float *vecXdata, float *vecYdata,
                   unsigned int M, unsigned int N) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_q6_k_sgemv_ptr;

  kernel_q6_k_sgemv_ptr =
    blas_cc->registerClKernel(getQ6KSgemvClKernel(), "kernel_mul_mv_q6_K_f32");

  if (!kernel_q6_k_sgemv_ptr) {
    ml_loge("Failed to register kernel_q6_k_sgemv_ptr");
    return;
  }

  const size_t q6k_bytes = 210 * M * N / 256;

  result = blas_cc->command_queue_inst_.enqueueSVMUnmap(matAdata);
  if (!result) {
    ml_loge("Failed to write data to input buffer A for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = blas_cc->command_queue_inst_.enqueueSVMUnmap(vecXdata);
  if (!result) {
    ml_loge("Failed to write data to input buffer B for kernel_q6_k_sgemv_ptr");
    return;
  }

  int ne00 = M; // number of rows in matrix X
  int ne01 = N; // number of columns in matrix X
  int ne02 = 1; // number of channels in matrix X
  int ne10 = M; // number of rows in vector A
  int ne11 = 1; // number of columns in vector A
  int ne12 = 1; // number of channels in vector A
  int ne13 = 1; // number of channels in vector A (Need to check)
  int ne0 = N;  // number of rows in output vector Y
  int ne1 = 1;  // number of columns in output vector Y

  int r2 = 1; // number of batches in vector A
  int r3 = 1; // number of batches in matrix X

  int nth0 = 2;
  int nth1 = 16;

  cl_ulong offset0 = 0;
  cl_ulong offset1 = 0;
  cl_ulong offsetd = 0;

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(0, matAdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 0 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(1, &offset0, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 1 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(2, vecXdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 2 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(3, &offset1, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 3 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(4, vecYdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 4 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(5, &offsetd, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 5 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(6, &ne00, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 6 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(7, &ne01, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 7 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(8, &ne02, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 8 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(9, &ne10, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 9 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(10, &ne12, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 10 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(11, &ne0, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 11 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(12, &ne1, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 12 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(13, &r2, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 13 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(14, &r3, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 14 for kernel_q6_k_sgemv_ptr");
    return;
  }

#define N_SIMDWIDTH 16
#define N_SIMDGROUP 2

  const int work_groups_count[3] = {((ne0 + N_SIMDGROUP - 1) / N_SIMDGROUP) *
                                      (N_SIMDGROUP * N_SIMDWIDTH),
                                    ne1, 1};
  /// @todo: create a group size by device & input
  const int work_group_size[3] = {32, 1, 1};

  result = opencl::CommandQueueManager::Global().DispatchCommand(
    kernel_q6_k_sgemv_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel q6_k_sgemv");
    return;
  }

  result = blas_cc->command_queue_inst_.enqueueSVMMap(vecYdata,
                                                      N * sizeof(float), true);

  if (!result) {
    ml_loge(
      "Failed to read data from the output buffer for kernel_q6_k_sgemv_ptr");

    return;
  }
}

void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

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
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

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

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

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
  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_addition_ptr =
    blas_cc->registerClKernel(getAdditionClKernel(), "addition_cl");
  if (!kernel_addition_ptr) {
    return;
  }

  addition_cl_internal<float>(kernel_addition_ptr, input, res, size_input,
                              size_res);
}

void sscal_cl(float *X, const unsigned int N, const float alpha) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

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
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

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

void flatten_block_q4_0_cl(const void *src, void *dst_q, void *dst_d,
                           unsigned int num_blocks) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    getConvertBlockQ4_0Kernel(), "kernel_convert_block_q4_0_noshuffle");
  if (!kernel_ptr) {
    ml_loge("Failed to register kernel_ptr for flatten_block_q4_0_cl");
    return;
  }

  int argIdx = 0;

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src);
  if (!result) {
    ml_loge("Failed to set kernel argument 0 for flatten_block_q4_0_cl");
    return;
  }

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, dst_q);
  if (!result) {
    ml_loge("Failed to set kernel argument 1 for flatten_block_q4_0_cl");
    return;
  }

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, dst_d);
  if (!result) {
    ml_loge("Failed to set kernel argument 2 for flatten_block_q4_0_cl");
    return;
  }

  const int work_groups_count[3] = {(int)num_blocks, 1, 1};
  const int work_group_size[3] = {64, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for flatten_block_q4_0_cl");
    return;
  }
}

void restore_block_q4_0_cl(const void *src_q, const void *src_d, void *dst,
                           unsigned int num_blocks) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    getConvertBlockQ4_0Kernel(), "kernel_restore_block_q4_0");
  if (!kernel_ptr) {
    ml_loge("Failed to register kernel_ptr for restore_block_q4_0_cl");
    return;
  }

  int argIdx = 0;

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src_q);
  if (!result) {
    ml_loge("Failed to set kernel argument 0 for restore_block_q4_0_cl");
    return;
  }

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src_d);
  if (!result) {
    ml_loge("Failed to set kernel argument 1 for restore_block_q4_0_cl");
    return;
  }

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, dst);
  if (!result) {
    ml_loge("Failed to set kernel argument 2 for restore_block_q4_0_cl");
    return;
  }

  const int work_groups_count[3] = {(int)num_blocks, 1, 1};
  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for restore_block_q4_0_cl");
    return;
  }
}

} // namespace nntrainer
