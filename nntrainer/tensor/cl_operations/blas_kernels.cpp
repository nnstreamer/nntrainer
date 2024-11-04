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

#include <blas_kernel_strings.h>
#include <blas_kernels.h>

namespace nntrainer {

void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              unsigned int dim1, unsigned int dim2, unsigned int lda) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemv_ptr =
      cl_context_ref.registerClKernel(sgemv_cl_kernel_, "sgemv_cl");
    if (!kernel_sgemv_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;
    size_t dim2_size = sizeof(float) * dim2;
    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          dim1 * dim2 * sizeof(float), true, (void *)matAdata);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim2_size, true,
                          (void *)vecXdata);

    opencl::Buffer inOutY(cl_context_ref.context_inst_, dim1_size, false,
                          vecYdata);

    result = kernel_sgemv_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(3, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sgemv_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    vecYdata = (float *)(inOutY.MapBuffer(cl_context_ref.command_queue_inst_, 0,
                                          dim1_size, true));
    result = inOutY.UnMapBuffer(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1) {

  bool result = false;

  float cl_ret = 0;

  do {
    ClContext::SharedPtrClKernel kernel_dot_ptr =
      cl_context_ref.registerClKernel(dot_cl_kernel_, "dot_cl");
    if (!kernel_dot_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim1_size, true,
                          (void *)vecAdata);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim1_size, true,
                          (void *)vecXdata);

    opencl::Buffer dotResult(cl_context_ref.context_inst_, sizeof(float), false,
                             &cl_ret);

    result = kernel_dot_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(3, &dotResult, sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_dot_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    float *tmp = (float *)(dotResult.MapBuffer(
      cl_context_ref.command_queue_inst_, 0, sizeof(float), true));
    cl_ret = *tmp;
    result = dotResult.UnMapBuffer(cl_context_ref.command_queue_inst_, tmp);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc) {

  std::string kernel_func_;
  std::string sgemm_cl_kernel_;

  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans";
    sgemm_cl_kernel_ = sgemm_cl_noTrans_kernel_;
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA";
    sgemm_cl_kernel_ = sgemm_cl_transA_kernel_;
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB";
    sgemm_cl_kernel_ = sgemm_cl_transB_kernel_;
  } else {
    kernel_func_ = "sgemm_cl_transAB";
    sgemm_cl_kernel_ = sgemm_cl_transAB_kernel_;
  }

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemm_ptr =
      cl_context_ref.registerClKernel(sgemm_cl_kernel_, kernel_func_);
    if (!kernel_sgemm_ptr) {
      break;
    }

    // sizes will be same for transpose
    size_t m_k_size = M * K * sizeof(float);
    size_t k_n_size = K * N * sizeof(float);
    size_t m_n_size = M * N * sizeof(float);

    opencl::Buffer inputA(cl_context_ref.context_inst_, m_k_size, true,
                          (void *)A);

    opencl::Buffer inputB(cl_context_ref.context_inst_, k_n_size, true,
                          (void *)B);

    opencl::Buffer inOutC(cl_context_ref.context_inst_, m_n_size, false, C);

    result = kernel_sgemm_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(1, &inputB, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(2, &inOutC, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(3, &K, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(5, &ldb, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(6, &ldc, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)M, (int)N, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sgemm_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    C = (float *)(inOutC.MapBuffer(cl_context_ref.command_queue_inst_, 0,
                                   m_n_size, true));
    result = inOutC.UnMapBuffer(cl_context_ref.command_queue_inst_, C);
    if (!result) {
      break;
    }

  } while (false);
}

void addition_cl(const float *input, float *res, unsigned int size) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_addition_ptr =
      cl_context_ref.registerClKernel(addition_cl_kernel_, "addition_cl");
    if (!kernel_addition_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * size;
    opencl::Buffer inputA(cl_context_ref.context_inst_, dim1_size, true,
                          (void *)input);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim1_size, false,
                            res);

    result =
      kernel_addition_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_addition_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_addition_ptr->SetKernelArguments(2, &size, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)size, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value
    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_addition_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    res = (float *)(inOutRes.MapBuffer(cl_context_ref.command_queue_inst_, 0,
                                       dim1_size, true));
    result = inOutRes.UnMapBuffer(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void sscal_cl(float *X, const unsigned int N, const float alpha) {
  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_ptr =
      cl_context_ref.registerClKernel(sscal_cl_kernel_, "sscal_cl");

    if (!kernel_ptr) {
      break;
    }

    size_t x_size = N * sizeof(float);

    opencl::Buffer inputX(cl_context_ref.context_inst_, x_size, false, X);

    result = kernel_ptr->SetKernelArguments(0, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_ptr->SetKernelArguments(1, &alpha, sizeof(float));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)N, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    X = (float *)(inputX.MapBuffer(cl_context_ref.command_queue_inst_, 0,
                                   x_size, 0));
    result = inputX.UnMapBuffer(cl_context_ref.command_queue_inst_, X);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
