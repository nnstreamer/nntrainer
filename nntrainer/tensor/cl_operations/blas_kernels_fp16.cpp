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

#include <blas_kernel_strings.h>
#include <blas_kernels.h>

namespace nntrainer {

void sgemv_cl(const __fp16 *matAdata, const __fp16 *vecXdata, __fp16 *vecYdata,
              unsigned int dim1, unsigned int dim2, unsigned int lda) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemv_fp16_ptr =
      cl_context_ref.registerClKernel(sgemv_cl_kernel_fp16_, "sgemv_cl_fp16");
    if (!kernel_sgemv_fp16_ptr) {
      break;
    }

    size_t dim1_size = sizeof(cl_half) * dim1;
    size_t dim2_size = sizeof(cl_half) * dim2;
    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          dim1 * dim2 * sizeof(cl_half), true,
                          (void *)matAdata);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim2_size, true,
                          (void *)vecXdata);

    opencl::Buffer inOutY(cl_context_ref.context_inst_, dim1_size, false,
                          vecYdata);

    result =
      kernel_sgemv_fp16_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_sgemv_fp16_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_sgemv_fp16_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16_ptr->SetKernelArguments(3, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sgemv_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    vecYdata = (__fp16 *)(inOutY.MapBuffer(cl_context_ref.command_queue_inst_,
                                           0, dim1_size, true));
    result = inOutY.UnMapBuffer(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

__fp16 dot_cl(const __fp16 *vecAdata, const __fp16 *vecXdata,
              unsigned int dim1) {

  bool result = false;

  __fp16 cl_ret = 0;

  do {
    ClContext::SharedPtrClKernel kernel_dot_fp16_ptr =
      cl_context_ref.registerClKernel(dot_cl_kernel_fp16_, "dot_cl_fp16");

    if (!kernel_dot_fp16_ptr) {
      break;
    }

    size_t dim1_size = sizeof(cl_half) * dim1;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim1_size, true,
                          (void *)vecAdata);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim1_size, true,
                          (void *)vecXdata);

    opencl::Buffer dotResult(cl_context_ref.context_inst_, sizeof(__fp16),
                             false, &cl_ret);

    result =
      kernel_dot_fp16_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_dot_fp16_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_fp16_ptr->SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_dot_fp16_ptr->SetKernelArguments(3, &dotResult, sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_dot_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    __fp16 *tmp = (__fp16 *)(dotResult.MapBuffer(
      cl_context_ref.command_queue_inst_, 0, sizeof(__fp16), true));
    cl_ret = *tmp;
    result = dotResult.UnMapBuffer(cl_context_ref.command_queue_inst_, tmp);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

void sgemm_cl(bool TransA, bool TransB, const __fp16 *A, const __fp16 *B,
              __fp16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc) {

  std::string kernel_func_;
  std::string sgemm_cl_kernel_fp16_;

  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans_fp16";
    sgemm_cl_kernel_fp16_ = sgemm_cl_noTrans_kernel_fp16_;
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA_fp16";
    sgemm_cl_kernel_fp16_ = sgemm_cl_transA_kernel_fp16_;
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB_fp16";
    sgemm_cl_kernel_fp16_ = sgemm_cl_transB_kernel_fp16_;
  } else {
    kernel_func_ = "sgemm_cl_transAB_fp16";
    sgemm_cl_kernel_fp16_ = sgemm_cl_transAB_kernel_fp16_;
  }

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemm_fp16_ptr =
      cl_context_ref.registerClKernel(sgemm_cl_kernel_fp16_, kernel_func_);
    if (!kernel_sgemm_fp16_ptr) {
      break;
    }

    // sizes will be same for transpose
    size_t m_k_size = M * K * sizeof(cl_half);
    size_t k_n_size = K * N * sizeof(cl_half);
    size_t m_n_size = M * N * sizeof(cl_half);

    opencl::Buffer inputA(cl_context_ref.context_inst_, m_k_size, true,
                          (void *)A);

    opencl::Buffer inputB(cl_context_ref.context_inst_, k_n_size, true,
                          (void *)B);

    opencl::Buffer inOutC(cl_context_ref.context_inst_, m_n_size, false, C);

    result =
      kernel_sgemm_fp16_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_sgemm_fp16_ptr->SetKernelArguments(1, &inputB, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_sgemm_fp16_ptr->SetKernelArguments(2, &inOutC, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(3, &K, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(5, &ldb, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16_ptr->SetKernelArguments(6, &ldc, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)M, (int)N, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sgemm_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    C = (__fp16 *)(inOutC.MapBuffer(cl_context_ref.command_queue_inst_, 0,
                                    m_n_size, true));
    result = inOutC.UnMapBuffer(cl_context_ref.command_queue_inst_, C);
    if (!result) {
      break;
    }

  } while (false);
}

void addition_cl(const __fp16 *input, __fp16 *res, unsigned int size) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_addition_fp16_ptr =
      cl_context_ref.registerClKernel(addition_cl_kernel_fp16_,
                                      "addition_cl_fp16");
    if (!kernel_addition_fp16_ptr) {
      break;
    }

    size_t dim1_size = sizeof(cl_half) * size;
    opencl::Buffer inputA(cl_context_ref.context_inst_, dim1_size, true,
                          (void *)input);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim1_size, false,
                            res);

    result =
      kernel_addition_fp16_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_addition_fp16_ptr->SetKernelArguments(1, &inOutRes,
                                                          sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_addition_fp16_ptr->SetKernelArguments(2, &size, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)size, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value
    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_addition_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    res = (__fp16 *)(inOutRes.MapBuffer(cl_context_ref.command_queue_inst_, 0,
                                        dim1_size, true));
    result = inOutRes.UnMapBuffer(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void sscal_cl(__fp16 *X, const unsigned int N, const float alpha) {
  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sscal_fp16_ptr =
      cl_context_ref.registerClKernel(sscal_cl_kernel_fp16_, "sscal_cl_fp16");

    if (!kernel_sscal_fp16_ptr) {
      break;
    }

    size_t x_size = N * sizeof(cl_half);

    opencl::Buffer inputX(cl_context_ref.context_inst_, x_size, false, X);

    result =
      kernel_sscal_fp16_ptr->SetKernelArguments(0, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_sscal_fp16_ptr->SetKernelArguments(1, &alpha, sizeof(float));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)N, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sscal_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    // to avoid cache inconsistency
    X = (__fp16 *)(inputX.MapBuffer(cl_context_ref.command_queue_inst_, 0,
                                    x_size, 0));
    result = inputX.UnMapBuffer(cl_context_ref.command_queue_inst_, X);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
