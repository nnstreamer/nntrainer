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

#include <blas_kernels.h>

namespace nntrainer {

std::string sgemv_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
    __kernel void sgemv_cl_fp16(const __global half* A, const __global half* X,
                      __global half* Y, unsigned int N, unsigned int lda) {                                            
        unsigned int i;
        i = get_global_id(0);                         
        half y0 = 0.0f;
        for (unsigned int j = 0; j < N; j++)                         
            y0 += A[i + j * lda] * X[j]; 
        Y[i] = y0;                            
          
    })";

std::string dot_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void dot_cl_fp16(const __global half* A, const __global half* X, unsigned int K, __global half* res) {
        *res = 0;
        for (unsigned int i = 0; i < K; i++){
            *res += A[i] * X[i];
        }
    })";

std::string sgemm_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void sgemm_cl_fp16(const __global half* A, const __global half* B,
                      __global half* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        half c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          half a, b;
          a = A[m * lda + k];
          b = B[k * ldb + n];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

std::string addition_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void addition_cl_fp16(__global const half* input, __global half* output, const unsigned int size) {
    size_t idx = get_global_id(0);
    if (idx < size) {
        output[idx] = output[idx] + input[idx];
    }
  })";

/**
 * @brief defining global kernel objects
 */
opencl::Kernel kernel_sgemv_fp16;
opencl::Kernel kernel_sgemm_fp16;
opencl::Kernel kernel_dot_fp16;
opencl::Kernel kernel_addition_fp16;

void sgemv_cl(const __fp16 *matAdata, const __fp16 *vecXdata, __fp16 *vecYdata,
              unsigned int dim1, unsigned int dim2, unsigned int lda,
              RunLayerContext &context) {

  bool result = false;

  do {
    result = context.clCreateKernel(sgemv_cl_kernel_fp16_,
                                    context.LayerKernel::SGEMV_FP16,
                                    kernel_sgemv_fp16);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(cl_half) * dim1;
    size_t dim2_size = sizeof(cl_half) * dim2;
    opencl::Buffer inputA(context.context_inst_, dim1 * dim2 * sizeof(cl_half),
                          true, nullptr);

    opencl::Buffer inputX(context.context_inst_, dim2_size, true, nullptr);

    opencl::Buffer inOutY(context.context_inst_, dim1_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(context.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16.SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16.SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16.SetKernelArguments(3, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemv_fp16.SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      kernel_sgemv_fp16, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

__fp16 dot_cl(const __fp16 *vecAdata, const __fp16 *vecXdata, unsigned int dim1,
              RunLayerContext &context) {

  bool result = false;

  __fp16 cl_ret = 0;

  do {
    result = context.clCreateKernel(
      dot_cl_kernel_fp16_, context.LayerKernel::DOT_FP16, kernel_dot_fp16);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(cl_half) * dim1;

    opencl::Buffer inputA(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inputX(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer dotResult(context.context_inst_, sizeof(__fp16), true,
                             &cl_ret);

    result = inputA.WriteData(context.command_queue_inst_, vecAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(context.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = kernel_dot_fp16.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_fp16.SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_fp16.SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_dot_fp16.SetKernelArguments(3, &dotResult, sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      kernel_dot_fp16, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = dotResult.ReadData(context.command_queue_inst_, &cl_ret);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

void sgemm_cl(const __fp16 *A, const __fp16 *B, __fp16 *C, unsigned int M,
              unsigned int N, unsigned int K, unsigned int lda,
              unsigned int ldb, unsigned int ldc, RunLayerContext &context) {

  bool result = false;

  do {
    result = context.clCreateKernel(sgemm_cl_kernel_fp16_,
                                    context.LayerKernel::SGEMM_FP16,
                                    kernel_sgemm_fp16);
    if (!result) {
      break;
    }

    size_t m_k_size = M * K * sizeof(cl_half);
    size_t k_n_size = K * N * sizeof(cl_half);
    size_t m_n_size = M * N * sizeof(cl_half);

    opencl::Buffer inputA(context.context_inst_, m_k_size, true, nullptr);

    opencl::Buffer inputB(context.context_inst_, k_n_size, true, nullptr);

    opencl::Buffer inOutC(context.context_inst_, m_n_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, A);
    if (!result) {
      break;
    }

    result = inputB.WriteData(context.command_queue_inst_, B);
    if (!result) {
      break;
    }

    result = inOutC.WriteData(context.command_queue_inst_, C);
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16.SetKernelArguments(1, &inputB, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16.SetKernelArguments(2, &inOutC, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16.SetKernelArguments(3, &K, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16.SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16.SetKernelArguments(5, &ldb, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_fp16.SetKernelArguments(6, &ldc, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)M, (int)N, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      kernel_sgemm_fp16, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutC.ReadData(context.command_queue_inst_, C);
    if (!result) {
      break;
    }

  } while (false);
}

void addition_cl(const __fp16 *input, __fp16 *res, unsigned int size,
                 RunLayerContext &context) {

  bool result = false;

  do {
    result = context.clCreateKernel(addition_cl_kernel_fp16_,
                                    context.LayerKernel::ADD_FP16,
                                    kernel_addition_fp16);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(cl_half) * size;
    opencl::Buffer inputA(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inOutRes(context.context_inst_, dim1_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, input);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result =
      kernel_addition_fp16.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_addition_fp16.SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_addition_fp16.SetKernelArguments(2, &size, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)size, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value
    result = context.command_queue_inst_.DispatchCommand(
      kernel_addition_fp16, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
