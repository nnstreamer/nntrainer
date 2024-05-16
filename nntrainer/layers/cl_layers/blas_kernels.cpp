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

#include <blas_kernels.h>

namespace nntrainer {

std::string sgemv_cl_kernel_ =
  R"(__kernel void sgemv_cl(const __global float* A, const __global float* X,
                      __global float* Y, unsigned int M, unsigned int N) {                                            
        unsigned int i;
        i = get_global_id(0);                         
        float y0 = 0.0f;
        for (unsigned int j = 0; j < M; j++)                         
            y0 += A[i + j * N] * X[j]; 
        Y[i] = y0;                            
          
    })";

std::string dot_cl_kernel_ =
  R"(__kernel void dot_cl(const __global float* A, const __global float* X, unsigned int K, __global float* res) {
        *res = 0;
        for (unsigned int i = 0; i < K; i++){
            *res += A[i] * X[i];
        }
    })";

std::string sgemm_cl_kernel_ =
  R"(__kernel void sgemm_cl(const __global float* A, const __global float* B,
                      __global float* C, unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        for (unsigned int n = 0; n < N; ++n) {
          float c = 0.0f;
          for (unsigned int k = 0; k < K; ++k) {
            float a, b;
            a = A[m * lda + k];
            b = B[k * ldb + n];
            c += a * b;
          }
          C[m * ldc + n] = c;
        }
    })";

/**
 * @brief declaring global kernel objects
 */
opencl::Kernel kernel_sgemv;
opencl::Kernel kernel_sgemm;
opencl::Kernel kernel_dot;

void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              unsigned int dim1, unsigned int dim2, unsigned int lda,
              RunLayerContext &context) {

  bool result = false;

  do {
    result = context.clCreateKernel(sgemv_cl_kernel_,
                                    context.LayerKernel::SGEMV, kernel_sgemv);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;
    size_t dim2_size = sizeof(float) * dim2;
    opencl::Buffer inputA(context.context_inst_, dim1_size * dim2_size, true,
                          nullptr);

    opencl::Buffer inputX(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inOutY(context.context_inst_, dim2_size, true, nullptr);

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

    result = kernel_sgemv.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv.SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv.SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv.SetKernelArguments(3, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemv.SetKernelArguments(4, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      kernel_sgemv, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

float dot_cl(const float *matAdata, const float *vecXdata, unsigned int dim1,
             RunLayerContext &context) {

  bool result = false;

  float cl_ret = 0;

  do {
    result = context.clCreateKernel(dot_cl_kernel_, context.LayerKernel::DOT,
                                    kernel_dot);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;

    opencl::Buffer inputA(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inputX(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer dotResult(context.context_inst_, sizeof(float), true,
                             &cl_ret);

    result = inputA.WriteData(context.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(context.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = kernel_dot.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot.SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot.SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_dot.SetKernelArguments(3, &dotResult, sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      kernel_dot, work_groups_count, work_group_size);
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

void sgemm_cl(const float *A, const float *B, float *C, unsigned int M,
              unsigned int N, unsigned int K, unsigned int lda,
              unsigned int ldb, unsigned int ldc, RunLayerContext &context) {

  bool result = false;

  do {
    result = context.clCreateKernel(sgemm_cl_kernel_,
                                    context.LayerKernel::SGEMM, kernel_sgemm);
    if (!result) {
      break;
    }

    size_t m_size = sizeof(float) * M;
    size_t n_size = sizeof(float) * N;
    size_t k_size = sizeof(float) * K;
    opencl::Buffer inputA(context.context_inst_, m_size * k_size, true,
                          nullptr);

    opencl::Buffer inputB(context.context_inst_, k_size * n_size, true,
                          nullptr);

    opencl::Buffer inOutC(context.context_inst_, m_size * n_size, true,
                          nullptr);

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

    result = kernel_sgemm.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm.SetKernelArguments(1, &inputB, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm.SetKernelArguments(2, &inOutC, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm.SetKernelArguments(3, &M, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm.SetKernelArguments(4, &N, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm.SetKernelArguments(5, &K, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm.SetKernelArguments(6, &lda, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm.SetKernelArguments(7, &ldb, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm.SetKernelArguments(8, &ldc, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)M, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      kernel_sgemm, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutC.ReadData(context.command_queue_inst_, C);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
