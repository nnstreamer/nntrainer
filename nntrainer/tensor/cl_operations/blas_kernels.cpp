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
                      __global float* Y, unsigned int N, unsigned int lda) {                                            
        unsigned int i;
        i = get_global_id(0);                         
        float y0 = 0.0f;
        for (unsigned int j = 0; j < N; j++)                         
            y0 += A[i + j * lda] * X[j]; 
        Y[i] = y0;                            
          
    })";

std::string dot_cl_kernel_ =
  R"(__kernel void dot_cl(const __global float* A, const __global float* X, unsigned int K, __global float* res) {
        *res = 0;
        for (unsigned int i = 0; i < K; i++){
            *res += A[i] * X[i];
        }
    })";

std::string sgemm_cl_noTrans_kernel_ =
  R"(__kernel void sgemm_cl_noTrans(const __global float* A, const __global float* B,
                      __global float* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        float c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          float a, b;
          a = A[m * lda + k];
          b = B[k * ldb + n];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

std::string sgemm_cl_transA_kernel_ =
  R"(__kernel void sgemm_cl_transA(const __global float* A, const __global float* B,
                      __global float* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
        
        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        float c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          float a, b;
          a = A[k * lda + m];
          b = B[k * ldb + n];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

std::string sgemm_cl_transB_kernel_ =
  R"(__kernel void sgemm_cl_transB(const __global float *A, const __global float *B,
                              __global float *C, unsigned int K,
                              unsigned int lda, unsigned int ldb,
                              unsigned int ldc) {

        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        float c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          float a, b;
          a = A[m * lda + k];
          b = B[n * ldb + k];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

std::string sgemm_cl_transAB_kernel_ =
  R"(__kernel void sgemm_cl_transAB(const __global float *A, const __global float *B,
                               __global float *C, unsigned int K,
                               unsigned int lda, unsigned int ldb,
                               unsigned int ldc) {

        unsigned int m = get_global_id(0);
        unsigned int n = get_global_id(1);
        float c = 0.0f;
        for (unsigned int k = 0; k < K; ++k) {
          float a, b;
          a = A[k * lda + m];
          b = B[n * ldb + k];
          c += a * b;
        }
        C[m * ldc + n] = c;
    })";

std::string addition_cl_kernel_ =
  R"(__kernel void addition_cl(__global const float* input, __global float* output, const unsigned int size) {
    #pragma printf_support
    size_t idx = get_global_id(0);
    if (idx < size) {
        output[idx] = output[idx] + input[idx];
    }
  })";

std::string sscal_cl_kernel_ =
  R"(__kernel void sscal_cl(__global float* X, const float alpha) {
        
        unsigned int i = get_global_id(0);
        X[i] *= alpha;
    })";

/**
 * @brief defining global kernel objects
 */
opencl::Kernel kernel_sgemv;
opencl::Kernel kernel_sgemm_transAB;
opencl::Kernel kernel_sgemm_transA;
opencl::Kernel kernel_sgemm_transB;
opencl::Kernel kernel_sgemm_noTrans;
opencl::Kernel kernel_dot;
opencl::Kernel kernel_addition;
opencl::Kernel kernel_sscal;

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
    opencl::Buffer inputA(context.context_inst_, dim1 * dim2 * sizeof(float),
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

    result = kernel_sgemv.SetKernelArguments(3, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemv.SetKernelArguments(4, &lda, sizeof(int));
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

float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1,
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

    result = inputA.WriteData(context.command_queue_inst_, vecAdata);
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

void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc,
              RunLayerContext &context) {

  opencl::Kernel *kernel_sgemm = nullptr;
  RunLayerContext::LayerKernel layerKernel;
  std::string sgemm_cl_kernel_;

  if (!TransA && !TransB) {
    kernel_sgemm = &kernel_sgemm_noTrans;
    layerKernel = context.LayerKernel::SGEMM_NOTRANS;
    sgemm_cl_kernel_ = sgemm_cl_noTrans_kernel_;
  } else if (TransA && !TransB) {
    kernel_sgemm = &kernel_sgemm_transA;
    layerKernel = context.LayerKernel::SGEMM_TRANSA;
    sgemm_cl_kernel_ = sgemm_cl_transA_kernel_;
  } else if (!TransA && TransB) {
    kernel_sgemm = &kernel_sgemm_transB;
    layerKernel = context.LayerKernel::SGEMM_TRANSB;
    sgemm_cl_kernel_ = sgemm_cl_transB_kernel_;
  } else {
    kernel_sgemm = &kernel_sgemm_transAB;
    layerKernel = context.LayerKernel::SGEMM_TRANSAB;
    sgemm_cl_kernel_ = sgemm_cl_transAB_kernel_;
  }

  bool result = false;

  do {
    result =
      context.clCreateKernel(sgemm_cl_kernel_, layerKernel, *kernel_sgemm);
    if (!result) {
      break;
    }

    // sizes will be same for transpose
    size_t m_k_size = M * K * sizeof(float);
    size_t k_n_size = K * N * sizeof(float);
    size_t m_n_size = M * N * sizeof(float);

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

    result = kernel_sgemm->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm->SetKernelArguments(1, &inputB, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm->SetKernelArguments(2, &inOutC, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm->SetKernelArguments(3, &K, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm->SetKernelArguments(5, &ldb, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm->SetKernelArguments(6, &ldc, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)M, (int)N, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      *kernel_sgemm, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutC.ReadData(context.command_queue_inst_, C);
    if (!result) {
      break;
    }

  } while (false);
}

void addition_cl(const float *input, float *res, unsigned int size,
                 RunLayerContext &context) {

  bool result = false;

  do {
    result = context.clCreateKernel(addition_cl_kernel_,
                                    context.LayerKernel::ADD, kernel_addition);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * size;
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

    result = kernel_addition.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_addition.SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_addition.SetKernelArguments(2, &size, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)size, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value
    result = context.command_queue_inst_.DispatchCommand(
      kernel_addition, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void sscal_cl(float *X, const unsigned int N, const float alpha,
              RunLayerContext &context) {
  bool result = false;

  do {
    result = context.clCreateKernel(sscal_cl_kernel_,
                                    context.LayerKernel::SSCAL, kernel_sscal);
    if (!result) {
      break;
    }

    size_t x_size = N * sizeof(float);

    opencl::Buffer inputX(context.context_inst_, x_size, false, nullptr);

    result = inputX.WriteData(context.command_queue_inst_, X);
    if (!result) {
      break;
    }

    result = kernel_sscal.SetKernelArguments(0, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sscal.SetKernelArguments(1, &alpha, sizeof(float));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)N, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      kernel_sscal, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inputX.ReadData(context.command_queue_inst_, X);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
