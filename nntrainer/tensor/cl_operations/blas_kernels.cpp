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

std::string transpose_cl_kernel_axis0 =
  R"(__kernel void transpose_cl_axis0(__global const float* in, 
                               __global float* output,
                               const int batch_size, 
                               const int channels, 
                               const int height, 
                               const int width) {

    // Calculate h and w from the global IDs
    int h = get_global_id(0);
    int w = get_global_id(1);

    if (h < height && w < width) {
        for (int c = 0; c < channels; ++c) {
            for (int n = 0; n < batch_size; ++n) {
                // Calculate the input and output indices
                int input_index = n * (channels * height * width) + c * (height * width) + h * width + w;
                int output_index = n * (channels * height * width) + h * (channels * width) + c * width + w;

                // Transpose channel and height, copying data from input to output
                output[output_index] = in[input_index];
            }
        }
    }

})";

std::string transpose_cl_kernel_axis1 =
  R"(__kernel void transpose_cl_axis1(__global const float* in, 
                               __global float* output,
                               const int batch_size, 
                               const int channels, 
                               const int height, 
                               const int width) {

    // Calculate h and w from the global IDs
    int h = get_global_id(0);
    int w = get_global_id(1);

    if (h < height && w < width) {
        for (int c = 0; c < channels; ++c) {
            for (int n = 0; n < batch_size; ++n) {
                // Calculate the input and output indices
                int input_index = n * (channels * height * width) + c * (height * width) + h * width + w;
                int output_index = n * (channels * height * width) + c * (height * width) + w * height + h;

                // Transpose height and width, copying data from input to output
                output[output_index] = in[input_index];
            }
        }
    }

})";

std::string transpose_cl_kernel_axis2 =
  R"(__kernel void transpose_cl_axis2(__global const float* in, 
                               __global float* output,
                               const int batch_size, 
                               const int channels, 
                               const int height, 
                               const int width) {

    // Calculate c and w from the global IDs
    int c = get_global_id(0);
    int w = get_global_id(1);

    if (c < channels && w < width) {
        for (int h = 0; h < height; ++h) {
            for (int n = 0; n < batch_size; ++n) {
                // Calculate the input and output indices
                int input_index = n * (channels * height * width) + c * (height * width) + h * width + w;
                int output_index = n * (channels * height * width) + w * (height * channels) + h * channels + c;

                // Transpose channel and width, copying data from input to output
                output[output_index] = in[input_index];
            }
        }
    }

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
opencl::Kernel kernel_transpose_axis0;
opencl::Kernel kernel_transpose_axis1;
opencl::Kernel kernel_transpose_axis2;

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
                          dim1 * dim2 * sizeof(float), true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim2_size, true,
                          nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_, dim1_size, true,
                          nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

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

    result = inOutY.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
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
                          nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim1_size, true,
                          nullptr);

    opencl::Buffer dotResult(cl_context_ref.context_inst_, sizeof(float), true,
                             &cl_ret);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, vecAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

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

    result = dotResult.ReadData(cl_context_ref.command_queue_inst_, &cl_ret);
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
                          nullptr);

    opencl::Buffer inputB(cl_context_ref.context_inst_, k_n_size, true,
                          nullptr);

    opencl::Buffer inOutC(cl_context_ref.context_inst_, m_n_size, true,
                          nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, A);
    if (!result) {
      break;
    }

    result = inputB.WriteData(cl_context_ref.command_queue_inst_, B);
    if (!result) {
      break;
    }

    result = inOutC.WriteData(cl_context_ref.command_queue_inst_, C);
    if (!result) {
      break;
    }

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

    result = inOutC.ReadData(cl_context_ref.command_queue_inst_, C);
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
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim1_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, input);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

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

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
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

    opencl::Buffer inputX(cl_context_ref.context_inst_, x_size, false, nullptr);

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, X);
    if (!result) {
      break;
    }

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

    result = inputX.ReadData(cl_context_ref.command_queue_inst_, X);
    if (!result) {
      break;
    }

  } while (false);
}

void transpose_cl_axis0(const float *in, float *res,
                        unsigned int input_batch_size,
                        unsigned int input_channels, unsigned int input_height,
                        unsigned int input_width) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_transpose_ptr =
      cl_context_ref.registerClKernel(transpose_cl_kernel_axis0,
                                      "transpose_cl_axis0");

    if (!kernel_transpose_ptr) {
      break;
    }

    size_t dim_size = sizeof(float) * input_batch_size * input_height *
                      input_width * input_channels;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim_size, true,
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, in);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_transpose_ptr->SetKernelArguments(2, &input_batch_size,
                                                      sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_transpose_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void transpose_cl_axis1(const float *in, float *res,
                        unsigned int input_batch_size,
                        unsigned int input_channels, unsigned int input_height,
                        unsigned int input_width) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_transpose_ptr =
      cl_context_ref.registerClKernel(transpose_cl_kernel_axis1,
                                      "transpose_cl_axis1");

    if (!kernel_transpose_ptr) {
      break;
    }

    size_t dim_size = sizeof(float) * input_batch_size * input_height *
                      input_width * input_channels;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim_size, true,
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, in);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_transpose_ptr->SetKernelArguments(2, &input_batch_size,
                                                      sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_transpose_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void transpose_cl_axis2(const float *in, float *res,
                        unsigned int input_batch_size,
                        unsigned int input_channels, unsigned int input_height,
                        unsigned int input_width) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_transpose_ptr =
      cl_context_ref.registerClKernel(transpose_cl_kernel_axis2,
                                      "transpose_cl_axis2");

    if (!kernel_transpose_ptr) {
      break;
    }

    size_t dim_size = sizeof(float) * input_batch_size * input_height *
                      input_width * input_channels;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim_size, true,
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, in);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_transpose_ptr->SetKernelArguments(2, &input_batch_size,
                                                      sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)input_channels, (int)input_width, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_transpose_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
