// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_strings.cpp
 * @date	April 01 2025
 * @brief	All blas OpenCL kernel strings
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "blas_kernel_strings.h"

namespace nntrainer {

const std::string &getSgemvClKernel() {
  static const std::string sgemv_cl_kernel_ =
    R"(__kernel void sgemv_cl(const __global float* A, const __global float* X,
                          __global float* Y, unsigned int N, unsigned int lda) {                                            
            unsigned int i;
            i = get_global_id(0);                         
            float y0 = 0.0f;
            for (unsigned int j = 0; j < N; j++)                         
                y0 += A[i + j * lda] * X[j]; 
            Y[i] = y0;                            
              
        })";
  return sgemv_cl_kernel_;
}

const std::string &getSgemvClNoTransKernel() {
  static const std::string sgemv_cl_noTrans_kernel_ =
    R"(__kernel void sgemv_cl_noTrans(const __global float* A, const __global float* X,
                          __global float* Y, unsigned int N, unsigned int lda) {                                            
            unsigned int i;
            i = get_global_id(0);                         
            float y0 = 0.0f;
            for (unsigned int j = 0; j < N; j++)                         
                y0 += A[j + i * lda] * X[j]; 
            Y[i] = y0;                            
              
        })";
  return sgemv_cl_noTrans_kernel_;
}

const std::string &getDotClKernel() {
  static const std::string dot_cl_kernel_ =
    R"(__kernel void dot_cl(const __global float* A, const __global float* X, unsigned int K, __global float* res) {
            *res = 0;
            for (unsigned int i = 0; i < K; i++){
                *res += A[i] * X[i];
            }
        })";
  return dot_cl_kernel_;
}

const std::string &getSgemmClNoTransKernel() {
  static const std::string sgemm_cl_noTrans_kernel_ =
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
  return sgemm_cl_noTrans_kernel_;
}

const std::string &getSgemmClTransAKernel() {
  static const std::string sgemm_cl_transA_kernel_ =
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
  return sgemm_cl_transA_kernel_;
}

const std::string &getSgemmClTransBKernel() {
  static const std::string sgemm_cl_transB_kernel_ =
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
  return sgemm_cl_transB_kernel_;
}

const std::string &getSgemmClTransABKernel() {
  static const std::string sgemm_cl_transAB_kernel_ =
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
  return sgemm_cl_transAB_kernel_;
}

const std::string &getAdditionClKernel() {
  static const std::string addition_cl_kernel_ =
    R"(__kernel void addition_cl(const __global float* input, __global float* output, unsigned int size_input, unsigned int size_res) {
        #pragma printf_support
        size_t idx = get_global_id(0);
        if (idx < size_res) {
            output[idx] = output[idx] + input[idx % size_input];
        }
      })";
  return addition_cl_kernel_;
}

const std::string &getSscalClKernel() {
  static const std::string sscal_cl_kernel_ =
    R"(__kernel void sscal_cl(__global float* X, const float alpha) {
            
            unsigned int i = get_global_id(0);
            X[i] *= alpha;
        })";
  return sscal_cl_kernel_;
}

const std::string &getTransposeClKernelAxis0() {
  static const std::string transpose_cl_kernel_axis0 =
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
  return transpose_cl_kernel_axis0;
}

const std::string &getTransposeClKernelAxis1() {
  static const std::string transpose_cl_kernel_axis1 =
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

  return transpose_cl_kernel_axis1;
}

const std::string &getTransposeClKernelAxis2() {
  static const std::string transpose_cl_kernel_axis2 =
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
  return transpose_cl_kernel_axis2;
}

const std::string &getSwiGluClKernel() {
  static const std::string swiglu_cl_kernel_ =
    R"(__kernel void swiglu_cl(__global const float *in1, __global const float *in2, __global float *out) {
int i = get_global_id(0);
float swish = in1[i] * exp(in1[i]) / (1 + exp(in1[i]));
out[i] = swish * in2[i];
})";
  return swiglu_cl_kernel_;
}

const std::string &getCopyClKernel() {
  static const std::string copy_cl_kernel_ =
    R"(__kernel void copy_cl(__global const float* input, 
                           __global float* output,
                           const int batchsize, 
                           const int channels, 
                           const int height, 
                           const int width) {
int i= get_global_id(0);
output[i] = input[i];
})";
  return copy_cl_kernel_;
}

const std::string &getConcatClAxis3Kernel() {
  static const std::string concat_cl_axis3_kernel_ =
    R"(__kernel void concat_cl_axis3(__global const float* in1, 
                                           __global const float* in2, 
                                           __global float* out,
                                           const int batch_size, 
                                           const int channels, 
                                           const int height, 
                                           const int width1, 
                                           const int width2) {
    int global_id = get_global_id(0);
    
    int total_width = width1 + width2;
    
    int width = total_width;

    // 4D space coordinates
    int w = global_id % total_width;
    int h = (global_id / total_width) % height;
    int c = (global_id / (total_width * height)) % channels;
    int b = global_id / (total_width * height * channels);

    int output_index = ((b * channels + c) * height + h) * total_width + w;
    
    // Determining if the index is in in1 or in2
    if (w < width1) {
        // in1 index calculation
        int input1_index = ((b * channels + c) * height + h) * width1 + w;
        out[output_index] = in1[input1_index];
  
    } else {
        // in2 index calculation
        int input2_index = ((b * channels + c) * height + h) * width2 + (w - width1);
        out[output_index] = in2[input2_index];
    }
})";
  return concat_cl_axis3_kernel_;
}

const std::string &getConcatClAxis2Kernel() {
  static const std::string concat_cl_axis2_kernel_ =
    R"(__kernel void concat_cl_axis2(__global const float* in1,
                             __global const float* in2,
                             __global float* out,
                             const int batch_size,
                             const int channels,
                             const int height1,
                             const int height2,
                             const int width) {
    
    int total_height = height1 + height2;
    int global_id = get_global_id(0);
    
    // Calculate the coordinates in the 4D space
    int w = global_id % width;
    int h = (global_id / width) % total_height;
    int c = (global_id / (width * total_height)) % channels;
    int b = global_id / (width * total_height * channels);

    // Calculate the offset for the current batch, channel, and width in the output tensor
    int output_index = ((b * channels + c) * total_height + h) * width + w;

    if (h < height1) {
        // Index within input1
        int input1_index = ((b * channels + c) * height1 + h) * width + w;
        out[output_index] = in1[input1_index];
    } else {
        // Index within input2
        int input2_index = ((b * channels + c) * height2 + (h - height1)) * width + w;
        out[output_index] = in2[input2_index];
    }

})";
  return concat_cl_axis2_kernel_;
}

const std::string &getConcatClAxis1Kernel() {
  static const std::string concat_cl_axis1_kernel_ =
    R"(__kernel void concat_cl_axis1(__global const float* in1, 
                                           __global const float* in2, 
                                           __global float* out,
                                           const int batch_size, 
                                           const int channels1, 
                                           const int channels2, 
                                           const int height, 
                                           const int width) {
    int global_id = get_global_id(0);
    
    int total_channels = channels1 + channels2;

    // Calculate the coordinates in the 4D space
    int w = global_id % width;
    int h = (global_id / width) % height;
    int c = (global_id / (width * height)) % total_channels;
    int b = global_id / (width * height * total_channels);

    // Calculate the offset for the current batch, height, and width in the output tensor
    int output_index = ((b * total_channels + c) * height + h) * width + w;

    if (c < channels1) {
        // Index within input1
        int input1_index = ((b * channels1 + c) * height + h) * width + w;
        out[output_index] = in1[input1_index];
    } else {
        // Index within input2
        int input2_index = ((b * channels2 + (c - channels1)) * height + h) * width + w;
        out[output_index] = in2[input2_index];
    }
})";
  return concat_cl_axis1_kernel_;
}

const std::string &getRMSNormClKernel() {
  static const std::string rmsnorm_cl_kernel_ =
    R"(__kernel void rmsnorm_cl(
    __global const float *input,  // Input tensor
    __global float *output,    // Output tensor
    __global const float *alpha,  // Alpha values (one for each width)
    float epsilon,
    int B,                  // Number of batches
    int C,                  // Number of channels
    int H,                  // Height of feature map
    int W                   // Width of feature map
) {
    // Compute the corresponding batch, height, and channel indices
    int n = get_global_id(0) / C;
    int c = get_global_id(0) % C;
    int h = get_global_id(1);
    int index = ((n * C + c) * H + h) * W;
    // Calculate RMS norm for the current channel, height, and batch
    float sum_squares = 0.0f;
    for (int j = 0; j < W; ++j) {
        sum_squares += input[index+j] * input[index+j];
    }
    sum_squares /= W;
    float rms_norm = sqrt(sum_squares + epsilon);
    // Each work item processes all width elements for its specific n, h, c
    for (int w = 0; w < W; ++w) {
        output[index+w] = (input[index+w] / rms_norm) * alpha[w];
    }
}
)";

  return rmsnorm_cl_kernel_;
}

#ifdef ENABLE_FP16
const std::string &getHgemvClKernel() {
  static const std::string hgemv_cl_kernel_ =
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
  return hgemv_cl_kernel_;
}

const std::string &getHgemvClNoTransKernel() {
  static const std::string hgemv_cl_noTrans_kernel_ =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
        __kernel void sgemv_cl_noTrans_fp16(const __global half* A, const __global half* X,
                          __global half* Y, unsigned int N, unsigned int lda) {                                            
            unsigned int i;
            i = get_global_id(0);                         
            half y0 = 0.0f;
            for (unsigned int j = 0; j < N; j++)                         
                y0 += A[j + i * lda] * X[j]; 
            Y[i] = y0;                            
              
        })";
  return hgemv_cl_noTrans_kernel_;
}

const std::string &getDotClKernelFP16() {
  static const std::string dot_cl_kernel_fp16_ =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
        __kernel void dot_cl_fp16(const __global half* A, const __global half* X, unsigned int K, __global half* res) {
            *res = 0;
            for (unsigned int i = 0; i < K; i++){
                *res += A[i] * X[i];
            }
        })";
  return dot_cl_kernel_fp16_;
}

const std::string &getHgemmClNoTransKernel() {
  static const std::string hgemm_cl_noTrans_kernel_ =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
        __kernel void sgemm_cl_noTrans_fp16(const __global half* A, const __global half* B,
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
  return hgemm_cl_noTrans_kernel_;
}

const std::string &getHgemmClTransAKernel() {
  static const std::string hgemm_cl_transA_kernel_ =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
        __kernel void sgemm_cl_transA_fp16(const __global half* A, const __global half* B,
                          __global half* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
            
            unsigned int m = get_global_id(0);
            unsigned int n = get_global_id(1);
            half c = 0.0f;
            for (unsigned int k = 0; k < K; ++k) {
              half a, b;
              a = A[k * lda + m];
              b = B[k * ldb + n];
              c += a * b;
            }
            C[m * ldc + n] = c;
        })";
  return hgemm_cl_transA_kernel_;
}

const std::string &getHgemmClTransBKernel() {
  static const std::string hgemm_cl_transB_kernel_ =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
        __kernel void sgemm_cl_transB_fp16(const __global half* A, const __global half* B,
                          __global half* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
            
            unsigned int m = get_global_id(0);
            unsigned int n = get_global_id(1);
            half c = 0.0f;
            for (unsigned int k = 0; k < K; ++k) {
              half a, b;
              a = A[m * lda + k];
              b = B[n * ldb + k];
              c += a * b;
            }
            C[m * ldc + n] = c;
        })";
  return hgemm_cl_transB_kernel_;
}

const std::string &getHgemmClTransABKernel() {
  static const std::string hgemm_cl_transAB_kernel_ =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
        __kernel void sgemm_cl_transAB_fp16(const __global half* A, const __global half* B,
                          __global half* C, unsigned int K, unsigned int lda, unsigned int ldb, unsigned int ldc) {
            
            unsigned int m = get_global_id(0);
            unsigned int n = get_global_id(1);
            half c = 0.0f;
            for (unsigned int k = 0; k < K; ++k) {
              half a, b;
              a = A[k * lda + m];
              b = B[n * ldb + k];
              c += a * b;
            }
            C[m * ldc + n] = c;
        })";
  return hgemm_cl_transAB_kernel_;
}

const std::string &getAdditionClKernelFP16() {
  static const std::string addition_cl_kernel_fp16_ =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
        __kernel void addition_cl_fp16(const __global half* input, __global half* output, unsigned int size_input, unsigned int size_res) {
        size_t idx = get_global_id(0);
        if (idx < size_res) {
            output[idx] = output[idx] + input[idx % size_input];
        }
      })";
  return addition_cl_kernel_fp16_;
}

const std::string &getHscalClKernel() {
  static const std::string hscal_cl_kernel_ =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
        __kernel void sscal_cl_fp16(__global half* X, const float alpha) {
            
            unsigned int i = get_global_id(0);
            X[i] *= alpha;
        })";
  return hscal_cl_kernel_;
}

const std::string &getTransposeClAxis0KernelFP16() {
  static const std::string transpose_cl_kernel_fp16_axis0 =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
        __kernel void transpose_cl_fp16_axis0(__global const half* in, 
                                   __global half* output,
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
  return transpose_cl_kernel_fp16_axis0;
}

const std::string &getTransposeClAxis1KernelFP16() {
  static const std::string transpose_cl_kernel_fp16_axis1 =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
        __kernel void transpose_cl_fp16_axis1(__global const half* in, 
                                   __global half* output,
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
  return transpose_cl_kernel_fp16_axis1;
}

const std::string &getTransposeClAxis2KernelFP16() {
  static const std::string transpose_cl_kernel_fp16_axis2 =
    R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
        __kernel void transpose_cl_fp16_axis2(__global const half* in, 
                                   __global half* output,
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
  return transpose_cl_kernel_fp16_axis2;
}

const std::string &getSwiGluClKernelFP16() {
  static const std::string swiglu_cl_kernel_fp16_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void swiglu_cl_fp16(__global const half *in1, __global const half *in2, __global half *out) {
    int i = get_global_id(0);
    half swish = in1[i] * exp(in1[i]) / (1 + exp(in1[i]));
    out[i] = swish * in2[i];
})";
  return swiglu_cl_kernel_fp16_;
}

const std::string &getCopyClKernelFP16() {
  static const std::string copy_cl_kernel_fp16_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void copy_cl_fp16(__global const half* input, 
                               __global half* output,
                               const int batchsize, 
                               const int channels, 
                               const int height, 
                               const int width) {

    int i= get_global_id(0);
    output[i] = input[i];
    
})";
  return copy_cl_kernel_fp16_;
}

const std::string &getConcatClAxis3KernelFP16() {
  static const std::string concat_cl_axis3_kernel_fp16_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void concat_cl_axis3_fp16(__global const half* in1, 
                                           __global const half* in2, 
                                           __global half* out,
                                           const int batch_size, 
                                           const int channels, 
                                           const int height, 
                                           const int width1, 
                                           const int width2) {
    int global_id = get_global_id(0);
    
    int total_width = width1 + width2;
    
    int width = total_width;

    // 4D space coordinates
    int w = global_id % total_width;
    int h = (global_id / total_width) % height;
    int c = (global_id / (total_width * height)) % channels;
    int b = global_id / (total_width * height * channels);

    int output_index = ((b * channels + c) * height + h) * total_width + w;
    
    // Determining if the index is in in1 or in2
    if (w < width1) {
        // in1 index calculation
        int input1_index = ((b * channels + c) * height + h) * width1 + w;
        out[output_index] = in1[input1_index];
  
    } else {
        // in2 index calculation
        int input2_index = ((b * channels + c) * height + h) * width2 + (w - width1);
        out[output_index] = in2[input2_index];
    }
})";
  return concat_cl_axis3_kernel_fp16_;
}

const std::string &getConcatClAxis2KernelFP16() {
  static const std::string concat_cl_axis2_kernel_fp16_ =
    R"(__kernel void concat_cl_axis2_fp16(__global const half* in1,
                           __global const half* in2,
                           __global half* out,
                           const int batch_size,
                           const int channels,
                           const int height1,
                           const int height2,
                           const int width) {
  
  int total_height = height1 + height2;
  int global_id = get_global_id(0);
  
  // Calculate the coordinates in the 4D space
  int w = global_id % width;
  int h = (global_id / width) % total_height;
  int c = (global_id / (width * total_height)) % channels;
  int b = global_id / (width * total_height * channels);

  // Calculate the offset for the current batch, channel, and width in the output tensor
  int output_index = ((b * channels + c) * total_height + h) * width + w;

  if (h < height1) {
      // Index within input1
      int input1_index = ((b * channels + c) * height1 + h) * width + w;
      out[output_index] = in1[input1_index];
  } else {
      // Index within input2
      int input2_index = ((b * channels + c) * height2 + (h - height1)) * width + w;
      out[output_index] = in2[input2_index];
  }

})";
  return concat_cl_axis2_kernel_fp16_;
}

const std::string &getConcatClAxis1KernelFP16() {
  static const std::string concat_cl_axis1_kernel_fp16_ =
    R"(__kernel void concat_cl_axis1_fp16(__global const half* in1, 
                                           __global const half* in2, 
                                           __global half* out,
                                           const int batch_size, 
                                           const int channels1, 
                                           const int channels2, 
                                           const int height, 
                                           const int width) {
    int global_id = get_global_id(0);
    
    int total_channels = channels1 + channels2;

    // Calculate the coordinates in the 4D space
    int w = global_id % width;
    int h = (global_id / width) % height;
    int c = (global_id / (width * height)) % total_channels;
    int b = global_id / (width * height * total_channels);

    // Calculate the offset for the current batch, height, and width in the output tensor
    int output_index = ((b * total_channels + c) * height + h) * width + w;

    if (c < channels1) {
        // Index within input1
        int input1_index = ((b * channels1 + c) * height + h) * width + w;
        out[output_index] = in1[input1_index];
    } else {
        // Index within input2
        int input2_index = ((b * channels2 + (c - channels1)) * height + h) * width + w;
        out[output_index] = in2[input2_index];
    }
})";
  return concat_cl_axis1_kernel_fp16_;
}

const std::string &getRMSNormClKernelFP16() {
  static const std::string rmsnorm_cl_kernel_fp16_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void rmsnorm_cl_fp16(
    __global const half *input,  // Input tensor
    __global half *output,    // Output tensor
    __global const half *alpha,  // Alpha values (one for each width)
    half epsilon,
    int B,                  // Number of batches
    int C,                  // Number of channels
    int H,                  // Height of feature map
    int W                   // Width of feature map
) {
    int global_id = get_global_id(0);  // Get the global work item index

    // Compute the corresponding batch, height, and channel indices
    int n = global_id / C;       // Batch index
    int c = global_id % C;                    // Height index
    int h = get_global_id(1);                    // Channel index
    int index = ((n * C + c) * H + h) * W;

    // Calculate RMS norm for the current channel, height, and batch
    half sum_squares = 0.0f;
    for (int j = 0; j < W; ++j) {
        sum_squares += input[index+j] * input[index+j];
    }
    sum_squares /= W;
    half rms_norm = sqrt(sum_squares + epsilon);
    // Each work item processes all width elements for its specific n, h, c
    for (int w = 0; w < W; ++w) {
        output[index+w] = (input[index+w] / rms_norm) * alpha[w];
    } 
}
)";
  return rmsnorm_cl_kernel_fp16_;
}

#endif
} // namespace nntrainer
