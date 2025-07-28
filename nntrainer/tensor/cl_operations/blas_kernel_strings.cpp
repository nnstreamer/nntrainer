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

const std::string &getQ6KSgemvClKernel() {
  static const std::string q6_k_sgemv_cl_kernel_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    #define QK_K 256
    #define N_SIMDWIDTH 16
    #define N_SIMDGROUP 2
    #define N_DST 1
    #define BLOCK_STRIDE (N_SIMDWIDTH / 16)

    typedef char int8_t;
    typedef uchar uint8_t;
    typedef short int16_t;
    typedef ushort uint16_t;
    typedef int int32_t;
    typedef uint uint32_t;

    typedef struct {
        uint8_t ql[QK_K / 2];
        uint8_t qh[QK_K / 4];
        int8_t  scales[QK_K / 16];
        half d;
    } block_q6_K;

    kernel void kernel_mul_mv_q6_K_f32(
        global void * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
    ) {
        __local float reduction_buf[N_SIMDGROUP][N_SIMDWIDTH];

        src0 = (global void*)((global char*)src0 + offset0);
        src1 = (global float*)((global char*)src1 + offset1);
        dst = (global float*)((global char*)dst + offsetd);

        int nb = ne00 / QK_K;

        int r0 = get_group_id(0);
        int r1 = get_group_id(1);
        int im = get_group_id(2);
        int lid = get_local_id(0);
        int lsize = get_local_size(0);

        int row_group = lid / N_SIMDWIDTH;
        int lane = lid % N_SIMDWIDTH;
        int row = r0 * N_SIMDGROUP + row_group;

        int i12 = im % ne12;
        int i13 = im / ne12;

        ulong offset_src0 = (i12 / r2) * (nb * ne01) + (i13 / r3) * (nb * ne01 * ne02);

        global block_q6_K * x = (global block_q6_K *) src0 + row * nb + offset_src0;
        global float      * yy = (global float     *) src1 + r1 * ne10 + im * ne00 * ne1;

        uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;

        int tid  = lane / BLOCK_STRIDE;
        int ix   = lane % BLOCK_STRIDE;
        int ip   = tid / 8;
        int il   = tid % 8;
        int n    = 4;
        int l0   = n * il;
        int is   = 8 * ip + l0 / 16;

        int y_offset = 128 * ip + l0;
        int q_offset_l = 64 * ip + l0;
        int q_offset_h = 32 * ip + l0;

        float sumf = 0.0f;

        for (int i = ix; i < nb; i += BLOCK_STRIDE) {
            global uint8_t * q1 = x[i].ql + q_offset_l;
            global uint8_t * q2 = q1 + QK_K / 8;
            global uint8_t * qh = x[i].qh + q_offset_h;
            global int8_t  * sc = x[i].scales + is;
            global float   * y = yy + i * QK_K + y_offset;

            float dall = x[i].d;
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            for (int j = 0; j < 4; j++) {
                sums.s0 += y[j + 0]   * ((float)((q1[j] & 0xF) | ((qh[j] & kmask1) << 4)) - 32.f);
                sums.s1 += y[j + 32]  * ((float)((q2[j] & 0xF) | ((qh[j] & kmask2) << 2)) - 32.f);
                sums.s2 += y[j + 64]  * ((float)((q1[j] >> 4) | ((qh[j] & kmask3) >> 0)) - 32.f);
                sums.s3 += y[j + 96]  * ((float)((q2[j] >> 4) | ((qh[j] & kmask4) >> 2)) - 32.f);
            }

            sumf += dall * (sums.s0 * sc[0] + sums.s1 * sc[2] + sums.s2 * sc[4] + sums.s3 * sc[6]);
        }

        reduction_buf[row_group][lane] = sumf;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int offset = N_SIMDWIDTH / 2; offset > 0; offset >>= 1) {
            if (lane < offset) {
                reduction_buf[row_group][lane] += reduction_buf[row_group][lane + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lane == 0) {
            int global_row = r0 * N_SIMDGROUP + row_group;
            dst[r1 * ne0 + im * ne0 * ne1 + global_row] = reduction_buf[row_group][0];
        }
    }
    )";

  return q6_k_sgemv_cl_kernel_;
}

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
    R"(
    #define TS 16
    __kernel void sgemm_cl_noTrans(__global const float *A, __global const float *B,
                                   __global float *C, const int M, const int N,
                                   const int K) {
      const int globalRow = get_global_id(1); // M dimension
      const int globalCol = get_global_id(0); // N dimension

      __local float Asub[TS][TS];
      __local float Bsub[TS][TS];

      float sum = 0.0f;

      const int localRow = get_local_id(1);
      const int localCol = get_local_id(0);
      const int groupRow = TS * get_group_id(1);
      const int groupCol = TS * get_group_id(0);

      for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        const int tiledRowA = groupRow + localRow;
        const int tiledColA = t * TS + localCol;

        const int tiledRowB = t * TS + localRow;
        const int tiledColB = groupCol + localCol;

        // Load A
        if (tiledRowA < M && tiledColA < K)
          Asub[localRow][localCol] = A[tiledRowA * K + tiledColA];
        else
          Asub[localRow][localCol] = 0.0f;

        // Load B
        if (tiledRowB < K && tiledColB < N)
          Bsub[localRow][localCol] = B[tiledRowB * N + tiledColB];
        else
          Bsub[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
          sum += Asub[localRow][k] * Bsub[k][localCol];

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = sum;
    }
    )";
  return sgemm_cl_noTrans_kernel_;
}

const std::string &getSgemmClTransAKernel() {
  static const std::string sgemm_cl_transA_kernel_ =
    R"(
    #define TS 16
    __kernel void sgemm_cl_transA(__global const float *A, __global const float *B,
                                  __global float *C, const int M, const int N,
                                  const int K) {
      const int globalRow = get_global_id(1); // M
      const int globalCol = get_global_id(0); // N

      __local float Asub[TS][TS];
      __local float Bsub[TS][TS];

      float sum = 0.0f;

      const int localRow = get_local_id(1);
      const int localCol = get_local_id(0);
      const int groupRow = TS * get_group_id(1);
      const int groupCol = TS * get_group_id(0);

      for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        const int tiledRowA = t * TS + localCol;
        const int tiledColA = groupRow + localRow;

        if (tiledRowA < K && tiledColA < M)
          Asub[localRow][localCol] = A[tiledRowA * M + tiledColA];
        else
          Asub[localRow][localCol] = 0.0f;

        const int tiledRowB = t * TS + localRow;
        const int tiledColB = groupCol + localCol;

        if (tiledRowB < K && tiledColB < N)
          Bsub[localRow][localCol] = B[tiledRowB * N + tiledColB];
        else
          Bsub[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
          sum += Asub[localRow][k] * Bsub[k][localCol];

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = sum;
    }
    )";
  return sgemm_cl_transA_kernel_;
}

const std::string &getSgemmClTransBKernel() {
  static const std::string sgemm_cl_transB_kernel_ =
    R"(
    #define TS 16
    __kernel void sgemm_cl_transB(__global const float *A, __global const float *B,
                                  __global float *C, const int M, const int N,
                                  const int K) {
      const int globalRow = get_global_id(1);
      const int globalCol = get_global_id(0);

      __local float Asub[TS][TS];
      __local float Bsub[TS][TS];

      float sum = 0.0f;

      const int localRow = get_local_id(1);
      const int localCol = get_local_id(0);
      const int groupRow = TS * get_group_id(1);
      const int groupCol = TS * get_group_id(0);

      for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        const int tiledRowA = groupRow + localRow;
        const int tiledColA = t * TS + localCol;

        if (tiledRowA < M && tiledColA < K)
          Asub[localRow][localCol] = A[tiledRowA * K + tiledColA];
        else
          Asub[localRow][localCol] = 0.0f;

        const int tiledRowB = groupCol + localCol;
        const int tiledColB = t * TS + localRow;

        if (tiledRowB < N && tiledColB < K)
          Bsub[localRow][localCol] = B[tiledRowB * K + tiledColB];
        else
          Bsub[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
          sum += Asub[localRow][k] * Bsub[k][localCol];

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = sum;
    }
    )";
  return sgemm_cl_transB_kernel_;
}

const std::string &getSgemmClTransABKernel() {
  static const std::string sgemm_cl_transAB_kernel_ =
    R"(
    #define TS 16
    __kernel void sgemm_cl_transAB(__global const float *A, __global const float *B,
                                  __global float *C, const int M, const int N,
                                  const int K) {
      const int globalRow = get_global_id(1);
      const int globalCol = get_global_id(0);

      __local float Asub[TS][TS];
      __local float Bsub[TS][TS];

      float sum = 0.0f;

      const int localRow = get_local_id(1);
      const int localCol = get_local_id(0);
      const int groupRow = TS * get_group_id(1);
      const int groupCol = TS * get_group_id(0);

      for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        const int tiledRowA = t * TS + localCol;
        const int tiledColA = groupRow + localRow;

        if (tiledRowA < K && tiledColA < M)
          Asub[localRow][localCol] = A[tiledRowA * M + tiledColA];
        else
          Asub[localRow][localCol] = 0.0f;

        const int tiledRowB = groupCol + localCol;
        const int tiledColB = t * TS + localRow;

        if (tiledRowB < N && tiledColB < K)
          Bsub[localRow][localCol] = B[tiledRowB * K + tiledColB];
        else
          Bsub[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
          sum += Asub[localRow][k] * Bsub[k][localCol];

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = sum;
    }
    )";
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
    R"(
    __kernel void concat_cl_axis3(__global const float *input1,
                                  __global const float *input2, __global float *output,
                                  const int batch_size, const int channel_size,
                                  const int height_size, const int width1,
                                  const int width2) {
      // Get single global index
      const int global_idx = get_global_id(0);

      // Calculate total elements in one width concatenation
      const int total_elements = batch_size * channel_size * height_size;

      // Check if index is within bounds
      if (global_idx >= total_elements) {
        return;
      }

      // Calculate indices for batch, channel, and height
      const int batch_idx = global_idx / (channel_size * height_size);
      const int temp = global_idx % (channel_size * height_size);
      const int channel_idx = temp / height_size;
      const int height_idx = temp % height_size;

      // Calculate strides for input1
      const int stride_batch1 = channel_size * height_size * width1;
      const int stride_channel1 = height_size * width1;
      const int stride_height1 = width1;

      // Calculate strides for input2
      const int stride_batch2 = channel_size * height_size * width2;
      const int stride_channel2 = height_size * width2;
      const int stride_height2 = width2;

      // Calculate strides for output
      const int total_width = width1 + width2;
      const int stride_batch_out = channel_size * height_size * total_width;
      const int stride_channel_out = height_size * total_width;
      const int stride_height_out = total_width;

      // Calculate base indices
      const int base_idx1 = batch_idx * stride_batch1 +
                            channel_idx * stride_channel1 +
                            height_idx * stride_height1;

      const int base_idx2 = batch_idx * stride_batch2 +
                            channel_idx * stride_channel2 +
                            height_idx * stride_height2;

      const int base_idx_out = batch_idx * stride_batch_out +
                              channel_idx * stride_channel_out +
                              height_idx * stride_height_out;

      // Copy data from input1
      for (int w = 0; w < width1; w++) {
        output[base_idx_out + w] = input1[base_idx1 + w];
      }

      // Copy data from input2
      for (int w = 0; w < width2; w++) {
        output[base_idx_out + width1 + w] = input2[base_idx2 + w];
      }
    })";
  return concat_cl_axis3_kernel_;
}

const std::string &getConcatClAxis2Kernel() {
  static const std::string concat_cl_axis2_kernel_ =
    R"(
    __kernel void concat_cl_axis2(__global const float *input1,
                                  __global const float *input2,
                                  __global float *output, const int batch_size,
                                  const int channel_size, const int height1,
                                  const int height2, const int width_size) {
      // Get single global index
      const int global_idx = get_global_id(0);

      // Calculate total elements in one height concatenation
      const int total_elements = batch_size * channel_size * width_size;

      // Check if index is within bounds
      if (global_idx >= total_elements) {
        return;
      }

      // Calculate indices for batch, channel, and width
      const int batch_idx = global_idx / (channel_size * width_size);
      const int temp = global_idx % (channel_size * width_size);
      const int channel_idx = temp / width_size;
      const int width_idx = temp % width_size;

      // Calculate strides for input1
      const int stride_batch1 = channel_size * height1 * width_size;
      const int stride_channel1 = height1 * width_size;
      const int stride_height1 = width_size;

      // Calculate strides for input2
      const int stride_batch2 = channel_size * height2 * width_size;
      const int stride_channel2 = height2 * width_size;
      const int stride_height2 = width_size;

      // Calculate strides for output
      const int total_height = height1 + height2;
      const int stride_batch_out = channel_size * total_height * width_size;
      const int stride_channel_out = total_height * width_size;
      const int stride_height_out = width_size;

      // Calculate base indices
      const int base_idx1 =
        batch_idx * stride_batch1 + channel_idx * stride_channel1;

      const int base_idx2 =
        batch_idx * stride_batch2 + channel_idx * stride_channel2;

      const int base_idx_out =
        batch_idx * stride_batch_out + channel_idx * stride_channel_out;

      // Copy data from input1
      for (int h = 0; h < height1; h++) {
        output[base_idx_out + h * stride_height_out + width_idx] =
          input1[base_idx1 + h * stride_height1 + width_idx];
      }

      // Copy data from input2
      for (int h = 0; h < height2; h++) {
        output[base_idx_out + (height1 + h) * stride_height_out + width_idx] =
          input2[base_idx2 + h * stride_height2 + width_idx];
      }
})";
  return concat_cl_axis2_kernel_;
}

const std::string &getConcatClAxis1Kernel() {
  static const std::string concat_cl_axis1_kernel_ =
    R"(
    __kernel void concat_cl_axis1(__global const float *input1,
                                  __global const float *input2,
                                  __global float *output, const int batch_size,
                                  const int channel1, const int channel2,
                                  const int height_size, const int width_size) {
      // Get single global index
      const int global_idx = get_global_id(0);

      // Calculate total elements in one channel concatenation
      const int total_elements = batch_size * height_size * width_size;

      // Check if index is within bounds
      if (global_idx >= total_elements) {
        return;
      }

      // Calculate indices for batch, height, and width
      const int batch_idx = global_idx / (height_size * width_size);
      const int temp = global_idx % (height_size * width_size);
      const int height_idx = temp / width_size;
      const int width_idx = temp % width_size;

      // Calculate strides for input1
      const int stride_batch1 = channel1 * height_size * width_size;
      const int stride_channel1 = height_size * width_size;
      const int stride_height1 = width_size;

      // Calculate strides for input2
      const int stride_batch2 = channel2 * height_size * width_size;
      const int stride_channel2 = height_size * width_size;
      const int stride_height2 = width_size;

      // Calculate strides for output
      const int total_channels = channel1 + channel2;
      const int stride_batch_out = total_channels * height_size * width_size;
      const int stride_channel_out = height_size * width_size;
      const int stride_height_out = width_size;

      // Calculate base indices
      const int base_idx1 = batch_idx * stride_batch1;
      const int base_idx2 = batch_idx * stride_batch2;
      const int base_idx_out = batch_idx * stride_batch_out;

      // Calculate spatial offset
      const int spatial_offset = height_idx * stride_height_out + width_idx;

      // Copy data from input1
      for (int c = 0; c < channel1; c++) {
        output[base_idx_out + c * stride_channel_out + spatial_offset] =
          input1[base_idx1 + c * stride_channel1 + height_idx * stride_height1 +
                width_idx];
      }

      // Copy data from input2
      for (int c = 0; c < channel2; c++) {
        output[base_idx_out + (channel1 + c) * stride_channel_out +
              spatial_offset] = input2[base_idx2 + c * stride_channel2 +
                                        height_idx * stride_height2 + width_idx];
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

const std::string &getMulMatQ4_0Kernel() {
  static const std::string mul_mat_q4_0_kernel_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    #ifdef cl_intel_subgroups
    #pragma OPENCL EXTENSION cl_intel_subgroups : enable
    #else
    #pragma OPENCL EXTENSION cl_khr_subgroups : enable
    #endif

    #ifdef cl_intel_required_subgroup_size
    #pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
    #define INTEL_GPU 1
    #define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
    #define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
    #elif defined(cl_qcom_reqd_sub_group_size)
    #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
    #define ADRENO_GPU 1
    #define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
    #define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
    #endif

    #define QK4_0                   32
    #define QR4_0                   2
    #define QK_K                    256
    #define K_QUANTS_PER_ITERATION  2

    typedef char int8_t;
    typedef uchar uint8_t;
    typedef short int16_t;
    typedef ushort uint16_t;
    typedef int int32_t;
    typedef uint uint32_t;

    //------------------------------------------------------------------------------
    // block_q4_0
    //------------------------------------------------------------------------------
    struct block_q4_0
    {
        half d;
        uint8_t qs[QK4_0 / 2];
    };

    inline float mm_block_q_4_0_dot_y_flat(
            global uchar * x,
            global half  * dh,
            float sumy,
            float16 yl,
            int il
    ) {
        float           d   = *dh;
        global ushort * qs  = ((global ushort *)x + il/2);
        float           acc = 0.f;

        acc += yl.s0 * (qs[0] & 0x000F);
        acc += yl.s1 * (qs[0] & 0x0F00);
        acc += yl.s8 * (qs[0] & 0x00F0);
        acc += yl.s9 * (qs[0] & 0xF000);

        acc += yl.s2 * (qs[1] & 0x000F);
        acc += yl.s3 * (qs[1] & 0x0F00);
        acc += yl.sa * (qs[1] & 0x00F0);
        acc += yl.sb * (qs[1] & 0xF000);

        acc += yl.s4 * (qs[2] & 0x000F);
        acc += yl.s5 * (qs[2] & 0x0F00);
        acc += yl.sc * (qs[2] & 0x00F0);
        acc += yl.sd * (qs[2] & 0xF000);

        acc += yl.s6 * (qs[3] & 0x000F);
        acc += yl.s7 * (qs[3] & 0x0F00);
        acc += yl.se * (qs[3] & 0x00F0);
        acc += yl.sf * (qs[3] & 0xF000);

        return d * (sumy * -8.f + acc);
    }

    #ifdef INTEL_GPU
    #define N_DST 16 // each SIMD group works on 8 rows (in weights matrix)
    #define N_SIMDGROUP 1 // number of SIMD groups in a thread group
    #define N_SIMDWIDTH 16 // assuming SIMD group size is 16
    #elif defined (ADRENO_GPU)
    #define N_DST 16
    #define N_SIMDGROUP 1
    #define N_SIMDWIDTH 64
    #endif
    //
    // This variant performs 1d blocking with 16x output.
    // Eeach simdgroup outputs 16 values on `n0` dim (row in the output matrix).
    //
    inline void mul_mat_q_n_f32_1d_16x_flat(
            global uchar * src0_q,
            global half  * src0_d,
            global float * src1,
            global float * dst,
            int ne00,
            int ne01,
            int ne02,
            int ne10,
            int ne12,
            int ne0,
            int ne1,
            int r2,
            int r3
    ) {
        const int nb = ne00/QK4_0;

        int r0 = get_group_id(0);
        int r1 = get_group_id(1);
        int im = get_group_id(2);

        // (r0 * N_SIMDGROUP + get_sub_group_id()) is the linear global id of
        // a SIMD group in the grid. Each SIMD group produces N_DST values in the
        // result, hence uses nb blocks, i.e., the offset becomes first_row*nb.
        // Currently with llama2 7B, im is always 0.
        // TODO: how to handle im/gqa*(nb*ne0)?
        int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

        int i12 = im%ne12;
        int i13 = im/ne12;

        // The number of scales is the same as the number of blocks.
        ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
        // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
        ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * QK4_0/2;

        global uchar * x = (global uchar *) src0_q + offset0_q;
        global half  * d = (global half  *) src0_d + offset0_d;
        global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

        float16 yl;
        float16 sumf = (float16)(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

        int ix = get_sub_group_local_id()/2;
        int il = 8*(get_sub_group_local_id()%2);

        global float * yb = y + ix*QK4_0 + il;

        for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
            float sumy = 0.f;

            sumy += yb[0];
            sumy += yb[1];
            sumy += yb[2];
            sumy += yb[3];
            sumy += yb[4];
            sumy += yb[5];
            sumy += yb[6];
            sumy += yb[7];

            sumy += yb[16];
            sumy += yb[17];
            sumy += yb[18];
            sumy += yb[19];
            sumy += yb[20];
            sumy += yb[21];
            sumy += yb[22];
            sumy += yb[23];

            yl.s0 = yb[0];
            yl.s1 = yb[1]/256.f;

            yl.s2 = yb[2];
            yl.s3 = yb[3]/256.f;

            yl.s4 = yb[4];
            yl.s5 = yb[5]/256.f;

            yl.s6 = yb[6];
            yl.s7 = yb[7]/256.f;

            yl.s8 = yb[16]/16.f;
            yl.s9 = yb[17]/4096.f;

            yl.sa = yb[18]/16.f;
            yl.sb = yb[19]/4096.f;

            yl.sc = yb[20]/16.f;
            yl.sd = yb[21]/4096.f;

            yl.se = yb[22]/16.f;
            yl.sf = yb[23]/4096.f;

            sumf.s0 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  0*nb*QK4_0/2, d + ib +  0*nb, sumy, yl, il);
            sumf.s1 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  1*nb*QK4_0/2, d + ib +  1*nb, sumy, yl, il);
            sumf.s2 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  2*nb*QK4_0/2, d + ib +  2*nb, sumy, yl, il);
            sumf.s3 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  3*nb*QK4_0/2, d + ib +  3*nb, sumy, yl, il);

            sumf.s4 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  4*nb*QK4_0/2, d + ib +  4*nb, sumy, yl, il);
            sumf.s5 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  5*nb*QK4_0/2, d + ib +  5*nb, sumy, yl, il);
            sumf.s6 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  6*nb*QK4_0/2, d + ib +  6*nb, sumy, yl, il);
            sumf.s7 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  7*nb*QK4_0/2, d + ib +  7*nb, sumy, yl, il);

            sumf.s8 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  8*nb*QK4_0/2, d + ib +  8*nb, sumy, yl, il);
            sumf.s9 += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  9*nb*QK4_0/2, d + ib +  9*nb, sumy, yl, il);
            sumf.sa += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 10*nb*QK4_0/2, d + ib + 10*nb, sumy, yl, il);
            sumf.sb += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 11*nb*QK4_0/2, d + ib + 11*nb, sumy, yl, il);

            sumf.sc += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 12*nb*QK4_0/2, d + ib + 12*nb, sumy, yl, il);
            sumf.sd += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 13*nb*QK4_0/2, d + ib + 13*nb, sumy, yl, il);
            sumf.se += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 14*nb*QK4_0/2, d + ib + 14*nb, sumy, yl, il);
            sumf.sf += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 + 15*nb*QK4_0/2, d + ib + 15*nb, sumy, yl, il);

            yb += QK4_0 * (N_SIMDWIDTH/2);
        }

        float16 tot = (float16)(
            sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
            sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
            sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
            sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7),

            sub_group_reduce_add(sumf.s8), sub_group_reduce_add(sumf.s9),
            sub_group_reduce_add(sumf.sa), sub_group_reduce_add(sumf.sb),
            sub_group_reduce_add(sumf.sc), sub_group_reduce_add(sumf.sd),
            sub_group_reduce_add(sumf.se), sub_group_reduce_add(sumf.sf)
        );

        if (get_sub_group_local_id() == 0) {
            if (first_row + 0 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
            }
            if (first_row + 1 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
            }
            if (first_row + 2 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
            }
            if (first_row + 3 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
            }

            if (first_row + 4 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
            }
            if (first_row + 5 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
            }
            if (first_row + 6 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
            }
            if (first_row + 7 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
            }

            if (first_row + 8 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 8] = tot.s8;
            }
            if (first_row + 9 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 9] = tot.s9;
            }
            if (first_row + 10 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 10] = tot.sa;
            }
            if (first_row + 11 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 11] = tot.sb;
            }

            if (first_row + 12 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 12] = tot.sc;
            }
            if (first_row + 13 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 13] = tot.sd;
            }
            if (first_row + 14 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 14] = tot.se;
            }
            if (first_row + 15 < ne01) {
                dst[r1*ne0 + im*ne0*ne1 + first_row + 15] = tot.sf;
            }
        }
    }

    #ifdef INTEL_GPU
    REQD_SUBGROUP_SIZE_16
    #elif defined (ADRENO_GPU)
    REQD_SUBGROUP_SIZE_64
    #endif
    kernel void kernel_mul_mat_q4_0_f32_1d_16x_flat(
            global uchar * src0_q,
            global half  * src0_d,
            global float * src1,
            global float * dst,
            int ne00,
            int ne01,
            int ne02,
            int ne10,
            int ne12,
            int ne0,
            int ne1,
            int r2,
            int r3
    ) {
        mul_mat_q_n_f32_1d_16x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
    }
    )";
  return mul_mat_q4_0_kernel_;
}

const std::string &getConvertBlockQ4_0Kernel() {
  static const std::string convert_q4_0_block_kernel_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    #ifdef cl_intel_required_subgroup_size
    #pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
    #define INTEL_GPU 1
    #define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
    #define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
    #elif defined(cl_qcom_reqd_sub_group_size)
    #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
    #define ADRENO_GPU 1
    #define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
    #define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
    #endif

    #define QK4_0 32

    typedef uchar uint8_t;

    struct block_q4_0 {
      half d;
      uint8_t qs[QK4_0 / 2];
    };

    //------------------------------------------------------------------------------
    // kernel_convert_block_q4_0
    // Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
    // This kernel does not deshuffle the bits.
    // @todo: This kernel is not optimized for performance.
    //------------------------------------------------------------------------------
    kernel void kernel_convert_block_q4_0(global struct block_q4_0 *src0,
                                          global uchar *dst_q, global half *dst_d) {
      global struct block_q4_0 *b =
        (global struct block_q4_0 *)src0 + get_global_id(0);
      global uchar *q = (global uchar *)dst_q + QK4_0 / 2 * get_global_id(0);
      global half *d = (global half *)dst_d + get_global_id(0);

      *d = b->d;

      for (int i = 0; i < QK4_0 / 2; ++i) {
        q[i] = b->qs[i];
      }
    }
    )";
  return convert_q4_0_block_kernel_;
}

const std::string &getRestoreBlockQ4_0Kernel() {
  static const std::string restore_q4_0_block_kernel_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    #ifdef cl_intel_required_subgroup_size
    #pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
    #define INTEL_GPU 1
    #define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
    #define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
    #elif defined(cl_qcom_reqd_sub_group_size)
    #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
    #define ADRENO_GPU 1
    #define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
    #define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
    #endif

    #define QK4_0 32
    
    typedef uchar uint8_t;

    struct block_q4_0 {
      half d;
      uint8_t qs[QK4_0 / 2];
    };

    // @todo: This kernel is not optimized for performance.
    kernel void kernel_restore_block_q4_0(global uchar *src_q, global half *src_d,
                                          global struct block_q4_0 *dst) {
      global struct block_q4_0 *b =
        (global struct block_q4_0 *)dst + get_global_id(0);
      global uchar *q = (global uchar *)src_q + QK4_0 / 2 * get_global_id(0);
      global half *d = (global half *)src_d + get_global_id(0);

      b->d = *d;
      for (int i = 0; i < QK4_0 / 2; ++i) {
        b->qs[i] = q[i];
      }
    }
    )";

  return restore_q4_0_block_kernel_;
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
            float y0 = 0.0f;
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
            float y0 = 0.0f;
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
            float y = 0.0f;
            for (unsigned int i = 0; i < K; i++){
                y += A[i] * X[i];
            }
            *res = y;
        })";
  return dot_cl_kernel_fp16_;
}

const std::string &getHgemmClNoTransKernel() {
  static const std::string hgemm_cl_noTrans_kernel_ =
    R"(

    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #define TS 16
    __kernel void sgemm_cl_noTrans_fp16(__global const half *A,
                                        __global const half *B, __global half *C,
                                        const int M, const int N, const int K) {
      const int globalRow = get_global_id(1); // M dimension
      const int globalCol = get_global_id(0); // N dimension

      const int localRow = get_local_id(1);
      const int localCol = get_local_id(0);
      const int groupRow = TS * get_group_id(1);
      const int groupCol = TS * get_group_id(0);

      __local half Asub[TS][TS];
      __local half Bsub[TS][TS];

      float sum = 0.0f;

      for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        const int tiledRowA = groupRow + localRow;
        const int tiledColA = t * TS + localCol;

        const int tiledRowB = t * TS + localRow;
        const int tiledColB = groupCol + localCol;

        // Load A
        if (tiledRowA < M && tiledColA < K)
          Asub[localRow][localCol] = A[tiledRowA * K + tiledColA];
        else
          Asub[localRow][localCol] = (half)0.0h;

        // Load B
        if (tiledRowB < K && tiledColB < N)
          Bsub[localRow][localCol] = B[tiledRowB * N + tiledColB];
        else
          Bsub[localRow][localCol] = (half)0.0h;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
          sum += (float)(Asub[localRow][k]) * (float)(Bsub[k][localCol]);

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = (half)(sum);
    }
    )";
  return hgemm_cl_noTrans_kernel_;
}

const std::string &getHgemmClTransAKernel() {
  static const std::string hgemm_cl_transA_kernel_ =
    R"(
      #pragma OPENCL EXTENSION cl_khr_fp16 : enable
      #define TS 16
      __kernel void sgemm_cl_transA_fp16(__global const half *A,
                                      __global const half *B, __global half *C,
                                      const int M, const int N, const int K) {
      const int globalRow = get_global_id(1); // M
      const int globalCol = get_global_id(0); // N

      const int localRow = get_local_id(1);
      const int localCol = get_local_id(0);
      const int groupRow = TS * get_group_id(1);
      const int groupCol = TS * get_group_id(0);

      __local half Asub[TS][TS];
      __local half Bsub[TS][TS];

      float sum = 0.0f;

      for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        const int tiledRowA = t * TS + localCol;
        const int tiledColA = groupRow + localRow;

        const int tiledRowB = t * TS + localRow;
        const int tiledColB = groupCol + localCol;

        // Load Aᵗ (A[col * M + row])
        if (tiledRowA < K && tiledColA < M)
          Asub[localRow][localCol] = A[tiledRowA * M + tiledColA];
        else
          Asub[localRow][localCol] = (half)0.0h;

        // Load B (K x N)
        if (tiledRowB < K && tiledColB < N)
          Bsub[localRow][localCol] = B[tiledRowB * N + tiledColB];
        else
          Bsub[localRow][localCol] = (half)0.0h;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
          sum += (float)(Asub[localRow][k]) * (float)(Bsub[k][localCol]);

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = (half)(sum);
    }
    )";
  return hgemm_cl_transA_kernel_;
}

const std::string &getHgemmClTransBKernel() {
  static const std::string hgemm_cl_transB_kernel_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #define TS 16
    __kernel void sgemm_cl_transB_fp16(__global const half *A,
                                      __global const half *B, __global half *C,
                                      const int M, const int N, const int K) {
      const int globalRow = get_global_id(1); // M
      const int globalCol = get_global_id(0); // N

      const int localRow = get_local_id(1);
      const int localCol = get_local_id(0);
      const int groupRow = TS * get_group_id(1);
      const int groupCol = TS * get_group_id(0);

      __local half Asub[TS][TS];
      __local half Bsub[TS][TS];

      float sum = 0.0f;

      for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        const int tiledRowA = groupRow + localRow;
        const int tiledColA = t * TS + localCol;

        const int tiledRowB = groupCol + localCol;
        const int tiledColB = t * TS + localRow;

        // Load A (M x K)
        if (tiledRowA < M && tiledColA < K)
          Asub[localRow][localCol] = A[tiledRowA * K + tiledColA];
        else
          Asub[localRow][localCol] = (half)0.0h;

        // Load Bᵗ (B[col * K + row])
        if (tiledRowB < N && tiledColB < K)
          Bsub[localRow][localCol] = B[tiledRowB * K + tiledColB];
        else
          Bsub[localRow][localCol] = (half)0.0h;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
          sum += (float)(Asub[localRow][k]) * (float)(Bsub[k][localCol]);

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = (half)(sum);
    }
    )";
  return hgemm_cl_transB_kernel_;
}

const std::string &getHgemmClTransABKernel() {
  static const std::string hgemm_cl_transAB_kernel_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #define TS 16
    __kernel void sgemm_cl_transAB_fp16(__global const half *A,
                                        __global const half *B, __global half *C,
                                        const int M, const int N, const int K) {
      const int globalRow = get_global_id(1); // M
      const int globalCol = get_global_id(0); // N

      const int localRow = get_local_id(1);
      const int localCol = get_local_id(0);
      const int groupRow = TS * get_group_id(1);
      const int groupCol = TS * get_group_id(0);

      __local half Asub[TS][TS];
      __local half Bsub[TS][TS];

      float sum = 0.0f;

      for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        const int tiledRowA = t * TS + localCol;
        const int tiledColA = groupRow + localRow;

        const int tiledRowB = groupCol + localCol;
        const int tiledColB = t * TS + localRow;

        // Load Aᵗ (K x M)
        if (tiledRowA < K && tiledColA < M)
          Asub[localRow][localCol] = A[tiledRowA * M + tiledColA];
        else
          Asub[localRow][localCol] = (half)0.0h;

        // Load Bᵗ (N x K)
        if (tiledRowB < N && tiledColB < K)
          Bsub[localRow][localCol] = B[tiledRowB * K + tiledColB];
        else
          Bsub[localRow][localCol] = (half)0.0h;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k)
          sum += (float)(Asub[localRow][k]) * (float)(Bsub[k][localCol]);

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (globalRow < M && globalCol < N)
        C[globalRow * N + globalCol] = (half)(sum);
    }

    )";
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
    half swish = in1[i] * exp((float)in1[i]) / (1 + exp((float)in1[i]));
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
    __kernel void concat_cl_axis3_fp16(__global const half *input1,
                              __global const half *input2, __global half *output,
                              const int batch_size, const int channel_size,
                              const int height_size, const int width1,
                              const int width2) {
      // Get single global index
      const int global_idx = get_global_id(0);

      // Calculate total elements in one width concatenation
      const int total_elements = batch_size * channel_size * height_size;

      // Check if index is within bounds
      if (global_idx >= total_elements) {
        return;
      }

      // Calculate indices for batch, channel, and height
      const int batch_idx = global_idx / (channel_size * height_size);
      const int temp = global_idx % (channel_size * height_size);
      const int channel_idx = temp / height_size;
      const int height_idx = temp % height_size;

      // Calculate strides for input1
      const int stride_batch1 = channel_size * height_size * width1;
      const int stride_channel1 = height_size * width1;
      const int stride_height1 = width1;

      // Calculate strides for input2
      const int stride_batch2 = channel_size * height_size * width2;
      const int stride_channel2 = height_size * width2;
      const int stride_height2 = width2;

      // Calculate strides for output
      const int total_width = width1 + width2;
      const int stride_batch_out = channel_size * height_size * total_width;
      const int stride_channel_out = height_size * total_width;
      const int stride_height_out = total_width;

      // Calculate base indices
      const int base_idx1 = batch_idx * stride_batch1 +
                            channel_idx * stride_channel1 +
                            height_idx * stride_height1;

      const int base_idx2 = batch_idx * stride_batch2 +
                            channel_idx * stride_channel2 +
                            height_idx * stride_height2;

      const int base_idx_out = batch_idx * stride_batch_out +
                              channel_idx * stride_channel_out +
                              height_idx * stride_height_out;

      // Copy data from input1
      for (int w = 0; w < width1; w++) {
        output[base_idx_out + w] = input1[base_idx1 + w];
      }

      // Copy data from input2
      for (int w = 0; w < width2; w++) {
        output[base_idx_out + width1 + w] = input2[base_idx2 + w];
      }
    })";
  return concat_cl_axis3_kernel_fp16_;
}

const std::string &getConcatClAxis2KernelFP16() {
  static const std::string concat_cl_axis2_kernel_fp16_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void concat_cl_axis2_fp16(__global const half *input1,
                                __global const half *input2, __global half *output,
                                const int batch_size, const int channel_size,
                                const int height1, const int height2,
                                const int width_size) {
      // Get single global index
      const int global_idx = get_global_id(0);

      // Calculate total elements in one height concatenation
      const int total_elements = batch_size * channel_size * width_size;

      // Check if index is within bounds
      if (global_idx >= total_elements) {
        return;
      }

      // Calculate indices for batch, channel, and width
      const int batch_idx = global_idx / (channel_size * width_size);
      const int temp = global_idx % (channel_size * width_size);
      const int channel_idx = temp / width_size;
      const int width_idx = temp % width_size;

      // Calculate strides for input1
      const int stride_batch1 = channel_size * height1 * width_size;
      const int stride_channel1 = height1 * width_size;
      const int stride_height1 = width_size;

      // Calculate strides for input2
      const int stride_batch2 = channel_size * height2 * width_size;
      const int stride_channel2 = height2 * width_size;
      const int stride_height2 = width_size;

      // Calculate strides for output
      const int total_height = height1 + height2;
      const int stride_batch_out = channel_size * total_height * width_size;
      const int stride_channel_out = total_height * width_size;
      const int stride_height_out = width_size;

      // Calculate base indices
      const int base_idx1 =
        batch_idx * stride_batch1 + channel_idx * stride_channel1;

      const int base_idx2 =
        batch_idx * stride_batch2 + channel_idx * stride_channel2;

      const int base_idx_out =
        batch_idx * stride_batch_out + channel_idx * stride_channel_out;

      // Copy data from input1
      for (int h = 0; h < height1; h++) {
        output[base_idx_out + h * stride_height_out + width_idx] =
          input1[base_idx1 + h * stride_height1 + width_idx];
      }

      // Copy data from input2
      for (int h = 0; h < height2; h++) {
        output[base_idx_out + (height1 + h) * stride_height_out + width_idx] =
          input2[base_idx2 + h * stride_height2 + width_idx];
      }
  })";
  return concat_cl_axis2_kernel_fp16_;
}

const std::string &getConcatClAxis1KernelFP16() {
  static const std::string concat_cl_axis1_kernel_fp16_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void concat_cl_axis1_fp16(__global const half *input1,
                                      __global const half *input2,
                                      __global half *output, const int batch_size,
                                      const int channel1, const int channel2,
                                      const int height_size,
                                      const int width_size) {
      // Get single global index
      const int global_idx = get_global_id(0);

      // Calculate total elements in one channel concatenation
      const int total_elements = batch_size * height_size * width_size;

      // Check if index is within bounds
      if (global_idx >= total_elements) {
        return;
      }

      // Calculate indices for batch, height, and width
      const int batch_idx = global_idx / (height_size * width_size);
      const int temp = global_idx % (height_size * width_size);
      const int height_idx = temp / width_size;
      const int width_idx = temp % width_size;

      // Calculate strides for input1
      const int stride_batch1 = channel1 * height_size * width_size;
      const int stride_channel1 = height_size * width_size;
      const int stride_height1 = width_size;

      // Calculate strides for input2
      const int stride_batch2 = channel2 * height_size * width_size;
      const int stride_channel2 = height_size * width_size;
      const int stride_height2 = width_size;

      // Calculate strides for output
      const int total_channels = channel1 + channel2;
      const int stride_batch_out = total_channels * height_size * width_size;
      const int stride_channel_out = height_size * width_size;
      const int stride_height_out = width_size;

      // Calculate base indices
      const int base_idx1 = batch_idx * stride_batch1;
      const int base_idx2 = batch_idx * stride_batch2;
      const int base_idx_out = batch_idx * stride_batch_out;

      // Calculate spatial offset
      const int spatial_offset = height_idx * stride_height_out + width_idx;

      // Copy data from input1
      for (int c = 0; c < channel1; c++) {
        output[base_idx_out + c * stride_channel_out + spatial_offset] =
          input1[base_idx1 + c * stride_channel1 + height_idx * stride_height1 +
                width_idx];
      }

      // Copy data from input2
      for (int c = 0; c < channel2; c++) {
        output[base_idx_out + (channel1 + c) * stride_channel_out +
              spatial_offset] = input2[base_idx2 + c * stride_channel2 +
                                        height_idx * stride_height2 + width_idx];
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
    half rms_norm = sqrt((float)(sum_squares + epsilon));
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
