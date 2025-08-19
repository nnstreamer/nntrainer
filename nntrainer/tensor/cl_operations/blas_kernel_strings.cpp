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

const std::string &getQ4_0_Ab_Bi_8x4_Kernel() {
  static const std::string q4_0_mul_mat_Ab_Bi_8x4_kernel_ =
    R"(
    // src0_q, src0_d, src1 are transposed as a preprocessing step
    // 4-bit weights are transposed in groups of 4 (unsigned short int)
    // consider weights originally "next to each other", now "on top of each other"
    // each fiber computes a 8x4 tile of output elements
    // using unshuffled weights

    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    #ifdef cl_intel_required_subgroup_size
    #pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
    #define INTEL_GPU 1
    #define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
    #elif defined(cl_qcom_reqd_sub_group_size)
    #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
    #define ADRENO_GPU 1
    #define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
    #endif

    #ifdef INTEL_GPU
    REQD_SUBGROUP_SIZE_16
    #elif defined(ADRENO_GPU)
    REQD_SUBGROUP_SIZE_128
    #endif
    kernel void kernel_mul_mat_Ab_Bi_8x4(global const ushort *src0_q, // quantized A
                                        global const half *src0_d,   // A scales
                                        global half4 *src1, // B (1d image)
                                        global float *dst,  // C
                                        int m,              // M
                                        int n,              // N with padding
                                        int k,              // K
                                        int n_no_padding    // N without padding
    ) {

      int m_4 = m >> 2;
      int n_4 = n >> 2;

      int gy = get_global_id(0);
      int gx = get_global_id(1);
      int gx_2 = gx << 2;

      half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
      half8 B;                              // registers for activations
      half4 dequantized_weights;            // registers for dequantized weights
      __global const ushort *weight_ptr = src0_q + gx_2; // pointer for weights
      __global const half *scale_ptr = src0_d + gx_2;    // pointer for scales

      for (int i = 0; i < k; i += 4) { // loop through K dimension

        B.s0123 = src1[gy * 2 + (i) * (n_4)];
        B.s4567 = src1[gy * 2 + (i) * (n_4) + 1];

        // keep (i/4) and (i/32) in parenthesis, rounds down
        // load 4 consecutive groups of 4 weights
        ushort4 bits4 = vload4(
          0, weight_ptr + (i / 4) * (m)); // (i/4) because weights grouped in 4s

        // load 4 consecutive scales
        half4 scale = vload4(
          0, scale_ptr + (i / 32) * (m)); // (i/32) because 1 scale per 32 elements

        // j=0
        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) *
                                scale.s0; // dequantize a row of the 16 weights
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
        c0 +=
          B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B.s0123 = src1[gy * 2 + (i + 1) * (n_4)];
        B.s4567 = src1[gy * 2 + (i + 1) * (n_4) + 1];
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) *
                                scale.s0; // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
        c0 +=
          B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B.s0123 = src1[gy * 2 + (i + 2) * (n_4)];
        B.s4567 = src1[gy * 2 + (i + 2) * (n_4) + 1];
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) *
                                scale.s0; // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
        c0 +=
          B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B.s0123 = src1[gy * 2 + (i + 3) * (n_4)];
        B.s4567 = src1[gy * 2 + (i + 3) * (n_4) + 1];
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) *
                                scale.s0; // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
        c0 +=
          B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;
      }

      int idx = (gy << 3) * m + (gx << 2); // vectorized store 16 elements

      // conditional check if store is to a valid location. Required when N is not a
      // multiple of 8 if statements allow registers to be reused for each store
      // provides a performance boost due to reduced register footprint, which
      // increases number of concurrent waves
      if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
      }
      if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
      }
      if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
      }
      if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
      }
      if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
      }
      if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
      }
      if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
      }
      if (idx + 3 < m * n_no_padding) {
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
      }
    }

    )";

  return q4_0_mul_mat_Ab_Bi_8x4_kernel_;
}

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
    R"(
    __kernel void swiglu_cl(
      __global const float * restrict in1,
      __global const float * restrict in2,
      __global       float * restrict out)
    {
      const int i = get_global_id(0);

      const float in1_val = in1[i];
      const float in2_val = in2[i];

      const float in1_exp = exp(in1_val);
      
      const float swish = in1_val * in1_exp / (1.0f + in1_exp);

      out[i] = swish * in2_val;
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
    R"(
    #ifdef cl_intel_required_subgroup_size
    #pragma OPENCL EXTENSION cl_intel_subgroups : enable
    #pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
    #define INTEL_GPU 1
    #define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
    #elif defined(cl_qcom_reqd_sub_group_size)
    #pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
    #define ADRENO_GPU 1
    #define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
    #endif
    
    #ifdef INTEL_GPU
    REQD_SUBGROUP_SIZE_32
    #elif defined(ADRENO_GPU)
    REQD_SUBGROUP_SIZE_64
    #endif
  __kernel void rmsnorm_cl(
    __global const float *input,  // Input tensor
    __global float *output,    // Output tensor
    __global const float *alpha,  // Alpha values (one for each width)
    float epsilon,
    int H,                  // Height of feature map
    int W                   // Width of feature map
) {
    // Compute the corresponding batch, height, and channel indices
    int h = get_group_id(0);
    int index = h * W;
    // Calculate RMS norm for the current channel, height, and batch
    __global const float4 *in = (__global const float4*)(input + index);
    float4 sum_squares_4 = 0.0f;
    for (int i = get_local_id(0); i < W / 4; i += get_local_size(0)) {
        sum_squares_4 += in[i] * in[i];
    }

    float sum_squares = sum_squares_4.x + sum_squares_4.y + sum_squares_4.z + sum_squares_4.w;
    sum_squares = sub_group_reduce_add(sum_squares);

    const float mean  = sum_squares / W;
    const float scale = 1.0f / sqrt(mean + epsilon);

    __global float4 *out = (__global float4*)(output + index);
    __global const float4 *a = (__global const float4*)(alpha);
    for (int i = get_local_id(0); i < W / 4; i += get_local_size(0)) {
        out[i] = in[i] * scale * a[i];
    }
}
)";

  return rmsnorm_cl_kernel_;
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
    // kernel_convert_block_q4_0_noshuffle
    // Flatten q4_0 weights and unshuffle the bits
    //------------------------------------------------------------------------------

    kernel void kernel_convert_block_q4_0_noshuffle(
        global struct block_q4_0 * src0,
        global uchar * dst_q,
        global half  * dst_d
    ) {
        global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
        global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
        global half  * d = (global half *) dst_d + get_global_id(0);

        *d = b->d;
        for (int i = 0; i < QK4_0/4; ++i) {
            uchar x0 = b->qs[2*i + 0];
            uchar x1 = b->qs[2*i + 1];

            q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
            q[i + QK4_0/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

    #ifdef ADRENO_GPU
            // Workaround for adreno - must have the following printf statement for
            // the kernel to work properly. Otherwise it produces incorrect result.
            // convert_uchar above also seems necessary.
            // Compare against a large number so that it does not print anything.
            // get_sub_group_local_id() also works.
            if (get_global_id(0) == 65536*4096) {
                printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
            }
    #endif
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

const std::string &getTranspose16BitKernel() {
  static const std::string transpose_16_kernel_ =
    R"(
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    // 16-bit transpose, loading/storing a 4x4 tile of elements (via buffers)
    kernel void kernel_transpose_16(
        __global const half4* input,   // was image1d_buffer_t
        __global       half4* output,  // was image1d_buffer_t
        const uint rows,               // = get_global_size(1)
        const uint cols                // = get_global_size(0)
    ) {
        const uint i   = get_global_id(0);
        const uint j   = get_global_id(1);
        const uint i_2 = i << 2;   // 4 * i
        const uint j_2 = j << 2;   // 4 * j

        // Load four consecutive rows (each element is half4)
        const half4 temp0 = input[(j_2 + 0) * cols + i];
        const half4 temp1 = input[(j_2 + 1) * cols + i];
        const half4 temp2 = input[(j_2 + 2) * cols + i];
        const half4 temp3 = input[(j_2 + 3) * cols + i];

        // Write transposed 4x4 tile (each write is a half4 column)
        output[(i_2 + 0) * rows + j] = (half4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0);
        output[(i_2 + 1) * rows + j] = (half4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1);
        output[(i_2 + 2) * rows + j] = (half4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2);
        output[(i_2 + 3) * rows + j] = (half4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3);
    }
  )";

  return transpose_16_kernel_;
}

const std::string &getTranspose32Bit16BitKernel() {
  static const std::string transpose_32_16_kernel_ =
    R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    // 32-bit transpose, loading/storing a 4x4 tile of elements
    // Only used for activations
    // converts to FP16
    // also adds zero padding for non multiple of 8 prompt lengths
    kernel void kernel_transpose_32_16(
      __global const float4 *input, // FP32 source
      __global half4 *output,       // FP16 destination
      const uint rows,       // original "rows" in tiles (height/4 of the source)
      const uint cols,       // original "cols" in tiles (width  of the source)
      const uint padded_rows // destination rows after padding
    ) {
      const uint i = get_global_id(0); // tile x
      const uint j = get_global_id(1); // tile y
      const uint i_2 = i << 2;         // i * 4
      const uint j_2 = j << 2;         // j * 4

      half4 t0 = (half4)(0.0h, 0.0h, 0.0h, 0.0h);
      half4 t1 = t0, t2 = t0, t3 = t0;

      const uint total = rows * cols * 16;

      // Read 4 rows from the same column i, converting FP32 -> FP16
      uint idx;

      idx = (j_2 + 0) * cols + i;
      if (idx < total)
        t0 = convert_half4(input[idx]);

      idx = (j_2 + 1) * cols + i;
      if (idx < total)
        t1 = convert_half4(input[idx]);

      idx = (j_2 + 2) * cols + i;
      if (idx < total)
        t2 = convert_half4(input[idx]);

      idx = (j_2 + 3) * cols + i;
      if (idx < total)
        t3 = convert_half4(input[idx]);

      output[(i_2 + 0) * padded_rows + j] = (half4)(t0.s0, t1.s0, t2.s0, t3.s0);
      output[(i_2 + 1) * padded_rows + j] = (half4)(t0.s1, t1.s1, t2.s1, t3.s1);
      output[(i_2 + 2) * padded_rows + j] = (half4)(t0.s2, t1.s2, t2.s2, t3.s2);
      output[(i_2 + 3) * padded_rows + j] = (half4)(t0.s3, t1.s3, t2.s3, t3.s3);
    }
  )";

  return transpose_32_16_kernel_;
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
      const int i = get_global_id(0);

      const half in1_val = in1[i];
      const half in2_val = in2[i];

      const half in1_exp = exp(in1_val);
      
      const half half_one = (half)(1.0f);
      const half swish = in1_val * in1_exp / (half_one + in1_exp);

      out[i] = swish * in2_val;    
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
