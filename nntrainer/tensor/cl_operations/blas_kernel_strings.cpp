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
            // dst[r1 * ne0 + im * ne0 * ne1 + row] = reduction_buf[row_group][0];
        }
    }
    )";

  return q6_k_sgemv_cl_kernel_;
}

const std::string &getQ6KQ81SgemvClKernel() {
  static const std::string q6_k_q8_1_sgemv_cl_kernel_ =
    R"(
      typedef char int8_t;
      typedef uchar uint8_t;
      typedef short int16_t;
      typedef ushort uint16_t;
      typedef int int32_t;
      typedef uint uint32_t;

      #define QK_K 256
      
      #define WARP_SIZE 32

      #define VDR_Q6_K_Q8_1_MMQ 8
      
      #define QR6_K 2
      #define QI6_K (QK_K / (4 * QR6_K))

      #define QK8_1 32
      #define QR8_1 1
      #define QI8_1 (QK8_1 / (4 * QR8_1))
      
      #define MMQ_TILE_Y_K (WARP_SIZE * QR6_K)

      typedef struct
      {
        int qs;
        int dm;
        int sc;
      } tile_x_sizes;

      // NOTE(m.wlasiuk) : https://github.com/ggml-org/llama.cpp/blob/c46503014db0d63fa7b1b28c58adfb51054e2dec/ggml/src/ggml-cuda/mmq.cuh#L158
      tile_x_sizes mmq_get_dp4a_tile_x_sizes_q6k(int mmq_y)
      {
        return (tile_x_sizes)
        {
          mmq_y * WARP_SIZE * 2 + mmq_y,
          mmq_y * WARP_SIZE / QI6_K + mmq_y / QI6_K,
          mmq_y * WARP_SIZE / 8 + mmq_y / 8
        };
      } 

      // NOTE(m.wlasiuk) : https://github.com/ggml-org/llama.cpp/blob/c46503014db0d63fa7b1b28c58adfb51054e2dec/ggml/src/ggml-cuda/vecdotq.cuh#L501
      int ggml_cuda_dp4a(const int a, const int b, int sum)
      {
        char4 a_vec = *(char*)&a;
        char4 b_vec = *(char*)&b;

        return sum + a_vec.s0 * b_vec.s0
                   + a_vec.s1 * b_vec.s1
                   + a_vec.s2 * b_vec.s2
                   + a_vec.s3 * b_vec.s3;
      }

      // NOTE(m.wlasiuk) : https://github.com/ggml-org/llama.cpp/blob/c46503014db0d63fa7b1b28c58adfb51054e2dec/ggml/src/ggml-cuda/vecdotq.cuh#L501
      float vec_dot_q6_K_q8_1_impl_mmq(
        const __global int * restrict v,
        const __global int * restrict u,
        const global int8_t * restrict sc,
        const float d6,
        const __global float* restrict d8)
      {
        float sumf_d = 0.0f;

        const int      sc_packed = *((const __global int *)sc);
        const int8_t * sc_reg = (const int8_t *)&sc_packed;

        for(int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 +=4)
        {
          int2 sumi_d = (int2)(0, 0);

          for(int i = i0; i < i0 + 2; i++)
          {
            sumi_d.x = ggml_cuda_dp4a(v[2 * i + 0], u[2 * i + 0], sumi_d.x);
            sumi_d.x = ggml_cuda_dp4a(v[2 * i + 1], u[2 * i + 1], sumi_d.x);

            sumi_d.y = ggml_cuda_dp4a(v[2 * i + 4], u[2 * i + 4], sumi_d.y);
            sumi_d.y = ggml_cuda_dp4a(v[2 * i + 5], u[2 * i + 5], sumi_d.y);
          }

          sumf_d += d8[i0/4] * (sc_reg[i0 / 2 + 0] * sumi_d.x + sc_reg[i0 / 2 + 1] * sumi_d.y);
        }

        return d6 * sumf_d;
      }

      // NOTE(m.wlasiuk) : https://github.com/ggml-org/llama.cpp/blob/c46503014db0d63fa7b1b28c58adfb51054e2dec/ggml/src/ggml-cuda/mmq.cuh#L1696
      // vec_dot_q6_K_q8_1_dp4a
      __kernel void kernel_mul_mv_q6_K_q8_1(
        const __global int * restrict x,
        const __global int * restrict y,
        __global float * restrict sum,
        const int k00,
        const int mmq_x,
        const int mmq_y,
        const int nwarps)
      {
        const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes_q6k(mmq_y);

        const __global int   * x_qs = x;
        const __global float * x_df = (const __global float*)(x_qs + txs.qs);
        const __global int   * x_sc = (const __global int*)(x_df + txs.dm);
        const __global int   * y_qs = y + 4;
        const __global float * y_df = (const __global float*)y;

        const int tid_x = get_local_id(0);
        const int tid_y = get_local_id(1);

        for(int k01 = 0; k01 < WARP_SIZE; k01 += QR6_K * VDR_Q6_K_Q8_1_MMQ)
        {
          const int k0 = k00 + k01;

          for(int j0 = 0; j0 < mmq_x; j0 += nwarps)
          {
            const int j = j0 + tid_y;

            for(int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE)
            {
              const int i = i0 + tid_x;

              const __global int8_t * sc = (const __global int8_t*)&x_sc[i * (WARP_SIZE / 8) + i / 8 + k0 / 16];

              sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE] += vec_dot_q6_K_q8_1_impl_mmq(
                &x_qs[i * (QR6_K * WARP_SIZE + 1) + k0],
                &y_qs[j * MMQ_TILE_Y_K + k01],
                sc,
                x_df[i * (WARP_SIZE / QI6_K) + i / QI6_K],
                &y_df[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
          }
        }
      }

      // Unpack data and run tile
      // TODO : mul_mat_q_process_tile
      // src/ggml-cuda/mmq.cuh L2522

      // Run all tiles based on input matrices dimensions
      // TODO : mul_mat_q <== make it kernel
      // src/ggml-cuda/mmq.cuh L2606

    )";
  return q6_k_q8_1_sgemv_cl_kernel_;
}

// NOTE(m.wlasiuk) : skip
const std::string &getQ4KQ81SgemvClKernel() {
  static const std::string q4_k_q8_1_sgemv_cl_kernel_ =
    R"(
    // PLACEHOLDER
      __kernel void kernel_mul_mv_q4_K_q8_1()
      {
      }

    //  #define WARP_SIZE 32
    //  #define QR4_K 2
    //  #define VDR_Q4_K_Q8_1_MMQ 2
    //  #define QI8_1 8
    //  #define QI4_K 32
    //  #define QK8_1 32
    //  #define MMQ_ITER_K 32
    //  #define MMQ_TILE_Y_K (WARP_SIZE * QR4_K)
    //  #define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
    //
    //  typedef struct
    //  {
    //    int qs;
    //    int dm;
    //    int sc;
    //  } tile_x_sizes;
    //
    //  tile_x_sizes mmq_get_dp4a_tile_x_sizes_q4k(int mmq_y)
    //  {
    //    return (tile_x_sizes)
    //    {
    //      mmq_y * WARP_SIZE + mmq_y,
    //      mmq_y * WARP_SIZE / QI4_K,
    //      mmq_y * WARP_SIZE / 8 + mmq_y / 8
    //    };
    //  }
    //
    //  int ggml_cuda_dp4a(const int a, const int b, int c)
    //  {
    //    char4 va = *(char4*)&a;
    //    char4 vb = *(char4*)&b;
    //
    //    return c + (va.s0 * vb.s0) + (va.s1 * vb.s1) + (va.s2 * vb.s2) + (va.s3 * vb.s3);
    //  }
    //
    //  float vec_dot_q4_K_q8_1_impl_mmq(
    //    const __global int * restrict v,
    //    const __global int * restrict u,
    //    const __global uint8_t * restrict sc,
    //    const __global uint8_t * restrict m,
    //    const half2 dm4,
    //    const __global half2 * restrict ds8)
    //  {
    //    float sumf_d = 0.0f;
    //    float sumf_m = 0.0f;
    //
    //    for(int i = 0; i < QR4_K * VDR_Q4_K_Q8_1_MMQ / QI8_1; i++)
    //    {
    //      int sumi_d = 0;
    //
    //      for(int j = 0; j < QI8_1; j++)
    //      {
    //        int v_val = (v[j] >> (4 * i)) & 0x0F0F0F0F;
    //        int u_val = v[i * QI8_1 + j];
    //
    //        sumi_d = ggml_cuda_dp4a(v_val, u_val, sumi_d);
    //      }
    //
    //      float2 ds8f = convert_float2(ds8[i]);
    //
    //      sumf_d += ds8f.x * (sc[i] * sumi_d);
    //      sumf_m += ds8f.y * m[i];
    //    }
    //
    //    float2 dm4f = convert_float2(dm4);
    //
    //    return dm4f.x * sumf_d - dm4f.y * sumf_m;
    //  } 
    //
    //  void vec_dot_q4_K_q8_1_dp4a(
    //    const __global int * restrict x,
    //    const __global int * restrict y,
    //    __global float* restrict sum,
    //    const int k00,
    //    const int mmq_x,
    //    const int mmq_y,
    //    const int nwarps)
    //  {                                            
    //    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes_q4k(mmq_y);
    //
    //    const __global int * x_qs = x;
    //    const __global half2 * x_dm = (const __global half2*)(x_qs + txs.qs);
    //    const __global int * x_sc = (const __global int*)(x_dm + txs.dm);
    //    const __global int * y_qs = y + 4;
    //    const __global half2* y_ds = (const __global half2*)y;
    //
    //    int tid_x = get_local_id(0);
    //    int tid_y = get_local_id(1);
    //
    //    int warp_id = tid_y;
    //
    //    for(int k01 = 0; k01 < WARP_SIZE; k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ)
    //    {
    //      const int k0 = k00 + k01;
    //
    //      for(int j0 = 0; j0 < mmq_x; j0 += nwarps)
    //      {
    //        const j = j0 + warp_id;
    //        
    //        if(j >= mmq_x)
    //        {
    //          constinue;
    //        }
    //      
    //        for(int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE)
    //        {
    //          if(i >= mmq_y)
    //          {
    //            constinue;
    //          }
    //          
    //          const __global uint8_t* sc = (const __global uint8_t*)&x_sc[i * (WARP_SIZE / 8) + i / 8 + k0 / 32] + 2 * (k01 / 16);
    //
    //          float result = vec_dot_q4_K_q8_1_impl_mmq(
    //            &x_qs[i * (WARP_SIZE + 1) + k0 / 2],
    //            &y_qs[j * MMQ_TILE_Y_K + k01],
    //            sc,
    //            sc + 8,
    //            x_dm[i],
    //            &y_ds[i * MMQ_TILE_Y_K + k01 / QI8_1]);
    //
    //          atomic_add(&sum[j0 / nwarps * mmq_y / WARP_SIZE + i0 / WARP_SIZE], result);
    //        }
    //      }
    //    }
    //  }
    //
    //  void load_tiles_q4_K(
    //    const __global char* restrict x,
    //    __local int* restrict x_tile,
    //    const int kbx0,
    //    const int i_max,
    //    const int stride,
    //    const int mmq_y,
    //    const int nwarps,
    //    const int need_check)
    //  {
    //  }
    //
    //  void mul_mat_q_process_tile(
    //    const __global_ char* restrict x,
    //    const int offset_x,
    //    const __global int* restrict y,
    //    const __global int* restrict ids_dst,
    //    __global_ float* restrict dst,
    //    __global float* restrict tmp_fixup,
    //    const int stride_row_x,
    //    const int ncols_y,
    //    const int stride_col_dst,
    //    const int tile_x_max_i,
    //    const int tile_y_max_j,
    //    const int kb0_start,
    //    const int kb0_stop,
    //    const int mmq_x,
    //    const int nwarps,
    //    const int need_check,
    //    const int fixup,
    //    __local int* data_mul_mat_q)
    //  {
    //    const int qk = QK4_K;
    //    const int mmq_y = 64; // NOTE : idk?
    //
    //    __local int* tile_y = data_mul_mat_q + mmq_x;
    //    __local int* tile_x = tile_y + GGML_PAD(mmq_x * (WARP_SIZE + WARP_SIZE / QI8_1), nwarps * WARP_SIZE);
    //
    //    float sum[mmq_x * mmq_y / nwarps * WARP_SIZE] = {0.0f};
    //
    //    const int blocks_per_iter = MMQ_ITER_K / qk;
    //    
    //    const int tid_x = get_local_id(0);
    //    const int tid_y = get_local_id(1);
    //
    //    for(int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter)
    //    {
    //      load_tiles_q4_K(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
    //
    //      {
    //        const __global int * by0 = y + ncols_y * (kb0 * (qk*sizeof(block_q8_1_mmq) / (4 * QK8_1*sizeof(int))));
    //        
    //        for(int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * WARP_SIZE)
    //        {
    //          int l = l0 + tid_y * WARP_SIZE + tid_x;
    //          if(!need_check || l < mmq_x * MMQ_TILE_Y_K)
    //          {
    //            tile_y[l] = by0[l];
    //          }
    //        }
    //      }
    //
    //      barrier(CLK_LOCAL_MEM_FENCE);
    //      vec_dot_q4_K_q8_1_dp4a(tile_x, tile_y, sum, 0, mmq_x, mmq_y, nwarps);
    //      barrier(CLK_LOCAL_MEM_FENCE);
    //      
    //      {
    //        const __global int * by0 = y + ncols_y * (kb0 * (qk * sizeof(block_q8_1_mmq) / (4 * QK8_1 * sizeof(int)) + sizeof(block_q8_1_mmq) / sizeof(int)));
    //
    //        for(int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * WARP_SIZE)
    //        {
    //          int l = l0 + tid_y * WARP_SIZE + tid_x;
    //          if(!need_check || l < mmq_x * MMQ_TILE_Y_K)
    //          {
    //            tile_y[l] = by0[l]
    //          }
    //        }
    //      }
    //
    //      barrier(CLK_LOCAL_MEM_FENCE);
    //      vec_dot_q4_K_q8_1_dp4a(tile_x, tile_y, sum, WARP_SIZE, mmq_x, mmq_y, nwarps);
    //      barrier(CLK_LOCAL_MEM_FENCE);
    //    }
    //
    //    if(fixup)
    //    {
    //      mmq_write_back_dp4a(sum, ids_dst, tmp_fixup + get_group_id(0) * (mmq_x * mmq_y), mmq_y, mmq_y, mmq_x);
    //    }
    //    else
    //    {
    //      mmq_write_back_dp4a(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    //    }
    //  }
    //
    //  // kernel mul_mat_q .....
    //)";
  return q4_k_q8_1_sgemv_cl_kernel_;
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
