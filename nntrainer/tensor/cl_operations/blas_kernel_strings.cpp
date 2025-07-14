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

const std::string &getQ4KGemmClKernel() {
  static const std::string q4_k_gemm_cl_kernel =
    R"(
    #define QK_K 256
    #define NCOL_I 16
    #define BLK_LEN 8
    #define WG_SIZE 64

    #define KMASK1 0x3f3f3f3fu
    #define KMASK2 0x0f0f0f0fu
    #define KMASK3 0x03030303u

    #ifdef cl_khr_fp16
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #endif

    typedef struct {
      half d[8];
      half dmin[8];
      uchar scales[96];
      uchar qs[1024];
    } block_q4_Kx8;

    typedef struct {
      float d[4];
      char qs[QK_K * 4];
      short bsums[QK_K / 4];
    } block_q8_Kx4;

    /// @note convert_float() does not work for Nvidia GPU
    inline float fp16_to_fp32(half h) { return convert_float(h); }
    #define REDUCE_ADD_SHORT8(v)                                                   \
      ((v).s0 + (v).s1 + (v).s2 + (v).s3 + (v).s4 + (v).s5 + (v).s6 + (v).s7)

    __kernel void mat_mul_q4_K_8x8_q8_K(const int n, __global float *restrict s,
                                        const int bs, // leading stride in s
                                        __global const block_q4_Kx8 *restrict vx,
                                        __global const block_q8_Kx4 *restrict vy,
                                        const int nr, const int nc) {
      const int tile_y = get_group_id(0);
      const int tile_x = get_group_id(1);
      const int lane = get_local_id(0);

      const int lane_m = lane / NCOL_I; // 0-3  (row inside 4x16 tile)
      const int lane_j = lane % NCOL_I; // 0-15 (col inside 4x16 tile)

      const int nb = n / QK_K; // #256-element blocks

      __local block_q4_Kx8 lB[2];
      __local block_q8_Kx4 lA;
      __local uint lutmp[2][32];

      float sumf = 0.0f;
      float sum_minf = 0.0f;

      for (int b = 0; b < nb; ++b) {
        // 1.  Copy one q8 block (A) and two q4 blocks (B0/B1) to LDS
        {
          __global const uchar *gB0 =
            (__global const uchar *)(vx + (tile_x * 2 + 0) * nb + b);
          __global const uchar *gB1 =
            (__global const uchar *)(vx + (tile_x * 2 + 1) * nb + b);
          __global const uchar *gA = (__global const uchar *)(vy + tile_y * nb + b);

          __local uchar *lBdst0 = (__local uchar *)&lB[0];
          __local uchar *lBdst1 = (__local uchar *)&lB[1];
          __local uchar *lAdst = (__local uchar *)&lA;

          const int vecsB = (int)(sizeof(block_q4_Kx8) / 16);
          const int vecsA = (int)(sizeof(block_q8_Kx4) / 16);

          for (int v = lane; v < vecsB; v += WG_SIZE) {
            vstore16(vload16(0, gB0 + v * 16), 0, lBdst0 + v * 16);
            vstore16(vload16(0, gB1 + v * 16), 0, lBdst1 + v * 16);
          }
          for (int v = lane; v < vecsA; v += WG_SIZE) {
            vstore16(vload16(0, gA + v * 16), 0, lAdst + v * 16);
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2.  All 64 lanes build the two 8x4 LUTs
        for (int v = lane; v < 128; v += WG_SIZE) {
          const int blk = v >> 6;       // 0 | 1
          const int sb = (v & 63) >> 3; // 0-7
          __local const uchar *src =
            (__local const uchar *)&lB[blk].scales[0] + sb * 12;
          __local uint *dst = lutmp[blk] + sb * 4;

          uint4 tmp4 = vload4(0, (const __local uint *)src);
          vstore4(tmp4, 0, dst);

          dst[3] = ((dst[2] >> 4) & KMASK2) | (((dst[1] >> 6) & KMASK3) << 4);
          uint t = dst[1] & KMASK1;
          dst[1] = (dst[2] & KMASK2) | (((dst[0] >> 6) & KMASK3) << 4);
          dst[2] = t;
          dst[0] &= KMASK1;
        }
        barrier(CLK_LOCAL_MEM_FENCE); // LUTs ready

        // 3.  Each lane accumulates one C-tile element
        const int blk = lane_j >> 3;
        const int lj = lane_j & 7;

        const float dB = fp16_to_fp32(lB[blk].d[lj]);
        const float dB_min = fp16_to_fp32(lB[blk].dmin[lj]);
        const float dA = lA.d[lane_m];

        __local const uchar *lbytes = (const __local uchar *)lutmp[blk];

    #pragma unroll 16
        for (int k = 0; k < 16; ++k) {      // QK_K/(2*BLK_LEN)
          const int idxB = k * 64 + lj * 8; // 8 × 8 q4 block stride
          const int idxA = (k >> 2) * 256 + (k & 3) * 32 + lane_m * 8;

          const int sc0 = lbytes[(k / 4) * 32 + lj];
          const int sc1 = lbytes[(k / 4) * 32 + 16 + lj];

          uchar8 q = vload8(0, lB[blk].qs + idxB);
          char8 a0 = vload8(0, lA.qs + idxA);
          char8 a1 = vload8(0, lA.qs + idxA + 128);

          uchar8 qlo = q & (uchar8)(0x0F); // low nibbles
          uchar8 qhi = q >> (uchar8)4;     // high nibbles

          short8 prod0 = convert_short8(qlo) * convert_short8(a0);
          short8 prod1 = convert_short8(qhi) * convert_short8(a1);

          int sumi =
            REDUCE_ADD_SHORT8(prod0) * sc0 + REDUCE_ADD_SHORT8(prod1) * sc1;

          sumf += (float)sumi * dB * dA;
        }

        // 4.  bias / min-d correction
        for (int sb = 0; sb < 8; ++sb) {
          __local const uchar *mins = lbytes + 8 + sb * 16;
          __local const short *bsum = (__local const short *)&lA.bsums[0] + sb * 8 +
                                      lane_m * 4 - ((sb & 1) * 6);

          int macc = mins[lj] * (bsum[0] + bsum[1]);
          sum_minf += (float)macc * dB_min * dA;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
      }

      // 5.  write one result element
      const int out_row = tile_y * 4 + lane_m;
      const int out_col = tile_x * NCOL_I + lane_j;
      s[out_row * bs + out_col] = sumf - sum_minf;
    }
  )";

  return q4_k_gemm_cl_kernel;
}

const std::string &getQ6KSgemvClKernel() {
  static const std::string q6_k_sgemv_cl_kernel_ =
    R"(
// ---------------------------------------------------------------
//  kernel_mul_mv_q6_K_f32_opt  –  with optional sub-group prefetch
// ---------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable
/* Prefetch is enabled only if the device says so */
#ifdef cl_intel_subgroup_buffer_prefetch
#pragma OPENCL EXTENSION cl_intel_subgroup_buffer_prefetch : enable
#endif

#define QK_K 256
#define SGW 16      /* sub-group width                */
#define SG_PER_WG 2 /* 2 sub-groups → 32 WI per WG    */
#define N_DST 8     /* rows produced per sub-group    */

typedef uchar uint8_t;
typedef char int8_t;

typedef struct __attribute__((packed)) {
  uint8_t ql[QK_K / 2];
  uint8_t qh[QK_K / 4];
  int8_t scales[QK_K / 16];
  half d;
} block_q6_K;

/* 2-bit masks reused in unpack */
static constant uint8_t KMASK1 = 0x03;
static constant uint8_t KMASK2 = 0x0C;
static constant uint8_t KMASK3 = 0x30;
static constant uint8_t KMASK4 = 0xC0;

__attribute__((intel_reqd_sub_group_size(SGW)))
__attribute__((reqd_work_group_size(SGW * SG_PER_WG, 1, 1))) kernel void
kernel_mul_mv_q6_K_f32(global void *restrict src0, ulong offset0,
                       global float *restrict src1, ulong offset1,
                       global float *restrict dst, ulong offsetd, int ne00,
                       int ne01, int ne02, int ne10, int ne12, int ne0, int ne1,
                       int r2, int r3) {
  /* ------ adjust pointers from host side-byte offsets ------------------ */
  src0 = (global void *)((global char *)src0 + offset0);
  src1 = (global float *)((global char *)src1 + offset1);
  dst = (global float *)((global char *)dst + offsetd);

  const int nb = ne00 / QK_K; /* blocks per row */

  /* ----------- work-group / sub-group bookkeeping ---------------------- */
  const int gid_x = get_group_id(0);
  const int gid_y = get_group_id(1);
  const int gid_z = get_group_id(2);

  const int sg_id = get_sub_group_id();      /* 0 or 1 in this WG */
  const int lane = get_sub_group_local_id(); /* 0 … 15            */

  const int first_row = (gid_x * SG_PER_WG + sg_id) * N_DST;

  /* ---------------- batch / head offsets into src0 --------------------- */
  const int i12 = gid_z % ne12;
  const int i13 = gid_z / ne12;
  const ulong base_off = (ulong)(i12 / r2) * (ulong)(nb * ne01) +
                         (ulong)(i13 / r3) * (ulong)(nb * ne01 * ne02);

  /* ---------------- lane-constant byte offsets ------------------------- */
  const int ip = lane >> 3; /* 0 / 1               */
  const int il = lane & 7;  /* 0 … 7               */
  const int l0 = il * 4;    /* 0,4,8, … 28         */

  const int scale_idx = ip * 8 + (l0 >> 4); /* 0 … 15              */
  const int y_off = 128 * ip + l0;
  const int ql_off = 64 * ip + l0;
  const int qh_off = 32 * ip + l0;

  /* ---------------- base pointer to the y-vector ----------------------- */
  global float *y_base = src1 + gid_y * ne10 + gid_z * ne00 * ne1 + y_off;

  float sum[N_DST] = {0.0f};

  /* ============================= block loop ============================ */
  for (int ib = 0; ib < nb; ++ib) {

    global float *yptr = y_base + ib * QK_K;

/* ---------- software prefetch of the NEXT block’s y (if enabled) -- */
#ifdef cl_intel_subgroup_buffer_prefetch
    if (__builtin_expect(ib + 1 < nb, 1)) {
      global uint *next = (global uint *)(yptr + QK_K); /* +256 floats */
      /* each ui8 == 8 uints  == 32 bytes per lane */
      intel_sub_group_block_prefetch_ui8(next + 0);  /* +0  B */
      intel_sub_group_block_prefetch_ui8(next + 8);  /* +32 B */
      intel_sub_group_block_prefetch_ui8(next + 16); /* +64 B */
      intel_sub_group_block_prefetch_ui8(next + 24); /* +96 B */
    }
#endif

    /* --------- 16 lane-private y scalars ------------------------------ */
    float y0 = yptr[0], y1 = yptr[32], y2 = yptr[64], y3 = yptr[96];
    float y4 = yptr[1], y5 = yptr[33], y6 = yptr[65], y7 = yptr[97];
    float y8 = yptr[2], y9 = yptr[34], yA = yptr[66], yB = yptr[98];
    float yC = yptr[3], yD = yptr[35], yE = yptr[67], yF = yptr[99];

/* ------------- process N_DST destination rows -------------------- */
#pragma unroll
    for (int dst_idx = 0; dst_idx < N_DST; ++dst_idx) {

      const int row = first_row + dst_idx;
      if (row >= ne0)
        break;

      global block_q6_K *blk =
        (global block_q6_K *)src0 + row * nb + base_off + ib;

      uchar4 q1v = vload4(0, (global uchar *)(blk->ql + ql_off));
      uchar4 q2v = vload4(0, (global uchar *)(blk->ql + ql_off + QK_K / 8));
      uchar4 qhv = vload4(0, (global uchar *)(blk->qh + qh_off));

      const uchar *q1 = (const uchar *)&q1v;
      const uchar *q2 = (const uchar *)&q2v;
      const uchar *qh = (const uchar *)&qhv;

      const float s0 = (float)blk->scales[scale_idx + 0];
      const float s1 = (float)blk->scales[scale_idx + 2];
      const float s2 = (float)blk->scales[scale_idx + 4];
      const float s3 = (float)blk->scales[scale_idx + 6];
      const float d = (float)blk->d;

#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint8_t q1j = q1[j], q2j = q2[j], qhj = qh[j];

        const float v0 = (float)((q1j & 0x0F) | ((qhj & KMASK1) << 4)) - 32.f;
        const float v1 = (float)((q2j & 0x0F) | ((qhj & KMASK2) << 2)) - 32.f;
        const float v2 = (float)((q1j >> 4) | ((qhj & KMASK3) >> 0)) - 32.f;
        const float v3 = (float)((q2j >> 4) | ((qhj & KMASK4) >> 2)) - 32.f;

        float yy0, yy1, yy2, yy3;
        switch (j) {
        case 0:
          yy0 = y0;
          yy1 = y1;
          yy2 = y2;
          yy3 = y3;
          break;
        case 1:
          yy0 = y4;
          yy1 = y5;
          yy2 = y6;
          yy3 = y7;
          break;
        case 2:
          yy0 = y8;
          yy1 = y9;
          yy2 = yA;
          yy3 = yB;
          break;
        default:
          yy0 = yC;
          yy1 = yD;
          yy2 = yE;
          yy3 = yF;
          break;
        }

        sum[dst_idx] +=
          d * (yy0 * v0 * s0 + yy1 * v1 * s1 + yy2 * v2 * s2 + yy3 * v3 * s3);
      }
    }
  }

/* ------------------- reduce inside sub-group + store ----------------- */
#pragma unroll
  for (int k = 0; k < N_DST; ++k) {
    float total = sub_group_reduce_add(sum[k]);
    if (lane == 0) {
      int row = first_row + k;
      if (row < ne0)
        dst[gid_y * ne0 + gid_z * ne0 * ne1 + row] = total;
    }
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
