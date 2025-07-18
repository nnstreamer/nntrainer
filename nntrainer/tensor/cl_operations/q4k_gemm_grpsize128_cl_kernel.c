#if 1
typedef float half;
typedef unsigned char uchar;
typedef unsigned int uint;
typedef long long uint4;
typedef struct {
  int s0, s1, s2;
} uint3;
#define convert_float(x) (x)
#define __kernel
#define __global
#define __local
#define get_group_id(x) (x)
#define get_local_id(x) (x)
#define CLK_LOCAL_MEM_FENCE
#endif

#define QK_K 256
#define TILE_H 4
#define TILE_W 32
#define WG_SIZE 128

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

__kernel void mm_q4Kx8_q8Kx4_grpsize128(const int n, __global float *restrict s,
                                     const int bs, // leading stride in s
                                     __global const block_q4_Kx8 *restrict vx,
                                     __global const block_q8_Kx4 *restrict vy,
                                     const int nr, const int nc) {
  const int tile_y = get_group_id(0);
  const int tile_x = get_group_id(1);
  const int grp_i = get_local_id(0);
  const int grp_y = grp_i / TILE_W; // 0-3  (row inside 4x32 tile)
  const int grp_x = grp_i % TILE_W; // 0-31 (col inside 4x32 tile)

  const int nb = n / QK_K; // #256-element blocks

  __local block_q4_Kx8 lB[4];
  __local block_q8_Kx4 lA;
  __local uint lutmp[4][32];

  float sumf = 0.0f;
  float sum_minf = 0.0f;
  
  for (int b = 0; b < nb; ++b) {
    // 1.  Copy one q8 block (A) and two q4 blocks (B0/B1) to LDS
    {
      __global const uchar *gB[4];
      gB[0] = (__global const uchar *)(vx + (tile_x * 4 + 0) * nb + b);
      gB[1] = (__global const uchar *)(vx + (tile_x * 4 + 1) * nb + b);
      gB[2] = (__global const uchar *)(vx + (tile_x * 4 + 2) * nb + b);
      gB[3] = (__global const uchar *)(vx + (tile_x * 4 + 3) * nb + b);
      __global const uchar *gA = (__global const uchar *)(vy + tile_y * nb + b);

      __local uchar *lBdst[4] = {(__local uchar *)&lB[0], (__local uchar *)&lB[1], (__local uchar *)&lB[2], (__local uchar *)&lB[3]};
      __local uchar *lAdst = (__local uchar *)&lA;

      const int vecsB = (int)(sizeof(block_q4_Kx8) / (16 * sizeof(uchar)));
      const int vecsA = (int)(sizeof(block_q8_Kx4) / (16 * sizeof(uchar)));

      for (int v = grp_i; v < vecsB; v += WG_SIZE) {
        vstore16(vload16(0, gB[0] + v * 16), 0, lBdst[0] + v * 16);
        vstore16(vload16(0, gB[1] + v * 16), 0, lBdst[1] + v * 16);
        vstore16(vload16(0, gB[2] + v * 16), 0, lBdst[2] + v * 16);
        vstore16(vload16(0, gB[3] + v * 16), 0, lBdst[3] + v * 16);
      }
      for (int v = grp_i; v < vecsA; v += WG_SIZE) {
        vstore16(vload16(0, gA + v * 16), 0, lAdst + v * 16);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2.  All 64 lanes build the two 8x4 LUTs
    for (int v = grp_i; v < WG_SIZE; v += WG_SIZE) {
      const int blk = v >> 5;       // 0, 1, 2, 3
      const int sb = (v & 0b11111) >> 2; // 0-7
      __local const uint *src =
        (__local const uint *)(&lB[blk].scales[0] + sb * 12);
      __local uint *dst = lutmp[blk] + sb * 4;

      dst[3] = ((src[2] >> 4) & KMASK2) | (((src[1] >> 6) & KMASK3) << 4);
      dst[2] = src[1] & KMASK1;
      dst[1] = (src[2] & KMASK2) | (((src[0] >> 6) & KMASK3) << 4);
      dst[0] = src[0] & KMASK1;
    }
    barrier(CLK_LOCAL_MEM_FENCE); // LUTs ready

    {
      // 3.  Each grp_i accumulates one C-tile element
      const int blk = grp_x >> 3;
      const int lj = grp_x & 7;
  
      const float dB = fp16_to_fp32(lB[blk].d[lj]);
      const float dB_min = fp16_to_fp32(lB[blk].dmin[lj]);
      const float dA = lA.d[grp_y];
  
      __local const uchar *lbytes = (const __local uchar *)lutmp[blk];
  
  #pragma unroll 16
      for (int k = 0; k < 16; ++k) {      // QK_K/(2*BLK_LEN)
        const int idxB = k * 64 + lj * 8; // 8 × 8 q4 block stride
        const int idxA = (k >> 2) * 256 + (k & 3) * 32 + grp_y * 8;
  
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
  #pragma unroll 8      
      for (int sb = 0; sb < 8; ++sb) {
        __local const uchar *mins = lbytes + 8 + sb * 16;
        __local const short *bsum = (__local const short *)&lA.bsums[0] + sb * 8 +
                                    grp_y * 4 - ((sb & 1) * 6);
  
        int macc = mins[lj] * (bsum[0] + bsum[1]);
        sum_minf += (float)macc * dB_min * dA;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // 5.  write one result element
  const int out_row = tile_y * TILE_H + grp_y;
  const int out_col = tile_x * TILE_W + grp_x;
  s[out_row * bs + out_col] = sumf - sum_minf;
}