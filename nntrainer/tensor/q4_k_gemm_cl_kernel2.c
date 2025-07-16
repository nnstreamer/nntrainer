#if 1
typedef float half;
typedef unsigned char uchar;
typedef unsigned int uint;
typedef long long uint4;
#define convert_float(x) (x)
#define __kernel
#define __global
#define __local
#define get_group_id(x) (x)
#define get_local_id(x) (x)
#define CLK_LOCAL_MEM_FENCE
#endif

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

    __kernel void mat_mul_q4_K_8x8_q8_K2(const int n, __global float *restrict s,
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

      __global const block_q4_Kx8 *restrict gB0_ = vx + (tile_x * 2 + 0) * nb;
      __global const block_q4_Kx8 *restrict gB1_ = vx + (tile_x * 2 + 1) * nb;
      __global const block_q4_Kx8 *restrict gA_ = vy + tile_y * nb;

      __local uint *restrict lBdst0 = (__local uint *)&lB[0];
      __local uint *restrict lBdst1 = (__local uint *)&lB[1];
      __local uint *restrict lAdst = (__local uint *)&lA;

      for (int b = 0; b < nb; ++b) {
        // 1.  Copy one q8 block (A) and two q4 blocks (B0/B1) to LDS
        {
          __global const uint *gB0 = (__global const uint *)(gB0_ + b);
          __global const uint *gB1 = (__global const uint *)(gB1_ + b);
          __global const uint *gA = (__global const uint *)(gA_ + b);

          const int vecsB = (int)(sizeof(block_q4_Kx8) / (16 * 4));
          const int vecsA = (int)(sizeof(block_q8_Kx4) / (16 * 4));

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