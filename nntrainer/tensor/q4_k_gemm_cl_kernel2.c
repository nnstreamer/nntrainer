#if 1
typedef float half;
typedef unsigned char uchar;
typedef unsigned int uint;
typedef long long uint4;
typedef struct {int s0, s1, s2;} uint3;
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

    __kernel void mat_mul_q4_K_8x8_q8_K2(const int n, float *restrict s,
                                        const int bs, // leading stride in s
                                        const block_q4_Kx8 *restrict vx,
                                        const block_q8_Kx4 *restrict vy,
                                        const int nr, const int nc) {
      const int tile_y = get_group_id(0);
      const int tile_x = get_group_id(1);
      const int lane = get_local_id(0);

      const int grp_y = lane / NCOL_I; // 0-3  (row inside 4x16 tile)
      const int grp_x = lane % NCOL_I; // 0-15 (col inside 4x16 tile)

      const int nb = n / QK_K; // #256-element blocks

      __local block_q4_Kx8 lB[2];
      __local block_q8_Kx4 lA;
      __local uint lutmp[2][32];

      float sumf = 0.0f;
      float sum_minf = 0.0f;

      const block_q4_Kx8 *restrict gB[2];
      gB[0] = vx + (tile_x * 2 + 0) * nb;
      gB[1] = vx + (tile_x * 2 + 1) * nb;
      const block_q4_Kx8 *restrict gA_ = vy + tile_y * nb;

      uint *restrict lBdst0 = (uint *)&lB[0];
      uint *restrict lBdst1 = (uint *)&lB[1];
      uint *restrict lAdst = (uint *)&lA;

      for (int b = 0; b < nb; ++b) {
        // 1.  Copy one q8 block (A) and two q4 blocks (B0/B1) to LDS
        {
          const uint *gB0 = (const uint *)(gB[0] + b);
          const uint *gB1 = (const uint *)(gB[1] + b);
          const uint *gA = (const uint *)(gA_ + b);

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
        if (lane < 16) {
          const int blk = lane >> 3;       // 0 | 1
          const int sb = lane && 0b111; // 0-7
          const uchar *src = (const uchar *)&lB[blk].scales[0] + sb * 12;
          uint *dst = lutmp[blk] + sb * 4;

          uint3 tmp3 = vload3(0, (const __local uint *)src);          

          dst[3] = ((tmp3.s2 >> 4) & KMASK2) | (((tmp3.s1 >> 6) & KMASK3) << 4);
          dst[2] = tmp3.s1 & KMASK1;
          dst[1] = (tmp3.s2 & KMASK2) | (((tmp3.s0 >> 6) & KMASK3) << 4);
          dst[0] = tmp3.s0 & KMASK1;
        }
        barrier(CLK_LOCAL_MEM_FENCE); // LUTs ready

        // 3.  Each lane accumulates one C-tile element
        {
          const int blk = grp_x >> 3;
          const int lj = grp_x & 0b111;
  
          const float dB = fp16_to_fp32(lB[blk].d[lj]);
          const float dB_min = fp16_to_fp32(lB[blk].dmin[lj]);
          const float dA = lA.d[grp_y];
  
          const uchar *dequant_lut = (const uchar *)lutmp[blk];
  
          #pragma unroll 16
          for (int k = 0; k < 16; ++k) {      // QK_K/(2*BLK_LEN)
            const int idxB = (8 * k + lj) * 8; // 8 × 8 q4 block stride
            const int idxA = (4 * k + grp_y) * 8;
  
            const int sc0 = dequant_lut[(k / 4) * 32 + lj];
            const int sc1 = dequant_lut[(k / 4) * 32 + 16 + lj];
  
            uchar8 q = vload8(0, lB[blk].qs + idxB);
            char8 a0 = vload8(0, lA.qs + idxA);
            char8 a1 = vload8(0, lA.qs + idxA + 128);
  
            uchar8 qlo = q & 0x0F; // low nibbles
            uchar8 qhi = q >> 4;     // high nibbles
  
            short8 prod0 = convert_short8(qlo) * convert_short8(a0);
            short8 prod1 = convert_short8(qhi) * convert_short8(a1);
  
            int sumi = REDUCE_ADD_SHORT8(prod0) * sc0 + REDUCE_ADD_SHORT8(prod1) * sc1;
  
            sumf += (float)sumi * dB * dA;
          }
          // 4.  bias / min-d correction
          for (int sb = 0; sb < 8; ++sb) {
            const uchar *mins = dequant_lut + 8 + sb * 16;
            const short *bsum = (const short *)&lA.bsums[0] + sb * 8 + grp_y * 4 - ((sb & 1) * 6);
  
            int macc = mins[lj] * (bsum[0] + bsum[1]);
            sum_minf += (float)macc * dB_min * dA;
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      // 5.  write one result element
      const int out_row = tile_y * 4 + grp_y;
      const int out_col = tile_x * NCOL_I + grp_x;
      s[out_row * bs + out_col] = sumf - sum_minf;
    }