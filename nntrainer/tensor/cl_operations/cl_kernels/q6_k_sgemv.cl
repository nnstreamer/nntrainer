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
  int8_t scales[QK_K / 16];
  half d;
} block_q6_K;

kernel void kernel_mul_mv_q6_K_f32(global void *src0, ulong offset0,
                                   global float *src1, ulong offset1,
                                   global float *dst, ulong offsetd, int ne00,
                                   int ne01, int ne02, int ne10, int ne12,
                                   int ne0, int ne1, int r2, int r3) {
  __local float reduction_buf[N_SIMDGROUP][N_SIMDWIDTH];

  src0 = (global void *)((global char *)src0 + offset0);
  src1 = (global float *)((global char *)src1 + offset1);
  dst = (global float *)((global char *)dst + offsetd);

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

  ulong offset_src0 =
    (i12 / r2) * (nb * ne01) + (i13 / r3) * (nb * ne01 * ne02);

  global block_q6_K *x = (global block_q6_K *)src0 + row * nb + offset_src0;
  global float *yy = (global float *)src1 + r1 * ne10 + im * ne00 * ne1;

  uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;

  int tid = lane / BLOCK_STRIDE;
  int ix = lane % BLOCK_STRIDE;
  int ip = tid / 8;
  int il = tid % 8;
  int n = 4;
  int l0 = n * il;
  int is = 8 * ip + l0 / 16;

  int y_offset = 128 * ip + l0;
  int q_offset_l = 64 * ip + l0;
  int q_offset_h = 32 * ip + l0;

  float sumf = 0.0f;

  for (int i = ix; i < nb; i += BLOCK_STRIDE) {
    global uint8_t *q1 = x[i].ql + q_offset_l;
    global uint8_t *q2 = q1 + QK_K / 8;
    global uint8_t *qh = x[i].qh + q_offset_h;
    global int8_t *sc = x[i].scales + is;
    global float *y = yy + i * QK_K + y_offset;

    float dall = x[i].d;
    float4 sums = {0.f, 0.f, 0.f, 0.f};

    for (int j = 0; j < 4; j++) {
      sums.s0 +=
        y[j + 0] * ((float)((q1[j] & 0xF) | ((qh[j] & kmask1) << 4)) - 32.f);
      sums.s1 +=
        y[j + 32] * ((float)((q2[j] & 0xF) | ((qh[j] & kmask2) << 2)) - 32.f);
      sums.s2 +=
        y[j + 64] * ((float)((q1[j] >> 4) | ((qh[j] & kmask3) >> 0)) - 32.f);
      sums.s3 +=
        y[j + 96] * ((float)((q2[j] >> 4) | ((qh[j] & kmask4) >> 2)) - 32.f);
    }

    sumf += dall * (sums.s0 * sc[0] + sums.s1 * sc[2] + sums.s2 * sc[4] +
                    sums.s3 * sc[6]);
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
