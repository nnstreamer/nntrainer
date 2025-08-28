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
