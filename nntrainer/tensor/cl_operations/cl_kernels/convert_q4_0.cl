#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable
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

REQD_SUBGROUP_SIZE_16
kernel void
kernel_convert_q4_0_to_y_x_yblock16(global const struct block_q4_0 *src,
                                    global uchar *dst_q, global half *dst_d,
                                    int width) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int sid = get_sub_group_local_id();

  const int input_index = (y * width / 2 + x) / 16;

  global const struct block_q4_0 *in = src + input_index;
  if (sub_group_elect()) {
    dst_d[input_index] = in->d;
  }

  uint8_t val = in->qs[sid];
  const uint output_index = y % 16 + 16 * (x + (y / 16) * (width / 2));
  dst_q[output_index] = val;
}
