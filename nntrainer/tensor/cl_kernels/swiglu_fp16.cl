#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void swiglu_cl_fp16(__global const half *in1, __global const half *in2,
                             __global half *out) {
  const int i = get_global_id(0);

  const half in1_val = in1[i];
  const half in2_val = in2[i];

  const half in1_exp = exp(in1_val);

  const half half_one = (half)(1.0f);
  const half swish = in1_val * in1_exp / (half_one + in1_exp);

  out[i] = swish * in2_val;
}
