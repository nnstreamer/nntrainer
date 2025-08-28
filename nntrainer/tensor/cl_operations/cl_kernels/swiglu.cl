__kernel void swiglu_cl(__global const float *restrict in1,
                        __global const float *restrict in2,
                        __global float *restrict out) {
  const int i = get_global_id(0);

  const float in1_val = in1[i];
  const float in2_val = in2[i];

  const float in1_exp = exp(in1_val);

  const float swish = in1_val * in1_exp / (1.0f + in1_exp);

  out[i] = swish * in2_val;
}
