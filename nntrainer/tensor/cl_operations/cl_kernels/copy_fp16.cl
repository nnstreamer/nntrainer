#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void copy_cl_fp16(__global const half *input, __global half *output,
                           const int batchsize, const int channels,
                           const int height, const int width) {

  int i = get_global_id(0);
  output[i] = input[i];
}
