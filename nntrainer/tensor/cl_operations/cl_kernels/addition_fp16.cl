#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void addition_cl_fp16(const __global half *input,
                               __global half *output, unsigned int size_input,
                               unsigned int size_res) {
  size_t idx = get_global_id(0);
  if (idx < size_res) {
    output[idx] = output[idx] + input[idx % size_input];
  }
}
