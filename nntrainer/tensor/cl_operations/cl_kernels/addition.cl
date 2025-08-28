__kernel void addition_cl(const __global float *input, __global float *output,
                          unsigned int size_input, unsigned int size_res) {
#pragma printf_support
  size_t idx = get_global_id(0);
  if (idx < size_res) {
    output[idx] = output[idx] + input[idx % size_input];
  }
}
