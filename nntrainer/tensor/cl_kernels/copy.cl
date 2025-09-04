__kernel void copy_cl(__global const float *input, __global float *output,
                      const int batchsize, const int channels, const int height,
                      const int width) {
  int i = get_global_id(0);
  output[i] = input[i];
}
