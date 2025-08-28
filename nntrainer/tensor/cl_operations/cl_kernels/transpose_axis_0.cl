__kernel void transpose_cl_axis0(__global const float *in,
                                 __global float *output, const int batch_size,
                                 const int channels, const int height,
                                 const int width) {
  // Calculate h and w from the global IDs
  int h = get_global_id(0);
  int w = get_global_id(1);
  if (h < height && w < width) {
    for (int c = 0; c < channels; ++c) {
      for (int n = 0; n < batch_size; ++n) {
        // Calculate the input and output indices
        int input_index = n * (channels * height * width) +
                          c * (height * width) + h * width + w;
        int output_index = n * (channels * height * width) +
                           h * (channels * width) + c * width + w;
        // Transpose channel and height, copying data from input to output
        output[output_index] = in[input_index];
      }
    }
  }
}
