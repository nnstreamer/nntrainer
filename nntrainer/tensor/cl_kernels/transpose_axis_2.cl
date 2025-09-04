__kernel void transpose_cl_axis2(__global const float *in,
                                 __global float *output, const int batch_size,
                                 const int channels, const int height,
                                 const int width) {
  // Calculate c and w from the global IDs
  int c = get_global_id(0);
  int w = get_global_id(1);
  if (c < channels && w < width) {
    for (int h = 0; h < height; ++h) {
      for (int n = 0; n < batch_size; ++n) {
        // Calculate the input and output indices
        int input_index = n * (channels * height * width) +
                          c * (height * width) + h * width + w;
        int output_index = n * (channels * height * width) +
                           w * (height * channels) + h * channels + c;
        // Transpose channel and width, copying data from input to output
        output[output_index] = in[input_index];
      }
    }
  }
}
