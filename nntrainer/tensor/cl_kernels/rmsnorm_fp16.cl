#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void
rmsnorm_cl_fp16(__global const half *input, // Input tensor
                __global half *output,      // Output tensor
                __global const half *alpha, // Alpha values (one for each width)
                half epsilon,
                int B, // Number of batches
                int C, // Number of channels
                int H, // Height of feature map
                int W  // Width of feature map
) {
  int global_id = get_global_id(0); // Get the global work item index

  // Compute the corresponding batch, height, and channel indices
  int n = global_id / C;    // Batch index
  int c = global_id % C;    // Height index
  int h = get_global_id(1); // Channel index
  int index = ((n * C + c) * H + h) * W;

  // Calculate RMS norm for the current channel, height, and batch
  half sum_squares = 0.0f;
  for (int j = 0; j < W; ++j) {
    sum_squares += input[index + j] * input[index + j];
  }
  sum_squares /= W;
  half rms_norm = sqrt((float)(sum_squares + epsilon));
  // Each work item processes all width elements for its specific n, h, c
  for (int w = 0; w < W; ++w) {
    output[index + w] = (input[index + w] / rms_norm) * alpha[w];
  }
}
