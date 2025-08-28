#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void concat_cl_axis3_fp16(__global const half *input1,
                                   __global const half *input2,
                                   __global half *output, const int batch_size,
                                   const int channel_size,
                                   const int height_size, const int width1,
                                   const int width2) {
  // Get single global index
  const int global_idx = get_global_id(0);

  // Calculate total elements in one width concatenation
  const int total_elements = batch_size * channel_size * height_size;

  // Check if index is within bounds
  if (global_idx >= total_elements) {
    return;
  }

  // Calculate indices for batch, channel, and height
  const int batch_idx = global_idx / (channel_size * height_size);
  const int temp = global_idx % (channel_size * height_size);
  const int channel_idx = temp / height_size;
  const int height_idx = temp % height_size;

  // Calculate strides for input1
  const int stride_batch1 = channel_size * height_size * width1;
  const int stride_channel1 = height_size * width1;
  const int stride_height1 = width1;

  // Calculate strides for input2
  const int stride_batch2 = channel_size * height_size * width2;
  const int stride_channel2 = height_size * width2;
  const int stride_height2 = width2;

  // Calculate strides for output
  const int total_width = width1 + width2;
  const int stride_batch_out = channel_size * height_size * total_width;
  const int stride_channel_out = height_size * total_width;
  const int stride_height_out = total_width;

  // Calculate base indices
  const int base_idx1 = batch_idx * stride_batch1 +
                        channel_idx * stride_channel1 +
                        height_idx * stride_height1;

  const int base_idx2 = batch_idx * stride_batch2 +
                        channel_idx * stride_channel2 +
                        height_idx * stride_height2;

  const int base_idx_out = batch_idx * stride_batch_out +
                           channel_idx * stride_channel_out +
                           height_idx * stride_height_out;

  // Copy data from input1
  for (int w = 0; w < width1; w++) {
    output[base_idx_out + w] = input1[base_idx1 + w];
  }

  // Copy data from input2
  for (int w = 0; w < width2; w++) {
    output[base_idx_out + width1 + w] = input2[base_idx2 + w];
  }
}
