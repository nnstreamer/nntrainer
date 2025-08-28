#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void concat_cl_axis1_fp16(__global const half *input1,
                                   __global const half *input2,
                                   __global half *output, const int batch_size,
                                   const int channel1, const int channel2,
                                   const int height_size,
                                   const int width_size) {
  // Get single global index
  const int global_idx = get_global_id(0);

  // Calculate total elements in one channel concatenation
  const int total_elements = batch_size * height_size * width_size;

  // Check if index is within bounds
  if (global_idx >= total_elements) {
    return;
  }

  // Calculate indices for batch, height, and width
  const int batch_idx = global_idx / (height_size * width_size);
  const int temp = global_idx % (height_size * width_size);
  const int height_idx = temp / width_size;
  const int width_idx = temp % width_size;

  // Calculate strides for input1
  const int stride_batch1 = channel1 * height_size * width_size;
  const int stride_channel1 = height_size * width_size;
  const int stride_height1 = width_size;

  // Calculate strides for input2
  const int stride_batch2 = channel2 * height_size * width_size;
  const int stride_channel2 = height_size * width_size;
  const int stride_height2 = width_size;

  // Calculate strides for output
  const int total_channels = channel1 + channel2;
  const int stride_batch_out = total_channels * height_size * width_size;
  const int stride_channel_out = height_size * width_size;
  const int stride_height_out = width_size;

  // Calculate base indices
  const int base_idx1 = batch_idx * stride_batch1;
  const int base_idx2 = batch_idx * stride_batch2;
  const int base_idx_out = batch_idx * stride_batch_out;

  // Calculate spatial offset
  const int spatial_offset = height_idx * stride_height_out + width_idx;

  // Copy data from input1
  for (int c = 0; c < channel1; c++) {
    output[base_idx_out + c * stride_channel_out + spatial_offset] =
      input1[base_idx1 + c * stride_channel1 + height_idx * stride_height1 +
             width_idx];
  }

  // Copy data from input2
  for (int c = 0; c < channel2; c++) {
    output[base_idx_out + (channel1 + c) * stride_channel_out +
           spatial_offset] = input2[base_idx2 + c * stride_channel2 +
                                    height_idx * stride_height2 + width_idx];
  }
}
