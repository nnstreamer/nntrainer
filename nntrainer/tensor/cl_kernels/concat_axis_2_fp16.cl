#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void concat_cl_axis2_fp16(__global const half *input1,
                                   __global const half *input2,
                                   __global half *output, const int batch_size,
                                   const int channel_size, const int height1,
                                   const int height2, const int width_size) {
  // Get single global index
  const int global_idx = get_global_id(0);

  // Calculate total elements in one height concatenation
  const int total_elements = batch_size * channel_size * width_size;

  // Check if index is within bounds
  if (global_idx >= total_elements) {
    return;
  }

  // Calculate indices for batch, channel, and width
  const int batch_idx = global_idx / (channel_size * width_size);
  const int temp = global_idx % (channel_size * width_size);
  const int channel_idx = temp / width_size;
  const int width_idx = temp % width_size;

  // Calculate strides for input1
  const int stride_batch1 = channel_size * height1 * width_size;
  const int stride_channel1 = height1 * width_size;
  const int stride_height1 = width_size;

  // Calculate strides for input2
  const int stride_batch2 = channel_size * height2 * width_size;
  const int stride_channel2 = height2 * width_size;
  const int stride_height2 = width_size;

  // Calculate strides for output
  const int total_height = height1 + height2;
  const int stride_batch_out = channel_size * total_height * width_size;
  const int stride_channel_out = total_height * width_size;
  const int stride_height_out = width_size;

  // Calculate base indices
  const int base_idx1 =
    batch_idx * stride_batch1 + channel_idx * stride_channel1;

  const int base_idx2 =
    batch_idx * stride_batch2 + channel_idx * stride_channel2;

  const int base_idx_out =
    batch_idx * stride_batch_out + channel_idx * stride_channel_out;

  // Copy data from input1
  for (int h = 0; h < height1; h++) {
    output[base_idx_out + h * stride_height_out + width_idx] =
      input1[base_idx1 + h * stride_height1 + width_idx];
  }

  // Copy data from input2
  for (int h = 0; h < height2; h++) {
    output[base_idx_out + (height1 + h) * stride_height_out + width_idx] =
      input2[base_idx2 + h * stride_height2 + width_idx];
  }
}
