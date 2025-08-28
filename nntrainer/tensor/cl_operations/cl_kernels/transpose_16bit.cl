#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 16-bit transpose, loading/storing a 4x4 tile of elements (via buffers)
kernel void
kernel_transpose_16(__global const half4 *input, // was image1d_buffer_t
                    __global half4 *output,      // was image1d_buffer_t
                    const uint rows,             // = get_global_size(1)
                    const uint cols              // = get_global_size(0)
) {
  const uint i = get_global_id(0);
  const uint j = get_global_id(1);
  const uint i_2 = i << 2; // 4 * i
  const uint j_2 = j << 2; // 4 * j

  // Load four consecutive rows (each element is half4)
  const half4 temp0 = input[(j_2 + 0) * cols + i];
  const half4 temp1 = input[(j_2 + 1) * cols + i];
  const half4 temp2 = input[(j_2 + 2) * cols + i];
  const half4 temp3 = input[(j_2 + 3) * cols + i];

  // Write transposed 4x4 tile (each write is a half4 column)
  output[(i_2 + 0) * rows + j] =
    (half4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0);
  output[(i_2 + 1) * rows + j] =
    (half4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1);
  output[(i_2 + 2) * rows + j] =
    (half4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2);
  output[(i_2 + 3) * rows + j] =
    (half4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3);
}
