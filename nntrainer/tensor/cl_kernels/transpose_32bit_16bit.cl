#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 32-bit transpose, loading/storing a 4x4 tile of elements
// Only used for activations
// converts to FP16
// also adds zero padding for non multiple of 8 prompt lengths
kernel void kernel_transpose_32_16(
  __global const float4 *input, // FP32 source
  __global half4 *output,       // FP16 destination
  const uint rows,       // original "rows" in tiles (height/4 of the source)
  const uint cols,       // original "cols" in tiles (width  of the source)
  const uint padded_rows // destination rows after padding
) {
  const uint i = get_global_id(0); // tile x
  const uint j = get_global_id(1); // tile y
  const uint i_2 = i << 2;         // i * 4
  const uint j_2 = j << 2;         // j * 4

  half4 t0 = (half4)(0.0h, 0.0h, 0.0h, 0.0h);
  half4 t1 = t0, t2 = t0, t3 = t0;

  const uint total = rows * cols * 16;

  // Read 4 rows from the same column i, converting FP32 -> FP16
  uint idx;

  idx = (j_2 + 0) * cols + i;
  if (idx < total)
    t0 = convert_half4(input[idx]);

  idx = (j_2 + 1) * cols + i;
  if (idx < total)
    t1 = convert_half4(input[idx]);

  idx = (j_2 + 2) * cols + i;
  if (idx < total)
    t2 = convert_half4(input[idx]);

  idx = (j_2 + 3) * cols + i;
  if (idx < total)
    t3 = convert_half4(input[idx]);

  output[(i_2 + 0) * padded_rows + j] = (half4)(t0.s0, t1.s0, t2.s0, t3.s0);
  output[(i_2 + 1) * padded_rows + j] = (half4)(t0.s1, t1.s1, t2.s1, t3.s1);
  output[(i_2 + 2) * padded_rows + j] = (half4)(t0.s2, t1.s2, t2.s2, t3.s2);
  output[(i_2 + 3) * padded_rows + j] = (half4)(t0.s3, t1.s3, t2.s3, t3.s3);
}
