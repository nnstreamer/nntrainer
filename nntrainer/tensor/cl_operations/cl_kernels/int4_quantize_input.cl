#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

kernel void quantize_input_int4(const __global half *restrict input,
                                __global char *restrict quantized_input,
                                __global half *restrict quan_var) {
  const uint offset = get_global_id(0);
  const uint input_offset = offset * SIZE_QUANTIZATION_GROUP;
  const uint quantize_block = SIZE_QUANTIZATION_GROUP / 4;
  half4 input_0;
  char4 quantized_value;
  half max[quantize_block];

  unroll_for(uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + (i * 4)]);
    max[i] = fmax(fmax(fabs(input_0[0]), fabs(input_0[1])),
                  fmax(fabs(input_0[2]), fabs(input_0[3])));
  }

  half max_value = fmax(fmax(fmax(max[0], max[1]), fmax(max[2], max[3])),
                        fmax(fmax(max[4], max[5]), fmax(max[6], max[7])));
  max_value = fmax(max_value, 0.001h);

  float quan_scale = convert_float(max_value) / 127.0f;

  for (uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + (i * 4)]);
    float4 buff = convert_float4(input_0) / quan_scale;
    quantized_value = convert_char4_rte(buff);
    vstore4(quantized_value, 0, &quantized_input[input_offset + (i * 4)]);
  }

  // Pair of quantizing_scale and quantized activation_sum for each group
  quan_var[offset * 2] = convert_half(quan_scale);
}

kernel void quantize_input_int4_pad(const __global half *restrict input,
                                    __global char *restrict quantized_input,
                                    __global half *restrict quan_var) {
  const uint group_id = get_global_id(0);
  const uint groups_in_row = SIZE_K / SIZE_QUANTIZATION_GROUP;
  const uint row_id = group_id / groups_in_row;
  const uint group_id_in_row = group_id % groups_in_row;
  // const uint input_offset = group_id * QUANTIZE_GROUP_SIZE;
  const uint input_offset =
    (row_id * SIZE_K_ORIG) + (group_id_in_row * SIZE_QUANTIZATION_GROUP);
  const uint output_offset = group_id * SIZE_QUANTIZATION_GROUP;
  const uint max_quantize_block = SIZE_QUANTIZATION_GROUP / 4;
  uint quantize_block;

  if (group_id_in_row == groups_in_row - 1) {
    quantize_block = (SIZE_QUANTIZATION_GROUP - (SIZE_K - SIZE_K_ORIG)) / 4;
  } else {
    quantize_block = SIZE_QUANTIZATION_GROUP / 4;
  }

  half4 input_0;
  char4 quantized_value;
  half max[max_quantize_block] = {0};

  unroll_for(uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + (i * 4)]);
    max[i] = fmax(fmax(fabs(input_0[0]), fabs(input_0[1])),
                  fmax(fabs(input_0[2]), fabs(input_0[3])));
  }

  half max_value = fmax(fmax(fmax(max[0], max[1]), fmax(max[2], max[3])),
                        fmax(fmax(max[4], max[5]), fmax(max[6], max[7])));
  max_value = fmax(max_value, 0.001h);

  float quan_scale = convert_float(max_value) / 127.0f;

  for (uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + (i * 4)]);
    float4 buff = convert_float4(input_0) / quan_scale;
    quantized_value = convert_char4_rte(buff);
    vstore4(quantized_value, 0, &quantized_input[output_offset + (i * 4)]);
  }

  char4 zero_value = 0;
  for (uint i = quantize_block; i < max_quantize_block; ++i) {
    vstore4(zero_value, 0, &quantized_input[output_offset + (i * 4)]);
  }

  // Pair of quantizing_scale and quantized activation_sum for each group
  quan_var[group_id * 2] = convert_half(quan_scale);
}