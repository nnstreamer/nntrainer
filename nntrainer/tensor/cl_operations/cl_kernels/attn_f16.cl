#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_SIZE 16

__kernel void attn_f16_q_mul_transpose_k(
  /* input matrix  Q                     */ const global half *restrict Q,
  /* input matrix  K                     */ const global half *restrict K,
  /* output matrix LOGITS                */ global half *restrict LOGITS,
  /* row for K and column for LOGITS     */ const uint n,
  /* column ount for  Q and column for K */ const uint d_k,
  /* 1.0f / sqrt(d_k)                    */ const float inverse_sqrt_d_k) {
  //
  //
  //
  // Q - [m * d_k]
  // K - [n * d_k]
  // LOGITS - [m * n]
  //
  // LOGITS = Q * kernel_time_transpose(K) / sqrt(d_k)

  const int global_row = get_global_id(0);
  const int global_column = get_global_id(1);

  const int local_row = get_local_id(0);
  const int local_column = get_local_id(1);

  local float local_Q[TILE_SIZE * TILE_SIZE];
  local float local_K[TILE_SIZE * TILE_SIZE];

  const int tile_count = d_k / TILE_SIZE;

  float sum = 0.0f;
  for (int t = 0; t < tile_count; t++) {
    local_Q[local_row * TILE_SIZE + local_column] =
      Q[global_row * d_k + t * TILE_SIZE + local_column];
    local_K[local_row * TILE_SIZE + local_column] =
      K[global_column * d_k + t * TILE_SIZE + local_row];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_SIZE; k++) {
      sum += local_Q[local_row * TILE_SIZE + k] *
             local_K[local_column * TILE_SIZE + k];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  LOGITS[global_row * n + global_column] = sum * inverse_sqrt_d_k;
}

__kernel void attn_f16_softmax_row(
  /* inout matrix LOGITS [m, n] */ global half *restrict LOGITS,
  /* rows of LOGITS             */ const int m,
  /* columns of LOGITS          */ const int n) {
  //
  //
  //
  // LOGITS = softmax_row(LOGITS)

  const int global_row = get_global_id(0);

  float max_val = LOGITS[global_row * n];

  for (int j = 1; j < n; j++) {
    const float val = LOGITS[global_row * n + j];

    if (val > max_val) {
      max_val = val;
    }
  }

  float sum_exp = 0.0f;
  for (int j = 0; j < n; j++) {
    const float e = exp(LOGITS[global_row * n + j] - max_val);

    LOGITS[global_row * n + j] = e;
    sum_exp += e;
  }

  for (int j = 0; j < n; j++) {
    LOGITS[global_row * n + j] /= sum_exp;
  }
}

__kernel void attn_f16_sm_mul_v(
  /*output of softmax row */ global const half *restrict SM,
  /*V [n, d_v]            */ global const half *restrict V,
  /*                      */ global half *restrict O,
  /*                      */ const int n,
  /*                      */ const int d_v) {
  //
  //
  //
  // O = SM * V

  local float local_S[TILE_SIZE][TILE_SIZE];
  local float local_V[TILE_SIZE][TILE_SIZE];

  const int global_row = get_global_id(0);
  const int global_column = get_global_id(1);

  const int local_row = get_local_id(0);
  const int local_col = get_local_id(1);

  const int tile_count = n / TILE_SIZE;

  float sum = 0.0f;
  for (int t = 0; t < tile_count; t++) {
    local_S[local_row][local_col] =
      SM[global_row * n + t * TILE_SIZE + local_col];
    local_V[local_row][local_col] =
      V[(t * TILE_SIZE + local_row) * d_v + global_column];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_SIZE; k++) {
      sum += local_S[local_row][k] * local_V[k][local_col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  O[global_row * d_v + global_column] = sum;
}