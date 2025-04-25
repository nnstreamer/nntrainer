#include "attention_block.h"

#include <algorithm>
#include <float_tensor.h>
#include <fstream>
#include <immintrin.h>
#include <tensor.h>
#include <tensor_dim.h>

void repack_B_tile(const float *B, float *B_tile, int row_tile_start,
                   int tile_rows, int n, int chunk_size, int N) {
  for (int row = 0; row < tile_rows; ++row) {
    const float *src =
      B + (row_tile_start + row) * N * chunk_size + n * chunk_size;
    float *dst = B_tile + row * chunk_size;
    std::memcpy(dst, src, sizeof(float) * chunk_size);
  }
}

void multiply_and_reduce_chunks(const float *A, const float *B, float *output,
                                int seq_len, int kv_head, int hidden_dim,
                                int grp_factor, int tile_size) {

  // const bool use_repacked = group_size >=4;
  const bool use_repacked = false;
  float *B_tile = use_repacked ? new float[tile_size * hidden_dim] : nullptr;

  const int group_stride = grp_factor * hidden_dim;
  const int row_stride = kv_head * hidden_dim;

  for (int n = 0; n < kv_head; ++n) {
    for (int g = 0; g < grp_factor; ++g) {
      const float *a_ptr = A + n * group_stride + g * hidden_dim;

      for (int row_tile_start = 0; row_tile_start < seq_len;
           row_tile_start += tile_size) {
        int tile_rows = std::min(tile_size, seq_len - row_tile_start);

        const float *b_ptr = nullptr;
        if (use_repacked) {
          repack_B_tile(B, B_tile, row_tile_start, tile_rows, n, hidden_dim, kv_head);
          b_ptr = B_tile;
        }

        for (int row = 0; row < tile_rows; ++row) {
          const float *b_row =
            use_repacked
              ? b_ptr + row * hidden_dim
              : B + (row_tile_start + row) * row_stride + n * hidden_dim;

          __m256 sum = _mm256_setzero_ps();
          int k = 0;

          for (; k + 7 < hidden_dim; k += 8) {
            __m256 va = _mm256_loadu_ps(a_ptr + k);
            __m256 vb = _mm256_loadu_ps(b_row + k);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
          }

          float tail_sum = 0.0f;
          for (; k < hidden_dim; ++k) {
            tail_sum += a_ptr[k] * b_row[k];
          }

          __m128 low = _mm256_castps256_ps128(sum);
          __m128 high = _mm256_extractf128_ps(sum, 1);
          __m128 sum128 = _mm_add_ps(low, high);
          sum128 = _mm_hadd_ps(sum128, sum128);
          sum128 = _mm_hadd_ps(sum128, sum128);
          float vec_sum = _mm_cvtss_f32(sum128);

          //output[((row_tile_start + row) * grp_factor + g) * kv_head + n] =
          //  vec_sum + tail_sum;
          output[(n * grp_factor + g) * seq_len + (row_tile_start + row)] =
            vec_sum + tail_sum;
        }
      }
    }
  }

  if (B_tile)
    delete[] B_tile;
};

void mult_with_avx(const float *A, const float *B, float *output, int num_rows,
                   int N, int chunk_size, int group_size, int tile_size) 
{

}
