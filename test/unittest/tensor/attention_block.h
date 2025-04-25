#pragma once
void mult_with_avx(const float *A, const float *B, float *output, int num_rows,
                   int N, int chunk_size, int group_size, int tile_size = 64);

/**
 * @brief MV mult. A is vector,
 * @param Q Query, is N*group_size
 * @param K K cache
 * @param output 
 * @param seq_len sequence_len
 * @param kv_head 
 * @param hidden_dim hidden dimension
 * @param grp_factor {# of Q head} / {# of KV head}
 * @param tile_size 
 */
void multiply_and_reduce_chunks(const float *Q, const float *K, float *output,
                                int seq_len, int kv_head, int hidden_dim,
                                int grp_factor, int tile_size = 64);

