#pragma once
#include <tensor.h>
#include <vector>
#include <utility>
#include <string>

using nntrainer::Tensor;
using std::vector;
using std::pair;
using std::string;

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

/**
 * @brief 
 * @param query 
 * @param kcache 
 * @param attn 
 * @param from 
 * @param to 
 * @param kv_head 
 * @param h_dim 
 * @param q_per_kv 
 * @param tile_size 
 * @return 
 */
int mult_with_no_transpose(const Tensor query, const Tensor kcache, Tensor attn,
                           int from, int to, int kv_head, int h_dim,
                           int q_per_kv, int tile_size = 64);

/**
 * @brief 
 * @param query 
 * @param kcache 
 * @param vcache 
 * @param attn_weight 
 * @param attn_output 
 * @param q_head 
 * @param h_dim 
 * @param tile_size 
 * @param tok_gen 
 * @param from 
 * @param to 
 * @param qhead_per_kvhead 
 * @param exec_time exec_time will be recorded here, list of {block name, time} pair
 * @return 
 */
int _mha(Tensor query, Tensor kcache, Tensor vcache, Tensor attn_weight,
         Tensor attn_output, size_t q_head, size_t h_dim, int tile_size,
         bool tok_gen, size_t from, size_t to, int qhead_per_kvhead,
         vector<pair<string, double>> *exec_time);

/**
 * @brief 
 * @param query (1, 1, to-from, head_q * h_dim)
 * @param kcache (1, 1, to, head_kv * h_dim)
 * @param vcache (1, 1, to, head_kv * h_dim)
 * @param attn_weight (head_q, 1, to-from, to)
 * @param attn_output (head_q, 1, to-from, h_dim)
 * @param q_head 
 * @param h_dim 
 * @param tile_size 
 * @param tok_gen 
 * @param from 
 * @param to 
 * @param qhead_per_kvhead 
 * @param exec_time exec_time will be recorded here, list of {block name, time} pair
 * @return 
 */
int _mha2(Tensor query, Tensor kcache, Tensor vcache, Tensor attn_weight,
          Tensor attn_output, size_t q_head, size_t h_dim, int tile_size,
          bool tok_gen, size_t from, size_t to, int qhead_per_kvhead,
          vector<pair<string, double>> *exec_time);

/**
 * @brief
 * @param query
 * @param kcache
 * @param vcache
 * @param attn_weight
 * @param attn_output
 * @param q_head
 * @param h_dim
 * @param tile_size
 * @param tok_gen
 * @param from
 * @param to
 * @param qhead_per_kvhead
 * @param exec_time exec_time will be recorded here, list of {block name, time}
 * pair
 * @return
 */
int _get_attn_weight(Tensor query, Tensor kcache, Tensor vcache,
                     Tensor attn_weight, Tensor attn_output, size_t q_head,
                     size_t h_dim, int tile_size, bool tok_gen, size_t from,
                     size_t to, int qhead_per_kvhead,
                     vector<pair<string, double>> *exec_time);

int _get_attn_weight2(Tensor query, Tensor kcache, Tensor vcache,
                      Tensor attn_weight, Tensor attn_output, size_t q_head,
                      size_t h_dim, int tile_size, bool tok_gen, size_t from,
                      size_t to, int qhead_per_kvhead,
                      vector<pair<string, double>> *exec_time);