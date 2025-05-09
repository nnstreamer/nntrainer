#include "attention_block.h"

#include <layers/acti_func.h>

#include <algorithm>
#include <chrono>
#include <fstream>

#include <math.h>

#include <immintrin.h>

#define MAX_CTX_SIZE 10240

#ifdef ENABLE_FP16
#define _MASK_NUM -1e4
#else
#define _MASK_NUM -1e10
#endif

using nntrainer::ActiFunc;
using std::cout;
using std::endl;
using std::exp;

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


int mult_with_no_transpose_prefill(const Tensor query, const Tensor kcache,
                                   Tensor attn, int from, int to, int kv_head,
                                   int h_dim, int q_per_kv, int tile_size) 
{
  return 0;
}

int mult_with_no_transpose_tokgen(const Tensor query, const Tensor kcache,
                                  Tensor attn, int from, int to, int kv_head,
                                  int h_dim, int q_per_kv, int tile_size)
{
  return 0;
}

int mult_with_no_transpose(const Tensor query, const Tensor kcache, Tensor attn,
                           int from, int to, int kv_head, int h_dim,
                           int q_per_kv, int tile_size) {
  // query: (1, 1, from-to, head_q * h_dim)
  // kcache: (1, 1, to, head_kv, h_dim)
  // attn: (q_head, 1, from-to, to)

  //cout << "query: " << endl;
  //cout << query << endl;

  //cout << "kcache: " << endl;
  //cout << kcache << endl;

  float *__Q = query.getData<float>();
  float *__K = kcache.getData<float>();
  float *__A = attn.getData<float>();
  float norm_factor = 1.0f / sqrt(h_dim);

  for (int head_idx_kv = 0, head_idx_q = 0; head_idx_kv < kv_head;
       ++head_idx_kv)
  {
    for (int group = 0; group < q_per_kv; ++group, ++head_idx_q)
    {
      for (int i = 0; i < to - from; ++i)
      {
        float *A = &__A[(head_idx_q * (to - from) + i) * to];
        //float *_Q = &__Q[i * (kv_head * q_per_kv) * h_dim];
        //float *Q = &_Q[head_idx_q * h_dim];
        float *Q = &__Q[(i * (kv_head * q_per_kv) + head_idx_q) * h_dim];
        float row_max = std::numeric_limits<float>::min();
        for (int j = 0; j <= i; ++j)
        {
          float acc = 0.0f;

          float *K = &__K[(j * kv_head + head_idx_kv) * h_dim];
          //{
          //  printf("Q[%d] = ", i);
          //  for (int ii = 0; ii < h_dim; ++ii)
          //  {
          //    printf("%.7f, ", Q[ii]);
          //  }
          //  printf("\n");

          //  printf("K[%d] = ", j);
          //  for (int ii = 0; ii < h_dim; ++ii) {
          //    printf("%.7f, ", K[ii]);
          //  }
          //  printf("\n");
          //}
          for (int k = 0; k < h_dim; ++k)
          {
            acc += K[k] * Q[k]; 
          }
          //A[j] = acc;
          A[j] = acc * norm_factor;
          row_max = std::max(row_max, A[j]);
        }

        float exp_sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
          A[j] -= row_max;
          A[j] = exp(A[j]);
          exp_sum += A[j];
        }
        for (int j = 0; j <= i; ++j) {
          A[j] /= exp_sum;
        }

        for (int j = i + 1; j < to; ++j) {
          A[j] = 0.0f;
        }
        //float *A = &__A[(head_idx_q * (to - from) + i) * to];
        //memcpy(A, tmp_row, sizeof(float) * to);
      }
    }
  }
  return 0;
}

void mult_with_avx(const float *A, const float *B, float *output, int num_rows,
                   int N, int chunk_size, int group_size, int tile_size) 
{

}

template <typename Func> double check_time(Func &&func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> dur = end - start;
  return dur.count();
}

int _mha(Tensor query, Tensor kcache, Tensor vcache, Tensor attn_weight,
         Tensor attn_output, size_t q_head, size_t h_dim, int tile_size,
         bool tok_gen, size_t from, size_t to, int qhead_per_kvhead,
         vector<pair<string, double>> *exec_time) {
  unsigned int kv_head_factor = qhead_per_kvhead;
  unsigned int kv_head = q_head / kv_head_factor;

  ActiFunc sm(nntrainer::ActivationType::ACT_SOFTMAX);

  auto preproc = [&]() {
    //cout << "query: " << endl;
    //cout << query << endl;
    //cout << "kcache: " << endl;
    //cout << kcache << endl;

    query.reshape(ml::train::TensorDim({1, to - from, q_head, h_dim}));
    if (to - from != 1)
      query.transpose("1:0:2", query);
    query.reshape(ml::train::TensorDim({q_head, 1, to - from, h_dim}));

    kcache.reshape(ml::train::TensorDim({1, to, kv_head, h_dim}));
    kcache.transpose("1:0:2", kcache);
    kcache.reshape(ml::train::TensorDim({kv_head, 1, to, h_dim}));
  };

  auto qk_mul = [&]() {
    //cout << "query: " << endl;
    //cout << query << endl;
    //cout << "kcache: " << endl;
    //cout << kcache << endl;
    attn_weight.reshape({q_head, 1, to - from, to});
    attn_weight.setZero();
    query.dotBatched(kcache, attn_weight, false, true);
    //cout << "attn_weight" << endl;
    //cout << attn_weight << endl;
  };

  auto norm_weight = [&]() { attn_weight.multiply_i(1 / sqrt((float)h_dim)); };

  auto calc_mask = [&]() {
    if (!from) {
      unsigned int mask_size = attn_weight.getDim().width();
      unsigned int mask_dim_height = mask_size;
      unsigned int mask_dim_width = mask_size;

      nntrainer::Tensor causal_mask(ml::train::TensorDim{
        1, 1, mask_size, mask_size, attn_weight.getTensorType()});

      causal_mask.setZero();

      for (unsigned int i = 0; i < mask_dim_height; ++i) {
        for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
          causal_mask.setValue(0, 0, i, j, _MASK_NUM);
        }
      }
      attn_weight.add_i(causal_mask);
    };
  };

  auto softmax = [&]() { sm.run_fn(attn_weight, attn_weight); };

  auto preproc2 = [&]() {
    vcache.reshape(ml::train::TensorDim({1, to, kv_head, h_dim}));
    vcache.transpose("1:0:2", vcache);
    vcache.reshape(ml::train::TensorDim({kv_head, 1, to, h_dim}));
  };

  auto attn_v_mul = [&]() {
    attn_output.reshape({q_head, 1, to - from, h_dim});

    //cout << "vcache" << endl;
    //cout << vcache << endl;
    //cout << "vcache.batch = " << vcache.batch() << endl;
    attn_weight.dotBatched(vcache, attn_output);
    //cout << "attn_output" << endl;
    //cout << attn_output << endl;
  };

  auto postproc = [&]() {
    if (to - from != 1) {
      attn_output.reshape(ml::train::TensorDim({1, q_head, to - from, h_dim}));
      attn_output.transpose("1:0:2", attn_output);
    }
    attn_output.reshape({to - from, 1, 1, h_dim * q_head});
  };

  if (exec_time) {
    exec_time->push_back({"preproc", check_time(preproc)});
    exec_time->push_back({"q * k", check_time(qk_mul)});
    exec_time->push_back({"norm_weight", check_time(norm_weight)});
    exec_time->push_back({"calc_mask", check_time(calc_mask)});
    exec_time->push_back({"softmax", check_time(softmax)});
    exec_time->push_back({"preproc2", check_time(preproc2)});
    exec_time->push_back({"attn * v", check_time(attn_v_mul)});
    exec_time->push_back({"postproc", check_time(postproc)});
  } else {
    check_time(preproc);
    check_time(qk_mul);
    check_time(norm_weight);
    check_time(calc_mask);
    check_time(softmax);
    check_time(preproc2);
    check_time(attn_v_mul);
    check_time(postproc);
  }

  return 0;
}

int _mha2(Tensor query, Tensor kcache, Tensor vcache, Tensor attn_weight,
          Tensor attn_output, size_t q_head, size_t h_dim, int tile_size,
          bool tok_gen, size_t from, size_t to, int qhead_per_kvhead,
          vector<pair<string, double>> *exec_time) {
  unsigned int kv_head_factor = qhead_per_kvhead;
  unsigned int kv_head = q_head / kv_head_factor;

  ActiFunc sm(nntrainer::ActivationType::ACT_SOFTMAX);

  auto preproc = [&]() {
    query.reshape(ml::train::TensorDim({1, to - from, q_head, h_dim}));
    if (to - from != 1)
      query.transpose("1:0:2", query);
    query.reshape(ml::train::TensorDim({q_head, 1, to - from, h_dim}));

    kcache.reshape(ml::train::TensorDim({1, to, kv_head, h_dim}));
    kcache.transpose("1:0:2", kcache);
    kcache.reshape(ml::train::TensorDim({kv_head, 1, to, h_dim}));
  };

  auto qk_mul = [&]() {
    // cout << "query: " << endl;
    // cout << query << endl;
    // cout << "kcache: " << endl;
    // cout << kcache << endl;
    attn_weight.reshape({q_head, 1, to - from, to});
    attn_weight.setZero();
    query.dotBatched(kcache, attn_weight, false, true);
  };

  auto qk_mul_no_preproc = [&]() {
    multiply_and_reduce_chunks(
      query.getData<float>(), kcache.getData<float>(),
      attn_weight.getData<float>(), to, kv_head,
      q_head, kv_head_factor, tile_size);
    //cout << "attn_weight" << endl;
    //cout << attn_weight << endl;
  };

  auto norm_weight = [&]() { attn_weight.multiply_i(1 / sqrt((float)h_dim)); };

  auto calc_mask = [&]() {
    if (!from) {
      unsigned int mask_size = attn_weight.getDim().width();
      unsigned int mask_dim_height = mask_size;
      unsigned int mask_dim_width = mask_size;

      nntrainer::Tensor causal_mask(ml::train::TensorDim{
        1, 1, mask_size, mask_size, attn_weight.getTensorType()});

      causal_mask.setZero();

      for (unsigned int i = 0; i < mask_dim_height; ++i) {
        for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
          causal_mask.setValue(0, 0, i, j, _MASK_NUM);
        }
      }
      attn_weight.add_i(causal_mask);
    };
  };

  auto softmax = [&]() { sm.run_fn(attn_weight, attn_weight); };

  auto preproc2 = [&]() {
    vcache.reshape(ml::train::TensorDim({1, to, kv_head, h_dim}));
    vcache.transpose("1:0:2", vcache);
    vcache.reshape(ml::train::TensorDim({kv_head, 1, to, h_dim}));
  };

  auto attn_v_mul = [&]() {
    attn_output.reshape({q_head, 1, to - from, h_dim});
    //cout << "vcache" << endl;
    //cout << vcache << endl;
    //cout << "vcache.batch = " << vcache.batch() << endl;
    attn_weight.dotBatched(vcache, attn_output);
    //cout << "attn_output" << endl;
    //cout << attn_output << endl;
  };

  auto postproc = [&]() {
    if (to - from != 1) {
      attn_output.reshape(ml::train::TensorDim({1, q_head, to - from, h_dim}));
      attn_output.transpose("1:0:2", attn_output);
    }
    attn_output.reshape({to - from, 1, 1, h_dim * q_head});
  };

  if (exec_time) {
    //exec_time->push_back({"preproc", check_time(preproc)});
    exec_time->push_back({"q * k", check_time(qk_mul_no_preproc)});
    exec_time->push_back({"norm_weight", check_time(norm_weight)});
    exec_time->push_back({"calc_mask", check_time(calc_mask)});
    exec_time->push_back({"softmax", check_time(softmax)});
    exec_time->push_back({"preproc2", check_time(preproc2)});
    exec_time->push_back({"attn * v", check_time(attn_v_mul)});
    exec_time->push_back({"postproc", check_time(postproc)});
  } else {
    //check_time(preproc);
    check_time(qk_mul_no_preproc);
    check_time(norm_weight);
    check_time(calc_mask);
    check_time(softmax);
    check_time(preproc2);
    check_time(attn_v_mul);
    check_time(postproc);
  }

  return 0;
}

int _get_attn_weight(Tensor query, Tensor kcache, Tensor vcache,
  Tensor attn_weight, Tensor attn_output, size_t q_head,
  size_t h_dim, int tile_size, bool tok_gen, size_t from,
  size_t to, int qhead_per_kvhead,
  vector<pair<string, double>>* exec_time)
{
  unsigned int kv_head_factor = qhead_per_kvhead;
  unsigned int kv_head = q_head / kv_head_factor;

  ActiFunc sm(nntrainer::ActivationType::ACT_SOFTMAX);

  auto preproc = [&]() {
    //cout << "query: " << endl;
    //cout << query << endl;
    //cout << "kcache: " << endl;
    //cout << kcache << endl;

    query.reshape(ml::train::TensorDim({1, to - from, q_head, h_dim}));
    if (to - from != 1)
      query.transpose("1:0:2", query);
    query.reshape(ml::train::TensorDim({q_head, 1, to - from, h_dim}));

    kcache.reshape(ml::train::TensorDim({1, to, kv_head, h_dim}));
    kcache.transpose("1:0:2", kcache);
    kcache.reshape(ml::train::TensorDim({kv_head, 1, to, h_dim}));
  };

  auto qk_mul = [&]() {
    //cout << "query: " << endl;
    //cout << query << endl;
    //cout << "kcache: " << endl;
    //cout << kcache << endl;
    attn_weight.reshape({q_head, 1, to - from, to});
    attn_weight.setZero();
    query.dotBatched(kcache, attn_weight, false, true);
  };

  auto norm_weight = [&]() { attn_weight.multiply_i(1 / sqrt((float)h_dim)); };

  auto calc_mask = [&]() {
    if (!from) {
      unsigned int mask_size = attn_weight.getDim().width();
      unsigned int mask_dim_height = mask_size;
      unsigned int mask_dim_width = mask_size;

      nntrainer::Tensor causal_mask(ml::train::TensorDim{
        1, 1, mask_size, mask_size, attn_weight.getTensorType()});

      causal_mask.setZero();

      for (unsigned int i = 0; i < mask_dim_height; ++i) {
        for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
          causal_mask.setValue(0, 0, i, j, _MASK_NUM);
        }
      }
      attn_weight.add_i(causal_mask);
    };
  };

  auto softmax = [&]() { sm.run_fn(attn_weight, attn_weight); };

  if (exec_time) {
    exec_time->push_back({"preproc", check_time(preproc)});
    exec_time->push_back({"q * k", check_time(qk_mul)});
    exec_time->push_back({"norm_weight", check_time(norm_weight)});
    exec_time->push_back({"calc_mask", check_time(calc_mask)});
    exec_time->push_back({"softmax", check_time(softmax)});
  } else {
    check_time(preproc);
    check_time(qk_mul);
    check_time(norm_weight);
    check_time(calc_mask);
    check_time(softmax);
  }

  return 0;
}

int _get_attn_weight2(Tensor query, Tensor kcache, Tensor vcache,
  Tensor attn_weight, Tensor attn_output, size_t q_head,
  size_t h_dim, int tile_size, bool tok_gen, size_t from,
  size_t to, int qhead_per_kvhead,
  vector<pair<string, double>>* exec_time)
{
  auto get_attn_weight = [&]() {
    mult_with_no_transpose(query, kcache, attn_weight, from, to,
                           q_head / qhead_per_kvhead, h_dim, qhead_per_kvhead);
  };
  if (exec_time)
  {
    exec_time->push_back({"get_attn_weight", check_time(get_attn_weight)});
  }
  else
  {
    check_time(get_attn_weight);
  }
  return 0;
}
