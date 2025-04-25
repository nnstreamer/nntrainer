// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_tensor_feature.cpp
 * @date        17 April 2025
 * @brief       Unit test feature for tensor.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "attention_block.h"
#include "nntrainer_test_util.h"
#include "util_func.h"
#include <algorithm>
#include <float_tensor.h>
#include <fstream>
#include <immintrin.h>
#include <layers/acti_func.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

using namespace nntrainer;
using namespace std;

struct AttnBlockTestData {
  Tensor input;
  Tensor kcache;
  Tensor vcache;
  Tensor attn_weight;
  Tensor output;

  AttnBlockTestData(size_t q_head, size_t h_dim, int tile_size, bool tok_gen,
                    size_t from, size_t to, int qhead_per_kvhead) {
    TensorDim::TensorType t_type;
    t_type.format = Tformat::NCHW;
    t_type.data_type = Tdatatype::FP32;

    auto seq_len = to - from;
    auto kv_head_factor = qhead_per_kvhead;
    auto kv_head = q_head / kv_head_factor;

    input = Tensor(1, 1, seq_len, q_head * h_dim, t_type);
    kcache = Tensor(1, 1, to, h_dim * kv_head, t_type);
    vcache = Tensor(1, 1, to, h_dim * kv_head, t_type);

    attn_weight = Tensor(q_head, 1, seq_len, to, t_type);
    output = Tensor(q_head, 1, seq_len, h_dim, t_type);

    input.setRandUniform(-0.5, 0.0);
    kcache.setRandUniform(-0.5, 0.0);
    vcache.setRandUniform(-0.5, 0.0);
  }
};

template<typename Func>
double check_time(Func&& func)
{
  auto start = chrono::high_resolution_clock::now();
  func();
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> dur = end - start;
  return dur.count();
}

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
 * @return
 */
int _mha(Tensor query, Tensor kcache, Tensor vcache, Tensor attn_weight,
         Tensor attn_output, size_t q_head, size_t h_dim, int tile_size,
         bool tok_gen, size_t from, size_t to, int qhead_per_kvhead, vector<pair<string, double>> *exec_time) {
  unsigned int kv_head_factor = qhead_per_kvhead;
  unsigned int kv_head = q_head / kv_head_factor;

  ActiFunc sm(nntrainer::ActivationType::ACT_SOFTMAX);

  auto preproc = [&]()
  {
    query.reshape(ml::train::TensorDim({1, to - from, q_head, h_dim}));
    kcache.reshape(ml::train::TensorDim({1, to, kv_head, h_dim}));
    vcache.reshape(ml::train::TensorDim({1, to, kv_head, h_dim}));

    if (to - from != 1)
      query.transpose("1:0:2", query);
    kcache.transpose("1:0:2", kcache);
    vcache.transpose("1:0:2", vcache);

    query.reshape(ml::train::TensorDim({q_head, 1, to - from, h_dim}));
    kcache.reshape(ml::train::TensorDim({kv_head, 1, to, h_dim}));
    vcache.reshape(ml::train::TensorDim({kv_head, 1, to, h_dim}));

    attn_weight.reshape({q_head, 1, to - from, to});
    attn_output.reshape({q_head, 1, to - from, h_dim});
  };

  auto qk_mul = [&]()
  {
    cout << query << endl;
    attn_weight.setZero();
    EXPECT_NO_THROW(query.dotBatched(kcache, attn_weight, false, true));
  };

  auto norm_weight = [&]() { attn_weight.multiply_i(1 / sqrt((float)h_dim)); };

  auto calc_mask = [&]()
  {
    if (!from) {
      unsigned int mask_size = attn_weight.getDim().width();
      unsigned int mask_dim_height = mask_size;
      unsigned int mask_dim_width = mask_size;

      nntrainer::Tensor causal_mask(ml::train::TensorDim{
        1, 1, mask_size, mask_size, attn_weight.getTensorType()});

      causal_mask.setZero();

  #ifdef ENABLE_FP16
  #define _MASK_NUM -1e4
  #else
  #define _MASK_NUM -1e10
  #endif

      for (unsigned int i = 0; i < mask_dim_height; ++i) {
        for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
          causal_mask.setValue(0, 0, i, j, _MASK_NUM);
        }
      }
      attn_weight.add_i(causal_mask);
    };
  };

  auto softmax = [&]()
  {
    sm.run_fn(attn_weight, attn_weight);
  };

  auto attn_v_mul = [&]()
  {
    EXPECT_NO_THROW(attn_weight.dotBatched(vcache, attn_output));
  };

  auto postproc = [&]() {
    if (to - from != 1) {
      attn_output.reshape(ml::train::TensorDim({1, q_head, to - from, h_dim}));
      attn_output.transpose("1:0:2", attn_output);
    }
    attn_output.reshape({to - from, 1, 1, h_dim * q_head});
  };

  if (exec_time)
  {
    exec_time->push_back({"preproc", check_time(preproc)});
    exec_time->push_back({"q * k", check_time(qk_mul)});
    exec_time->push_back({"norm_weight", check_time(norm_weight)});
    exec_time->push_back({"calc_mask", check_time(calc_mask)});
    exec_time->push_back({"softmax", check_time(softmax)});
    exec_time->push_back({"attn * v", check_time(attn_v_mul)});
    exec_time->push_back({"postproc", check_time(postproc)});
  }
  else
  {
    preproc();
    qk_mul();
    norm_weight();
    calc_mask();
    softmax();
    attn_v_mul();
    postproc();
  }

  return 0;
}

struct RepackTask {
  float *dst;
  const float *src_B;
  int row_tile_start;
  int tile_rows;
  int n;
  int chunk_size;
  int N;
};

void repack_B_tile_task(const RepackTask &task) {
  for (int row = 0; row < task.tile_rows; ++row) {
    const float *src = task.src_B +
                       (task.row_tile_start + row) * task.N * task.chunk_size +
                       task.n * task.chunk_size;
    float *dst = task.dst + row * task.chunk_size;
    std::memcpy(dst, src, sizeof(float) * task.chunk_size);
  }
}

/*
void compute_grouped_dot_with_pool(const float *A, const float *B,
                                   float *output, int num_rows, int N,
                                   int chunk_size, int group_size,
                                   int tile_size = 64) {
  // const bool use_repacked = group_size >= 4;
  const bool use_repacked = false;
  const int group_stride = group_size * chunk_size;
  const int row_stride = N * chunk_size;
  const int tile_count = (num_rows + tile_size - 1) / tile_size;

  std::vector<std::vector<float>> B_tile_buffers;
  if (use_repacked) {
    B_tile_buffers.resize((num_rows + tile_size - 1) / tile_size);
    for (auto &buf : B_tile_buffers) {
      buf.resize(tile_size * chunk_size);
    }
  }

  for (int n = 0; n < N; ++n) {
    std::vector<std::future<void>> futures;
    if (use_repacked) {
      for (int t = 0; t < tile_count; ++t) {
        int row_tile_start = t * tile_size;
        int tile_rows = std::min(tile_size, num_rows - row_tile_start);
        float *dst = B_tile_buffers[t].data();
        RepackTask task = {dst, B, row_tile_start, tile_rows, n, chunk_size, N};
        futures.emplace_back(nntrainer::Tensor::getThreadPool().submit_task(
          [task]() { repack_B_tile_task(task); }));
      }
      for (auto &fut : futures)
        fut.get();
    }

    for (int g = 0; g < group_size; ++g) {
      const float *a_ptr = A + n * group_stride + g * chunk_size;
      for (int t = 0; t < tile_count; ++t) {
        int row_tile_start = t * tile_size;
        int tile_rows = std::min(tile_size, num_rows - row_tile_start);

        const float *b_ptr =
          use_repacked ? B_tile_buffers[t].data()
                       : B + row_tile_start * row_stride + n * chunk_size;

        for (int row = 0; row < tile_rows; ++row) {
          const float *b_row = b_ptr + row * row_stride;
          __m256 sum = _mm256_setzero_ps();
          int k = 0;
          for (; k + 7 < chunk_size; k += 8) {
            __m256 va = _mm256_loadu_ps(a_ptr + k);
            __m256 vb = _mm256_loadu_ps(b_row + k);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
          }

          float tail_sum = 0.0f;
          for (; k < chunk_size; ++k)
            tail_sum += a_ptr[k] * b_row[k];

          __m128 low = _mm256_castps256_ps128(sum);
          __m128 high = _mm256_extractf128_ps(sum, 1);
          __m128 sum128 = _mm_add_ps(low, high);
          sum128 = _mm_hadd_ps(sum128, sum128);
          sum128 = _mm_hadd_ps(sum128, sum128);
          float vec_sum = _mm_cvtss_f32(sum128);

          output[((row_tile_start + row) * group_size + g) * N + n] =
            vec_sum + tail_sum;
        }
      }
    }
  }
}

*/

TEST(nntrainer_TensorDim, mha_01_p) {
  int q_head = 24;
  int h_dim = 128;
  int tile_size = 64;
  bool tok_gen = false;
  int to = 1024;
  int from = tok_gen ? to - 1 : 0;
  int qhead_per_kv = 1;
  vector<pair<string, double>> exec_time;

  auto t = AttnBlockTestData(q_head, h_dim, tile_size, tok_gen, from, to,
                             qhead_per_kv);

  auto start_time = std::chrono::high_resolution_clock::now();

  auto res = _mha(t.input, t.kcache, t.vcache, t.attn_weight, t.output, q_head,
                  h_dim, tile_size, tok_gen, from, to, 1, &exec_time);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto attn_block_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);

  for (int i = 0; i < exec_time.size(); ++i)
  {
    printf("%15s: %04.2f ms\n", exec_time[i].first.c_str(), exec_time[i].second);
  }

  std::cout << "attn_output" << std::endl;
  std::cout << t.output << std::endl;

  std::cout << "***** Dot Batched Time: " << attn_block_time.count() << " ms\n";
  std::cout << "--------------------------" << std::endl;

  EXPECT_EQ(res, 0);
}

TEST(nntrainer_TensorDim, mha_02_p) {
  int q_head = 24;
  int h_dim = 128;
  int tile_size = 64;
  bool tok_gen = true;
  int to = 1024;
  int from = tok_gen ? to - 1 : 0;
  int qhead_per_kv = 1;
  vector<pair<string, double>> exec_time;

  auto t = AttnBlockTestData(q_head, h_dim, tile_size, tok_gen, from, to,
                             qhead_per_kv);

  auto start_time = std::chrono::high_resolution_clock::now();

  auto res = _mha(t.input, t.kcache, t.vcache, t.attn_weight, t.output, q_head,
                  h_dim, tile_size, tok_gen, from, to, 1, &exec_time);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto attn_block_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);

  for (int i = 0; i < exec_time.size(); ++i) {
    printf("%15s: %04.2f ms\n", exec_time[i].first.c_str(),
           exec_time[i].second);
  }

  std::cout << "attn_output" << std::endl;
  std::cout << t.output << std::endl;

  std::cout << "***** Dot Batched Time: " << attn_block_time.count() << " ms\n";
  std::cout << "--------------------------" << std::endl;

  EXPECT_EQ(res, 0);
}

TEST(nntrainer_TensorDim, mha_03_p) {
  int q_head = 24;
  int h_dim = 128;
  int tile_size = 64;
  bool tok_gen = true;
  int to = 10240;
  int from = tok_gen ? to - 1 : 0;
  int qhead_per_kv = 1;
  vector<pair<string, double>> exec_time;

  auto t = AttnBlockTestData(q_head, h_dim, tile_size, tok_gen, from, to,
                             qhead_per_kv);

  auto start_time = std::chrono::high_resolution_clock::now();

  auto res = _mha(t.input, t.kcache, t.vcache, t.attn_weight, t.output, q_head,
                  h_dim, tile_size, tok_gen, from, to, 1, &exec_time);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto attn_block_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);

  for (int i = 0; i < exec_time.size(); ++i) {
    printf("%15s: %04.2f ms\n", exec_time[i].first.c_str(),
           exec_time[i].second);
  }

  std::cout << "attn_output" << std::endl;
  std::cout << t.output << std::endl;

  std::cout << "***** Dot Batched Time: " << attn_block_time.count() << " ms\n";
  std::cout << "--------------------------" << std::endl;

  EXPECT_EQ(res, 0);
}

TEST(nntrainer_TensorDim, dotBatched_01_p) {

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP32;

  unsigned int num_head = 24;
  unsigned int head_dim = 128;
  unsigned int sequence_len = 10240;
  unsigned int tile_size = 64;
  bool tg = true;

  unsigned int seq = tg ? 1 : sequence_len;

  unsigned int num_key_value_head = 1;
  unsigned int num_gqa_head = num_head / num_key_value_head;

  // nntrainer::Tensor input(1, 1, sequence_len, num_head * head_dim, t_type);
  nntrainer::Tensor input_org(1, 1, seq, num_head * head_dim, t_type);
  nntrainer::Tensor kcache_org(1, 1, sequence_len, head_dim * num_gqa_head,
                               t_type);

  nntrainer::Tensor input(1, 1, seq, num_head * head_dim, t_type);
  nntrainer::Tensor kcache(1, 1, sequence_len, head_dim * num_gqa_head, t_type);

  kcache_org.setRandUniform(-0.5, 0.0);
  input_org.setRandUniform(-0.5, 0.0);
  // kcache_org.setValue(0.01f);
  // input_org.setValue(1.0f);
  nntrainer::Tensor output(num_head, 1, seq, sequence_len, t_type);
  nntrainer::Tensor output2(num_head, 1, seq, sequence_len, t_type);
  float *out = nullptr, *out2 = nullptr;

  {
    output.setZero();
    input.copy(input_org);
    kcache.copy(kcache_org);

    auto start_time = std::chrono::high_resolution_clock::now();

    input.reshape(ml::train::TensorDim({1, seq, num_head, head_dim}));
    kcache.reshape(
      ml::train::TensorDim({1, sequence_len, num_gqa_head, head_dim}));

    // input.transpose("1:0:2", input);
    kcache.transpose("1:0:2", kcache);

    input.reshape(ml::train::TensorDim({num_head, 1, seq, head_dim}));

    kcache.reshape(
      ml::train::TensorDim({num_gqa_head, 1, sequence_len, head_dim}));

    std::cout << "input: " << std::endl;
    std::cout << input << std::endl;

    std::cout << "kcache: " << std::endl;
    std::cout << kcache << std::endl;

    EXPECT_NO_THROW(input.dotBatched(kcache, output, false, true));

    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << output << std::endl;

    out = output.getData<float>();
    std::cout << " transposed direction out of dotbatched -------------- "
              << std::endl;

    auto dotbatch_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time);
    std::cout << "***** Dot Bacthed Time: " << dotbatch_duration.count()
              << " ms\n";
    std::cout << "***** Throughput: "
              << (double)sequence_len * num_head * num_key_value_head *
                   head_dim / (dotbatch_duration.count() * 1e6)
              << " GFLOPs\n";
    std::cout << "--------------------------" << std::endl;
  }

  /*
  {
    input.copy(input_org);
    kcache.copy(kcache_org);
    output.setZero();

    start_time = std::chrono::high_resolution_clock::now();

    input.reshape(ml::train::TensorDim({1, seq, num_head, head_dim}));
    kcache.reshape(
      ml::train::TensorDim({1, sequence_len, num_gqa_head, head_dim}));

    input.transpose("1:0:2", input);
    kcache.transpose("1:0:2", kcache);

    input.reshape(ml::train::TensorDim({num_head, 1, seq, head_dim}));

    kcache.reshape(
      ml::train::TensorDim({num_gqa_head, 1, sequence_len, head_dim}));

    (void)nntrainer::Tensor::getThreadPool().submit_blocks(
      0, num_head, [&](const std::size_t start, const std::size_t end) {
        for (std::size_t i = start; i < end; ++i) {
          const nntrainer::Tensor this_b = input.getBatchSlice(i, 1);
          nntrainer::Tensor m_b = kcache.getBatchSlice(i, 1);
          nntrainer::Tensor result_b = output.getBatchSlice(i, 1);
          this_b.dot(m_b, result_b, false, true, 1.0);
        }
      });

    nntrainer::Tensor::getThreadPool().wait();

    end_time = std::chrono::high_resolution_clock::now();

    std::cout << output << std::endl;

    auto thread_pool_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time);

    std::cout << "***** Dot Bacthed Thread Time: "
              << thread_pool_duration.count() << " ms\n";
    std::cout << "***** Throughput: "
              << (double)sequence_len * num_head * num_key_value_head *
                   head_dim / (thread_pool_duration.count() * 1e6)
              << " GFLOPs\n";

    std::cout << "--------------------------" << std::endl;
  }

  */

  {
    output2.setZero();
    auto start_time = std::chrono::high_resolution_clock::now();
    multiply_and_reduce_chunks(
      input_org.getData<float>(), kcache_org.getData<float>(),
      output2.getData<float>(), sequence_len, num_head / num_key_value_head,
      head_dim, num_key_value_head, tile_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    // double ms = std::chrono::duration<double,
    // std::milli>(end_time-start_time).count();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                    start_time)
                .count();

    // output2 = output2.transpose("0:2:1", output2);
    // output2.reshape({num_head, 1, seq, sequence_len});
    out2 = output2.getData<float>();
    std::cout << "***** Time: " << ms << " ms\n";
    std::cout << "***** Throughput: "
              << (double)sequence_len * num_head * num_key_value_head *
                   head_dim / (ms * 1e6)
              << " GFLOPs\n";

    std::cout << output2 << std::endl;
  }

  for (int i = 0; i < num_head * 1 * seq * sequence_len; ++i) {
    EXPECT_NEAR(out[i], out2[i], 1e-5);
  }

  /*
  {
    std::cout << "--------------------------" << std::endl;
    output.setZero();
    start_time = std::chrono::high_resolution_clock::now();
    compute_grouped_dot_with_pool(
      input_org.getData<float>(), kcache_org.getData<float>(),
      output.getData<float>(), sequence_len, num_head / num_key_value_head,
      head_dim, num_key_value_head, tile_size);

    end_time = std::chrono::high_resolution_clock::now();
    // double ms = std::chrono::duration<double,
    // std::milli>(end_time-start_time).count();
    ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
  start_time) .count(); std::cout << "***** Time: " << ms << " ms\n"; std::cout
  << "***** Throughput: "
              << (double)sequence_len * num_head * num_key_value_head * head_dim
  / (ms * 1e6)
              << " GFLOPs\n";
    std::cout << output << std::endl;
  }
  */
}
