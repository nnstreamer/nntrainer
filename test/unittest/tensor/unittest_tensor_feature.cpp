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

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <algorithm>
#include <float_tensor.h>
#include <fstream>
#include <immintrin.h>
#include <nntrainer_error.h>
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

// N is size of B and size of A is N*group_size

void multiply_and_reduce_chunks(const float *A, const float *B, float *output,
                                int num_rows, int N, int chunk_size,
                                int group_size, int tile_size = 64) {

  // const bool use_repacked = group_size >=4;
  const bool use_repacked = false;
  float *B_tile = use_repacked ? new float[tile_size * chunk_size] : nullptr;

  const int group_stride = group_size * chunk_size;
  const int row_stride = N * chunk_size;

  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < group_size; ++g) {
      const float *a_ptr = A + n * group_stride + g * chunk_size;

      for (int row_tile_start = 0; row_tile_start < num_rows;
           row_tile_start += tile_size) {
        int tile_rows = std::min(tile_size, num_rows - row_tile_start);

        const float *b_ptr = nullptr;
        if (use_repacked) {
          repack_B_tile(B, B_tile, row_tile_start, tile_rows, n, chunk_size, N);
          b_ptr = B_tile;
        }

        for (int row = 0; row < tile_rows; ++row) {
          const float *b_row =
            use_repacked
              ? b_ptr + row * chunk_size
              : B + (row_tile_start + row) * row_stride + n * chunk_size;

          __m256 sum = _mm256_setzero_ps();
          int k = 0;

          for (; k + 7 < chunk_size; k += 8) {
            __m256 va = _mm256_loadu_ps(a_ptr + k);
            __m256 vb = _mm256_loadu_ps(b_row + k);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
          }

          float tail_sum = 0.0f;
          for (; k < chunk_size; ++k) {
            tail_sum += a_ptr[k] * b_row[k];
          }

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

  if (B_tile)
    delete[] B_tile;
};

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

  unsigned int num_key_value_head = 2;
  unsigned int num_gqa_head = num_head / num_key_value_head;

  // nntrainer::Tensor input(1, 1, sequence_len, num_head * head_dim, t_type);
  nntrainer::Tensor input_org(1, 1, seq, num_head * head_dim, t_type);
  nntrainer::Tensor kcache_org(1, 1, sequence_len, head_dim * num_gqa_head,
                               t_type);

  nntrainer::Tensor input(1, 1, seq, num_head * head_dim, t_type);
  nntrainer::Tensor kcache(1, 1, sequence_len, head_dim * num_gqa_head, t_type);

  kcache_org.setRandUniform(-0.5, 0.0);
  input_org.setRandUniform(-0.5, 0.0);

  nntrainer::Tensor output(num_head, 1, seq, sequence_len, t_type);
  output.setZero();
  input.copy(input_org);
  kcache.copy(kcache_org);

  auto start_time = std::chrono::high_resolution_clock::now();

  input.reshape(ml::train::TensorDim({1, seq, num_head, head_dim}));
  kcache.reshape(
    ml::train::TensorDim({1, sequence_len, num_gqa_head, head_dim}));

  input.transpose("1:0:2", input);
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

  float *dd = output.getData<float>();
  std::cout << " transposed direction out of dotbatched -------------- "
            << std::endl;
  std::cout << dd[0] << " " << dd[10240] << " " << dd[10240 * 2] << std::endl;

  auto dotbatch_duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                          start_time);

  std::cout << "***** Dot Bacthed Time: " << dotbatch_duration.count()
            << " ms\n";
  std::cout << "***** Throughput: "
            << (double)sequence_len * num_head * num_key_value_head * head_dim /
                 (dotbatch_duration.count() * 1e6)
            << " GFLOPs\n";
  std::cout << "--------------------------" << std::endl;

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

  std::cout << "***** Dot Bacthed Thread Time: " << thread_pool_duration.count()
            << " ms\n";
  std::cout << "***** Throughput: "
            << (double)sequence_len * num_head * num_key_value_head * head_dim /
                 (thread_pool_duration.count() * 1e6)
            << " GFLOPs\n";

  std::cout << "--------------------------" << std::endl;
  output.setZero();
  start_time = std::chrono::high_resolution_clock::now();
  multiply_and_reduce_chunks(
    input_org.getData<float>(), kcache_org.getData<float>(),
    output.getData<float>(), sequence_len, num_head / num_key_value_head,
    head_dim, num_key_value_head, tile_size);

  end_time = std::chrono::high_resolution_clock::now();
  // double ms = std::chrono::duration<double,
  // std::milli>(end_time-start_time).count();
  auto ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
      .count();
  std::cout << "***** Time: " << ms << " ms\n";
  std::cout << "***** Throughput: "
            << (double)sequence_len * num_head * num_key_value_head * head_dim /
                 (ms * 1e6)
            << " GFLOPs\n";

  std::cout << output << std::endl;

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
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
      .count();
  std::cout << "***** Time: " << ms << " ms\n";
  std::cout << "***** Throughput: "
            << (double)sequence_len * num_head * num_key_value_head * head_dim /
                 (ms * 1e6)
            << " GFLOPs\n";
  std::cout << output << std::endl;
}
