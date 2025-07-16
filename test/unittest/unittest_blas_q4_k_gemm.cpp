#include "nntrainer_test_util.h"
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include "ggml.h"

#include <gtest/gtest.h>

#include <numeric>

using namespace nntrainer;

typedef struct {
  union {
    struct {
      int16_t d;    // super-block scale for quantized scales
      int16_t dmin; // super-block scale for quantized mins
    };
    uint32_t dm;
  };
  uint8_t scales[12];  // scales and mins, quantized with 6 bits
  uint8_t qs[256 / 2]; // 4--bit quants
} block_q4_K_testonly;

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(src[i * N + j] - src2[i * N + j]));
      err_sum += std::abs(src[i * N + j] - src2[i * N + j]);
    }
  }
  // std::cout << "err_sum : " << err_sum << std::endl;
  return max_diff;
}

float compute_mse(const uint32_t M, const uint32_t N, float *ref_dst,
                  const size_t ref_dst_sz, float *dst, const size_t dst_sz,
                  bool print = false) {
  auto mean_squared_error = mse<float, float>(ref_dst, dst, M * N);
  auto cos_sim = cosine_similarity(ref_dst, dst, M * N);
  auto max_differ = find_max_diff(ref_dst, dst, M, N);

  auto sum = std::accumulate(dst, dst + dst_sz, 0.0);
  auto sum_gt = std::accumulate(ref_dst, ref_dst + ref_dst_sz, 0.0);
  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

float compute_mse(const uint32_t M, const uint32_t N,
                  std::vector<float> &ref_dst, std::vector<float> &dst,
                  bool print = false) {
  auto mean_squared_error =
    mse<float, float>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}

#if 1
template <typename T, bool random_init = false>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

static void run_q_4K_gemm2_test(const uint32_t M, const uint32_t K,
                                const uint32_t N) {
  nntrainer::init_backend();

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  auto debug_print_beg_end = [M, K, N](const float *const data,
                                       const uint32_t count = 12) {
    std::cout << "[";
    for (unsigned int i = 0; i < count; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "][";
    for (unsigned int i = M * N - count; i < M * N; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "]";
  };

  static constexpr uint32_t run_count = 16;

  std::vector<float> activation = generate_random_vector<float, false>(M * K);
  std::vector<float> weight = generate_random_vector<float, false>(N * K);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> cpu_q4_dst(M * N, 0.0f);

  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * N / q4_k_block_size;
  data_size *= K;

  // Generate result from SGEMM
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  // Initialize data
  void *gpu_q4_dst =
    blas_cc->context_inst_.createSVMRegion(M * N * sizeof(float));

  void *q4_weight_ptr = blas_cc->context_inst_.createSVMRegion(data_size);
  void *q4_weight_repack_ptr =
    blas_cc->context_inst_.createSVMRegion(data_size);

  blas_cc->command_queue_inst_.enqueueSVMMap(q4_weight_ptr, data_size, false);

  float *weights_f32_ptr = weight.data();

  float *activations_f32_ptr =
    (float *)blas_cc->context_inst_.createSVMRegion(M * K * sizeof(float));

  blas_cc->command_queue_inst_.enqueueSVMMap(activations_f32_ptr,
                                             M * K * sizeof(float), false);

  for (unsigned int i = 0; i < M * K; ++i) {
    activations_f32_ptr[i] = activation[i];
  }

  /// Quantize weight data
  nntrainer::quantize_q4_K(weights_f32_ptr, q4_weight_ptr, N, K, nullptr);
  nntrainer::repack_q4_K_to_q4_K_8(q4_weight_repack_ptr, q4_weight_ptr,
                                   data_size, N, K);

  // CPU Q4_K GEMM
  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    nntrainer::gemm_q4_K(M, N, K, activations_f32_ptr, K, q4_weight_repack_ptr,
                         N, cpu_q4_dst.data(), N);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  static constexpr size_t qk_k = 256;
  static constexpr size_t sizeof_block_q8_K = 292;

  unsigned int blocks_per_4_rows = (K + qk_k - 1) / qk_k;
  unsigned int qa_4_rows_size = sizeof_block_q8_K * 4 * blocks_per_4_rows;
  unsigned int M4 = ((M + 3) / 4);

  std::memset(activations_f32_ptr, 0x00, M * K * sizeof(float));

  if (M == 1) {
    ::quantize_row_q8_K((float *)activation.data(), activations_f32_ptr, K);
  } else {
    for (int i = 0; i < static_cast<int>(M4); i++) {
      ::ggml_quantize_mat_q8_K_4x8(
        activation.data() + 4 * i * K,
        reinterpret_cast<char *>(activations_f32_ptr) + i * qa_4_rows_size, K);
    }
  }

  // GPU Q4_K GEMM
  auto t3 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {

    nntrainer::sgemm_q4_k_cl2(M, N, K, q4_weight_repack_ptr, activations_f32_ptr,
                             (float *)gpu_q4_dst);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto gpu_dt = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);

  // Compute raports
  {
    uint32_t first_zero_index = UINT32_MAX;
    int zeros = 0;
    int nans = 0;

    for (uint32_t i = 0; i < M * N; ++i) {
      if (((float *)gpu_q4_dst)[i] == 0) {
        zeros++;
        if (first_zero_index == UINT32_MAX) {
          first_zero_index = i;
        }
      }

      if (std::isnan(((float *)gpu_q4_dst)[i])) {
        nans++;
      }
    }

    const auto mean_squared_error_dst_gpu = compute_mse(
      M, N, (float *)ref_dst.data(), M * N, (float *)gpu_q4_dst, M * N, false);
    const auto mean_squared_error_dst =
      compute_mse(M, N, (float *)ref_dst.data(), M * N,
                  (float *)cpu_q4_dst.data(), M * N, false);

    const auto data_size_mb = data_size / (1024 * 1024.0f);

    std::cout << "Q4_K GEMM : " << M << " x " << K << " x " << N << std::endl;
    std::cout << " - q4_K data size : " << data_size_mb << " [MB]" << std::endl;
    std::cout << " - time : CPU = " << dt.count() / (run_count * 1.0f) << " ms"
              << std::endl;
    std::cout << " - time : GPU = " << gpu_dt.count() / (run_count * 1.0f)
              << " ms" << std::endl;
    std::cout << " - sample : REF = ";
    debug_print_beg_end(ref_dst.data());
    std::cout << std::endl;
    std::cout << " - sample : CPU = ";
    debug_print_beg_end(cpu_q4_dst.data());
    std::cout << std::endl;
    std::cout << " - sample : GPU = ";
    debug_print_beg_end((float *)gpu_q4_dst);
    std::cout << std::endl;
    std::cout << " - zeros : " << zeros << " / " << M * N << " [ "
              << zeros * 100.0f / float(M * N) << " %] - first at [ "
              << first_zero_index << " ]" << std::endl;
    std::cout << " - nans : " << nans << " / " << M * N << " [ "
              << nans * 100.0f / float(M * N) << " %]" << std::endl;
    std::cout << " - MSE : CPU = " << mean_squared_error_dst << std::endl;
    std::cout << " - MSE : GPU = " << mean_squared_error_dst_gpu << std::endl;
  }
}

#define DECLARE_q_4_K_test_M_K_N(M, K, N)                                      \
  TEST(nntrainer_cpu_backend_standalone, run_q_4K_gemm2_##M##_##K##_##N) {         \
    run_q_4K_gemm2_test(M, K, N);                                                   \
  }

// DECLARE_q_4_K_test_M_K_N(1, 768, 1024);

DECLARE_q_4_K_test_M_K_N(256, 1024, 256);
DECLARE_q_4_K_test_M_K_N(3072, 8192, 3072);
DECLARE_q_4_K_test_M_K_N(256, 3072, 8192);
DECLARE_q_4_K_test_M_K_N(256, 8192, 3072);
DECLARE_q_4_K_test_M_K_N(256, 3072, 3072);

// DECLARE_q_4_K_test_M_K_N(256, 256, 3072);
// DECLARE_q_4_K_test_M_K_N(3072, 256, 256);
// DECLARE_q_4_K_test_M_K_N(1, 3072, 128000);
#endif