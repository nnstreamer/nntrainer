#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <cmath>
#include <string>
#include <limits>
#include <sp_thread_pool.hpp>
#include "nntrainer_test_util.h"
#include <cpu_backend.h>

#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

#include "ggml_interface.h"

// #include "../../nntrainer/tensor/cpu_backend/ggml_interface/ggml_interface.h"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;



/**
 * @brief Experimental. Suggests the hardware to huge TLB and sequential access
 */
static void _warm_pages(void *buf, size_t bytes) {
    if (madvise(buf, bytes, MADV_HUGEPAGE) != 0) {
        std::cerr << "madvise(HUGEPAGE) failed: " << std::strerror(errno) << "\n";
    }
    if (posix_madvise(buf, bytes, POSIX_MADV_SEQUENTIAL) != 0) {
        std::cerr << "posix_madvise(SEQUENTIAL) failed: " << std::strerror(errno) << "\n";
    }
    long page = sysconf(_SC_PAGESIZE);
    volatile uint8_t *p = reinterpret_cast<volatile uint8_t*>(buf);
    for (size_t off = 0; off < bytes; off += page) {
        (void)p[off];
    }
}

/**
 * @brief Experimental. Warms up the TLB pages by accessing each element in the array 
 */
void *align_and_warm(void *current_buf, size_t bytes) {
    size_t page = sysconf(_SC_PAGESIZE);
    size_t rounded = (bytes + page - 1) / page * page;
    void *raw;
    if (posix_memalign(&raw, page, rounded)) {
        perror("posix_memalign");
        std::exit(1);
    }
    float *new_buf = static_cast<float *>(raw);
    std::memcpy(new_buf, current_buf, bytes);

    _warm_pages(new_buf, rounded);
    return new_buf;
}

template <typename T, bool random_init = false>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

#define QK4_0 32
/**
 * @brief q4_0 block
 *
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;
/**
 * @brief q4_K block
 *
 */
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
/**
 * @brief q8_K block
 *
 */
typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;
/**
 * @brief q4_Kx8 block
 *
 */
struct block_q4_Kx8_testonly {
  int16_t d[8];       // super-block scale for quantized scales
  int16_t dmin[8];    // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

#define QK_K 256
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  uint16_t d;               // super-block scale
} block_q6_K_testonly;

float test_gemm_q4_0(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, const uint32_t task, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  int64_t q4_0_num_blocks = (K * N) / q4_0_block_size;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight = std::vector<char>(q4_0_data_size);

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  char *q4_0_offline_qWeight_ptr = (char *)q4_0_offline_qWeight.data();
  nntrainer::quantize_q4_0(weights, (void *)q4_0_offline_qWeight_ptr, N, K,
                           nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> q4_0_repacked_qWeight = std::vector<char>(q4_0_data_size);
  nntrainer::repack_q4_0_to_q4_0_8(q4_0_repacked_qWeight.data(),
                                   q4_0_offline_qWeight_ptr, q4_0_data_size, N,
                                   K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::__ggml_q4_0_8x8_q8_0_GEMM(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N, task);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_0: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  return dt.count() / 1000;
}

float test_gemm_q4_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, const uint32_t task, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * N / q4_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q4_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> repacked_qWeight = std::vector<char>(data_size);
  nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,
                                   data_size, N, K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::__ggml_q4_K_8x8_q8_K_GEMM(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                       N, dst.data(), N, task);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  return dt.count() / 1000;
}

float test_gemm_q6_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, const uint32_t task, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * N / q6_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q6_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::__ggml_gemm_q6_K(M, N, K, activations, K, (void *)offline_qWeight_ptr, N,
                       dst.data(), N, task);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q6_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  return dt.count() / 1000;
}


int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <mode>\n"
                  << "  mode: q6k | q4k | q40\n";
        return -1;
    }
    unsigned int M = std::stoul(argv[1]);
    unsigned int K = std::stoul(argv[2]);
    unsigned int N = std::stoul(argv[3]);
    std::string mode = argv[4];

    auto activation = generate_random_vector<float>(static_cast<size_t>(M) * K);
    auto weight     = generate_random_vector<float>(static_cast<size_t>(N) * K);
    std::vector<float> dst(M * N);

    const int runs = 3;
    std::vector<double> times;

    SP::ThreadPool::soft_boot();

    std::vector<unsigned int> task_counts = {1, 2, 4, 8, 16, 32, 64};

    unsigned int single_thread_time;
    double best_speed_up = 1.0;
    unsigned int best_tasks = 1;
    unsigned int best_time;

    std::cout << "\nBenchmarking for mode '" << mode << "' (" \
                "M=" << M << ", K=" << K << ", N=" << N << ")\n";
    std::cout << "Each test will be repeated " << runs << " times to collect reliable statistics\n";

    bool to_milli = false;

    for (unsigned int task: task_counts) {
        times.clear();
        for (int i = 0; i < runs; ++i) {
            std::fill(dst.begin(), dst.end(), 0.0f);
            size_t measured_time;
            if (mode == "q6k") {
                measured_time = test_gemm_q6_K(M, K, N, weight.data(), activation.data(), dst, task, false);
            } else if (mode == "q4k") {
                measured_time = test_gemm_q4_K(M, K, N, weight.data(), activation.data(), dst, task, false);
            } else if (mode == "q40") {
                measured_time = test_gemm_q4_0(M, K, N, weight.data(), activation.data(), dst, task, false);
            } else {
                std::cerr << "Unknown mode: " << mode << "\n";
                return -1;
            }
            times.push_back(measured_time);
            // std::cout << "Run " << (i+1) << ": " << measured_time << " us; " << measured_time / 1000 << " ms\n";
        }

        // If the numbers are too big, convert to milliseconds
        if (times[0] > 50000 || to_milli) {
            to_milli = true;
            for (size_t i = 0; i < times.size(); i++) times[i] = times[i] / 1000;
        }

        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / runs;
        double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / runs - mean * mean);

        if (task == 1) {
            single_thread_time = mean;
            best_time = single_thread_time;
        } else {
            if (single_thread_time / mean > best_speed_up) {
                best_speed_up = single_thread_time / mean;
                best_tasks = task;
                best_time = mean;
            }
        }
        
        std::cout << task << " task: " << "  Mean    : " << mean << (to_milli ? " ms " : " us")
                << "  Std Dev : " << stddev << (to_milli ? " ms \n" : " us\n");
    }

    std::cout << "Fastest time at " << best_time << (to_milli ? " ms " : " us ") << " with a speedup of " << best_speed_up
              << "x, using " << best_tasks << " tasks." << std::endl << std::endl;

    return 0;
}
