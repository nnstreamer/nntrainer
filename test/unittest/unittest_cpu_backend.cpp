#include <gtest/gtest.h>
#include "nntrainer_test_util.h"
#include <cpu_backend.h>
#include <vector>
#include <random>

template <typename T>
static inline std::vector<T> generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<T> vec(size);
    for (auto& val : vec){
        val = static_cast<T>(dist(gen));
    }
    return vec;
}

template <typename T>
static inline std::vector<T> generate_homogeneous_vector(size_t size, T value){
    std::vector<T> vec(size);
    for (auto& val : vec){
        val = value;
    }
    return vec;
}

typedef struct {
     union {
        struct {
            int16_t d;    // super-block scale for quantized scales
            int16_t dmin; // super-block scale for quantized mins
        };
        uint32_t dm;
    };
    uint8_t scales[12]; // scales and mins, quantized with 6 bits
    uint8_t qs[256/2];           // 4--bit quants
} block_q4_K_static;


TEST(nntrainer_cpu_backend_standalone, ele_add) {
    const unsigned int TEST_SIZE = 100;
    float alpha = 1.F;
    float beta = 0.F;
    unsigned int i_stride = 1;
    unsigned int o_stride = 1;

    std::vector<float> lhs = generate_random_vector<float>(TEST_SIZE);
    std::vector<float> rhs = generate_random_vector<float>(TEST_SIZE);
    std::vector<float> dst(TEST_SIZE);

    const float* lhs_ptr = (const float*) lhs.data();
    const float* rhs_ptr = (const float*) rhs.data();
    float* dst_ptr = (float*) dst.data();

    nntrainer::ele_add(TEST_SIZE, lhs_ptr, rhs_ptr, dst_ptr, alpha, beta, i_stride, o_stride);

    for (unsigned int i = 0; i < TEST_SIZE; ++i) {
        EXPECT_EQ(dst[i], lhs[i] + rhs[i]);
    }
}

TEST(nntrainer_cpu_backend_standalone, q4_K_GEMM) {
    const unsigned int M = 8;
    const unsigned int K = 16;
    const unsigned int N = 32;
    
    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 1.0f);
    std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.5F);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    nntrainer::sgemm(0/*ROW MAJOR*/, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, N, 0.F, ref_dst_ptr, N);

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_static);
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is an invalid weight data. Needs fix.
    std::vector<char> tmp_qWeight = std::vector<char>(data_size); 
    char* tmp_qWeight_ptr = (char*) tmp_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) tmp_qWeight_ptr, K, N, nullptr);
    // nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) tmp_qWeight, N, K, nullptr);
    ///@todo HOW BIG IS THE WEIGHT BUFFER? (N * K / len(q4_K) * sizeof(q4_K) bytes)?

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    ///@note do something like : nntrainer::repack_q4_K_to_q4_K_8x8(tmp_qWeight, N, K, q4_K_8x8_weight);

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    ///@note do something like : nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) tmp_qWeight, N, dst_ptr, N);

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have error, but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp

    /* 
        Room for optimization
        
        1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
            - PRO : We can save the time for repacking the weight
            - CON : We need to save the weight in two different formats (q4_K and q4_K_8x8), and such kernel works for specific HWs.
        2. Pre-allocation of runtime quantized activation buffer
        3. ???
    */
}



int main(int argc, char **argv) {
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