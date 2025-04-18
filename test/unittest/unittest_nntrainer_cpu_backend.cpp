// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend.cpp
 * @date	03 April 2025
 * @brief	This is unittest for cpu_backend standalone
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include "nntrainer_test_util.h"
#include <cpu_backend.h>
#include <vector>
#include <random>
#include <numeric>

#include <iostream>
#include <chrono>
using std::chrono::nanoseconds; // or microseconds
using std::chrono::microseconds; // or microseconds
using std::chrono::milliseconds; // or microseconds
using std::chrono::seconds; // or microseconds
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;

template <typename T>
static inline std::vector<T> generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F){
    std::random_device rd;
    std::mt19937 gen(42);
    // std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<T> vec(size);
    for (auto& val : vec){
        val = static_cast<T>(dist(gen));
    }
    return vec;
}

template <typename T>
inline std::vector<T> generate_random_positive_vector(size_t size, float min_val = 0.F, float max_val = 0.5F){
    std::random_device rd;
    std::mt19937 gen(42);
    // std::mt19937 gen(rd());
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

template <typename T>
static inline void print_matrix(T* src, int M, int N){
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            std::cout << src[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
inline void print_matrix_partially(T* src, int M, int N, int partial_m = 5, int partial_n = 5, int partial_len = 5){
    for (int k = 0; k < partial_len; ++k){
        std::cout << src[partial_m * N + partial_n + k] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
inline void print_vector_partially(T* src, int init_idx = 5, int partial_len = 5){
    for (int k = 0; k < partial_len; ++k){
        std::cout << src[init_idx + k] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
static inline void print_matrix_partially_n(const std::string &name, const T* src, int M, int N, int partial_m = 5, int partial_n = 5){
    std::cout << name << ":" << std::endl;
    std::cout << "--------------------------" << std::endl;
    for (int i = 0; i < partial_m; ++i) {
        for (int j = 0; j < partial_n; ++j) {
            std::cout << src[i * N + j] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------------------" << std::endl;
}

template <typename T>
static inline double find_max_diff(T* src, T* src2, int M, int N){
    float max_diff = 0;
    double err_sum = 0;
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            max_diff = std::max(max_diff, std::abs(src[i * N + j] - src2[i * N + j]));
            err_sum += std::abs(src[i * N + j] - src2[i * N + j]);
        }
    }
    // std::cout << "err_sum : " << err_sum << std::endl;
    return max_diff;
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
} block_q4_K_testonly;

typedef struct {
    float   d;              // delta
    int8_t  qs[256];       // quants
    int16_t bsums[256/16]; // sum of quants in groups of 16
} block_q8_K_testonly;


struct block_q4_Kx8_testonly {
  int16_t d[8];     // super-block scale for quantized scales
  int16_t dmin[8];  // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

static inline void print_q4_k_block_partially(void* block){
    block_q4_K_testonly* b = (block_q4_K_testonly*) block;
    std::cout << "d : " << b->d << std::endl;
    std::cout << "dmin : " << b->dmin << std::endl;
    // std::cout << "qs : ";
    // for (int i = 0; i < 256/2; ++i){
    //     uint8_t packed_val = b->qs[i];
    //     uint8_t val1 = packed_val & 0x0F;
    //     uint8_t val2 = (packed_val >> 4) & 0x0F;
    //     std::cout << (int)val1 << " " << (int)val2 << " ";
    // }
    std::cout << "qs 5~8 : ";
    for (int i = 5; i < 8; ++i){
        uint8_t packed_val = b->qs[i];
        uint8_t val1 = packed_val & 0x0F;
        uint8_t val2 = (packed_val >> 4) & 0x0F;
        std::cout << (int)val1 << " " << (int)val2 << " ";
    }
    std::cout << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_ele_add) {
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

TEST(nntrainer_cpu_backend_standalone, q4_K_quantization) {
    const unsigned int K = 768;
    const unsigned int N = 512;
    
    std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> weight_tmp(N * K);

    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();

    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;

    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, nullptr);

    nntrainer::dequantize_row_q4_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

    auto mean_squared_error = mse<float, float>( weight.data(), rhs_ptr_tmp, N * K);
    auto cos_sim = cosine_similarity( weight.data(), rhs_ptr_tmp, N * K);
    auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);
    
    const float eps = 1e-5;
    ///@todo Find proper metric and standard to assess
    EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
    EXPECT_NEAR(cos_sim, 0., eps * K * N);
    EXPECT_NEAR(max_differ, 0., eps * K * N);
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_q4_K_GEMM_latencyonly) {
    ///@note A(M, K) * W.T(N, K) = (M, N)

    // const unsigned int M = 8;
    // const unsigned int K = 16;
    // const unsigned int N = 32;
    // const unsigned int M = 512; // = sizez
    // const unsigned int K = 768; // = sizex
    // const unsigned int N = 1024; // = sizey
    const unsigned int M = 256; // = sizez
    const unsigned int K = 1024; // = sizex
    const unsigned int N = 512; // = sizey

    std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    // std::vector<float> activation = generate_random_vector<float>(M * K);
    // std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;
    
    const unsigned int num_blocks = K*N / (2*1024); 
    std::vector<block_q4_Kx8_testonly> repacked_qWeight(num_blocks);
    for (unsigned int i = 0; i < num_blocks; ++i) {
        // for (unsigned int j = 0; j < 8; ++j) {
        //     repacked_qWeight[i].d[j] = i;
        // }
        // for (unsigned int j = 0; j < 8; ++j) {
        //     repacked_qWeight[i].dmin[j] = i;
        // }
        // for (unsigned int j = 0; j < 96; ++j) {
        //     repacked_qWeight[i].scales[j] = i;
        // }
        // for (unsigned int j = 0; j < 1024; ++j) {
        //     // uint8_t packed_val = static_cast<uint8_t>(i + 1 + j % 2) << 4;
        //     // packed_val |= static_cast<uint8_t>(i + 2 + j % 3);
        //     uint8_t packed_val = static_cast<uint8_t>(i + 1) << 4;
        //     packed_val |= static_cast<uint8_t>(i + 2);
        //     repacked_qWeight[i].qs[j] = packed_val;
        // }
        for (unsigned int j = 0; j < 8; ++j) {
            repacked_qWeight[i].d[j] = 2;
        }
        for (unsigned int j = 0; j < 8; ++j) {
            repacked_qWeight[i].dmin[j] = 2;
        }
        for (unsigned int j = 0; j < 96; ++j) {
            repacked_qWeight[i].scales[j] = 2;
        }
        for (unsigned int j = 0; j < 1024; ++j) {
            // uint8_t packed_val = static_cast<uint8_t>(i + 1 + j % 2) << 4;
            // packed_val |= static_cast<uint8_t>(i + 2 + j % 3);
            uint8_t packed_val = static_cast<uint8_t>(2 + 1) << 4;
            packed_val |= static_cast<uint8_t>(2 + 2);
            repacked_qWeight[i].qs[j] = packed_val;
        }
    }

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    // block-params
    // ref)
    //     .blck_size                = QK_K,
    // .type_size                = sizeof(block_q4_K),
    int64_t ne00 = K, ne01 = N, ne02 = 1, ne03 = 1; // weight block params
    int64_t nb00, nb01, nb02, nb03; // weight block params
    int64_t ne10 = K, ne11 = M, ne12 = 1, ne13 = 1; // activation block params
    int64_t nb10, nb11, nb12, nb13; // activation block params
    int64_t ne0 = N, ne1 = M, ne2 = 1, ne3 = 1; // output block params
    int64_t nb0, nb1, nb2, nb3; // output block params
    
    nb00 = sizeof(block_q4_K_testonly); // ggml_type_size(type);
    nb01 = nb00 * (ne00 / /*QK_K*/ 256 );
    nb02 = nb01 * ne01;
    nb03 = nb02 * ne02;
    
    nb10 = sizeof(float);
    nb11 = nb10 * (ne10 / 1);
    nb12 = nb11 * ne11;
    nb13 = nb12 * ne12;

    nb0 = sizeof(float);
    nb1 = nb0 * (ne0 / 1);
    nb2 = nb1 * ne1;
    nb3 = nb2 * ne2;

    printf("nb02 : %ld, nb03 : %ld\n", nb02, nb03);
    printf("nb12 : %ld, nb13 : %ld\n", nb12, nb13);
    printf("nb2 : %ld, nb3 : %ld\n", nb2, nb3);


    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemm_q4_K : " << dt.count()
            << " ns " << std::endl;

    // print_matrix(dst_ptr, M, N);

    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    std::cout << "sum : "<< sum << std::endl;

}

TEST(nntrainer_cpu_backend_standalone, DISABLED_q4_K_GEMV_latencyonly_512) {
    ///@note A(M, K) * W.T(N, K) = (M, N)

    // const unsigned int M = 8;
    // const unsigned int K = 16;
    // const unsigned int N = 32;
    const unsigned int M = 1;
    const unsigned int K = 768;
    const unsigned int N = 512;
    
    std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.5F);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;
    
    const unsigned int num_blocks = K*N / (2*1024); 
    std::vector<block_q4_Kx8_testonly> repacked_qWeight(K*N / (2*1024));
    for (unsigned int i = 0; i < num_blocks; ++i) {
        for (unsigned int j = 0; j < 8; ++j) {
            repacked_qWeight[i].d[j] = 1;
        }
        for (unsigned int j = 0; j < 8; ++j) {
            repacked_qWeight[i].dmin[j] = 1;
        }
        for (unsigned int j = 0; j < 12; ++j) {
            repacked_qWeight[i].scales[j] = 1;
        }
        for (unsigned int j = 0; j < 1024; ++j) {
            repacked_qWeight[i].qs[j] = 1;
        }
    }

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemv_q4_K : " << dt.count()
            << " ns " << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_q4_K_GEMV_3B_1x1440x1440) {
    ///@note A(M, K) * W.T(N, K) = (M, N)
    ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

    const unsigned int M = 1; // = sizez
    const unsigned int K = 1440; // = sizex
    const unsigned int N = 5760; // = sizey
    
    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    // std::vector<float> activation = generate_random_vector<float>(M * K);
    // std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> activation = generate_random_positive_vector<float>(M * K);
    std::vector<float> weight = generate_random_positive_vector<float>(N * K);
    // std::vector<float> activation = set_custom_value<float>(M * K);
    // std::vector<float> weight = set_custom_value<float>(N * K);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is might be an invalid(too huge?) size for weight data. Needs double checking.
    data_size = N * K / q4_k_block_size * q4_k_type_size;
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, /*imatrix*/nullptr);
    ///@note Step1 is validated with unittest TC : q4_k_quantization

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    std::vector<char> repacked_qWeight = std::vector<char>(data_size); 
    nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,  data_size, /*row*/N, /*col*/K); // W is transposed, so it is N*K ?
    ///@note Needs validation!
    ///@note double-check for : row / col order (since is this function consider a transpoed weight? Or is it just generalized for all matrices?)
    ///@note double-check for data_size (temporally allocated the same size with offline_qWeight, but itself is not validated, yet.)
    ///@note super-quick check with gemm op! (but this might make unclear diagnosis : repacking problem? or gemm kernel problem? ...)

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemm_q4_K : " << dt.count()
            << " ns " << std::endl;
    ///@note Needs validation!

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have a huge quantization error(32bit to 4.12bit), but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp
    ///@note Needs validation!
    auto mean_squared_error = mse<float, float>( ref_dst_ptr, dst_ptr, M*N);
    auto cos_sim = cosine_similarity( ref_dst_ptr, dst_ptr, M*N);
    auto max_differ = find_max_diff(ref_dst_ptr, dst_ptr, M, N);
    
    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

    print_matrix_partially<float>(ref_dst_ptr, M, 0); // for it is a vector
    print_matrix_partially<float>(dst_ptr, M, 0);

    print_vector_partially<float>(ref_dst_ptr);
    print_vector_partially<float>(dst_ptr);

    std::cout << "MSE : " << mean_squared_error << ", COS_SIM : " << cos_sim << ", MAX_DIFFER : " << max_differ << ", SUM : " << sum << ", SUM_GT : " << sum_gt << std::endl;

    /* 
        Room for optimization
        
        1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
            - PRO : We can save the time for repacking the weight
            - CON : We need to save the weight in two different formats (q4_K and q4_K_8x8), and such kernel works for specific HWs.
        2. Pre-allocation of runtime quantized activation buffer
        3. ???
    */
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_q4_K_GEMV_3B_1x1440x5760) {
    ///@note A(M, K) * W.T(N, K) = (M, N)
    ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

    const unsigned int M = 1; // = sizez
    const unsigned int K = 1440; // = sizex
    const unsigned int N = 5760; // = sizey
    
    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    // std::vector<float> activation = generate_random_vector<float>(M * K);
    // std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> activation = generate_random_positive_vector<float>(M * K);
    std::vector<float> weight = generate_random_positive_vector<float>(N * K);
    // std::vector<float> activation = set_custom_value<float>(M * K);
    // std::vector<float> weight = set_custom_value<float>(N * K);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is might be an invalid(too huge?) size for weight data. Needs double checking.
    data_size = N * K / q4_k_block_size * q4_k_type_size;
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, /*imatrix*/nullptr);
    ///@note Step1 is validated with unittest TC : q4_k_quantization

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    std::vector<char> repacked_qWeight = std::vector<char>(data_size); 
    nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,  data_size, /*row*/N, /*col*/K); // W is transposed, so it is N*K ?
    ///@note Needs validation!
    ///@note double-check for : row / col order (since is this function consider a transpoed weight? Or is it just generalized for all matrices?)
    ///@note double-check for data_size (temporally allocated the same size with offline_qWeight, but itself is not validated, yet.)
    ///@note super-quick check with gemm op! (but this might make unclear diagnosis : repacking problem? or gemm kernel problem? ...)

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemm_q4_K : " << dt.count()
            << " ns " << std::endl;
    ///@note Needs validation!

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have a huge quantization error(32bit to 4.12bit), but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp
    ///@note Needs validation!
    auto mean_squared_error = mse<float, float>( ref_dst_ptr, dst_ptr, M*N);
    auto cos_sim = cosine_similarity( ref_dst_ptr, dst_ptr, M*N);
    auto max_differ = find_max_diff(ref_dst_ptr, dst_ptr, M, N);
    
    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

    print_matrix_partially<float>(ref_dst_ptr, M, 0); // for it is a vector
    print_matrix_partially<float>(dst_ptr, M, 0);

    print_vector_partially<float>(ref_dst_ptr);
    print_vector_partially<float>(dst_ptr);

    std::cout << "MSE : " << mean_squared_error << ", COS_SIM : " << cos_sim << ", MAX_DIFFER : " << max_differ << ", SUM : " << sum << ", SUM_GT : " << sum_gt << std::endl;

    /* 
        Room for optimization
        
        1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
            - PRO : We can save the time for repacking the weight
            - CON : We need to save the weight in two different formats (q4_K and q4_K_8x8), and such kernel works for specific HWs.
        2. Pre-allocation of runtime quantized activation buffer
        3. ???
    */
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_q4_K_GEMM_3B_1024x1440x1440) {
    ///@note A(M, K) * W.T(N, K) = (M, N)
    ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

    const unsigned int M = 1024; // = sizez
    const unsigned int K = 1440; // = sizex
    const unsigned int N = 1440; // = sizey
    
    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    // std::vector<float> activation = generate_random_vector<float>(M * K);
    // std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> activation = generate_random_positive_vector<float>(M * K);
    std::vector<float> weight = generate_random_positive_vector<float>(N * K);
    // std::vector<float> activation = set_custom_value<float>(M * K);
    // std::vector<float> weight = set_custom_value<float>(N * K);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is might be an invalid(too huge?) size for weight data. Needs double checking.
    data_size = N * K / q4_k_block_size * q4_k_type_size;
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, /*imatrix*/nullptr);
    ///@note Step1 is validated with unittest TC : q4_k_quantization

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    std::vector<char> repacked_qWeight = std::vector<char>(data_size); 
    nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,  data_size, /*row*/N, /*col*/K); // W is transposed, so it is N*K ?
    ///@note Needs validation!
    ///@note double-check for : row / col order (since is this function consider a transpoed weight? Or is it just generalized for all matrices?)
    ///@note double-check for data_size (temporally allocated the same size with offline_qWeight, but itself is not validated, yet.)
    ///@note super-quick check with gemm op! (but this might make unclear diagnosis : repacking problem? or gemm kernel problem? ...)

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemm_q4_K : " << dt.count()
            << " ns " << std::endl;
    ///@note Needs validation!

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have a huge quantization error(32bit to 4.12bit), but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp
    ///@note Needs validation!
    auto mean_squared_error = mse<float, float>( ref_dst_ptr, dst_ptr, M*N);
    auto cos_sim = cosine_similarity( ref_dst_ptr, dst_ptr, M*N);
    auto max_differ = find_max_diff(ref_dst_ptr, dst_ptr, M, N);
    
    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

    print_matrix_partially<float>(ref_dst_ptr, M, 0); // for it is a vector
    print_matrix_partially<float>(dst_ptr, M, 0);

    print_vector_partially<float>(ref_dst_ptr);
    print_vector_partially<float>(dst_ptr);

    std::cout << "MSE : " << mean_squared_error << ", COS_SIM : " << cos_sim << ", MAX_DIFFER : " << max_differ << ", SUM : " << sum << ", SUM_GT : " << sum_gt << std::endl;

    /* 
        Room for optimization
        
        1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
            - PRO : We can save the time for repacking the weight
            - CON : We need to save the weight in two different formats (q4_K and q4_K_8x8), and such kernel works for specific HWs.
        2. Pre-allocation of runtime quantized activation buffer
        3. ???
    */
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_q4_K_GEMM_3B_1024x1440x5760) {
    ///@note A(M, K) * W.T(N, K) = (M, N)
    ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

    const unsigned int M = 1024; // = sizez
    // const unsigned int K = 1024; // = sizex
    const unsigned int K = 1440; // = sizex // K should be multiple of 256 -> needs padding??? this is for the weight as well.....
    const unsigned int N = 5760; // = sizey
    
    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    // std::vector<float> activation = generate_random_vector<float>(M * K);
    // std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> activation = generate_random_positive_vector<float>(M * K);
    std::vector<float> weight = generate_random_positive_vector<float>(N * K);
    // std::vector<float> activation = set_custom_value<float>(M * K);
    // std::vector<float> weight = set_custom_value<float>(N * K);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is might be an invalid(too huge?) size for weight data. Needs double checking.
    // data_size = N * K * q4_k_type_size;
    data_size = ((K + q4_k_block_size - 1) / q4_k_block_size) * N * q4_k_type_size;
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, /*imatrix*/nullptr);
    ///@note Step1 is validated with unittest TC : q4_k_quantization

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    std::vector<char> repacked_qWeight = std::vector<char>(data_size); 
    nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,  data_size, /*row*/N, /*col*/K); // W is transposed, so it is N*K ?
    ///@note Needs validation!
    ///@note double-check for : row / col order (since is this function consider a transpoed weight? Or is it just generalized for all matrices?)
    ///@note double-check for data_size (temporally allocated the same size with offline_qWeight, but itself is not validated, yet.)
    ///@note super-quick check with gemm op! (but this might make unclear diagnosis : repacking problem? or gemm kernel problem? ...)

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemm_q4_K : " << dt.count()
            << " ns " << std::endl;
    ///@note Needs validation!

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have a huge quantization error(32bit to 4.12bit), but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp
    ///@note Needs validation!
    auto mean_squared_error = mse<float, float>( ref_dst_ptr, dst_ptr, M*N);
    auto cos_sim = cosine_similarity( ref_dst_ptr, dst_ptr, M*N);
    auto max_differ = find_max_diff(ref_dst_ptr, dst_ptr, M, N);
    
    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

    print_matrix_partially<float>(ref_dst_ptr, M, 0); // for it is a vector
    print_matrix_partially<float>(dst_ptr, M, 0);

    print_vector_partially<float>(ref_dst_ptr);
    print_vector_partially<float>(dst_ptr);

    std::cout << "MSE : " << mean_squared_error << ", COS_SIM : " << cos_sim << ", MAX_DIFFER : " << max_differ << ", SUM : " << sum << ", SUM_GT : " << sum_gt << std::endl;

    /* 
        Room for optimization
        
        1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
            - PRO : We can save the time for repacking the weight
            - CON : We need to save the weight in two different formats (q4_K and q4_K_8x8), and such kernel works for specific HWs.
        2. Pre-allocation of runtime quantized activation buffer
        3. ???
    */
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_q4_K_GEMM) {
    ///@note A(M, K) * W.T(N, K) = (M, N)
    ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

    // const unsigned int M = 8;
    // const unsigned int K = 16;
    // const unsigned int N = 32;
    const unsigned int M = 512; // = sizez
    const unsigned int K = 768; // = sizex
    const unsigned int N = 1024; // = sizey
    
    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    // std::vector<float> activation = generate_random_positive_vector<float>(M * K);
    // std::vector<float> weight = generate_random_positive_vector<float>(N * K);
    std::vector<float> activation = generate_random_vector<float>(M * K);
    std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is might be an invalid(too huge?) size for weight data. Needs double checking.
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, N, K, /*imatrix*/nullptr);
    // nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, /*imatrix*/nullptr);
    ///@note Step1 is validated with unittest TC : q4_k_quantization
    // print_q4_k_block_partially(offline_qWeight_ptr);

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    std::vector<char> repacked_qWeight = std::vector<char>(data_size); 
    nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,  data_size, /*row*/N, /*col*/K); // W is transposed, so it is N*K ?

    ///@note Needs validation!
    ///@note double-check for : row / col order (since is this function consider a transpoed weight? Or is it just generalized for all matrices?)
    ///@note double-check for data_size (temporally allocated the same size with offline_qWeight, but itself is not validated, yet.)
    ///@note super-quick check with gemm op! (but this might make unclear diagnosis : repacking problem? or gemm kernel problem? ...)

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemm_q4_K : " << dt.count() << " ns " << std::endl;
    ///@note Needs validation!

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have a huge quantization error(32bit to 4.12bit), but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp
    ///@note Needs validation!
    auto mean_squared_error = mse<float, float>( ref_dst_ptr, dst_ptr, M*N);
    auto cos_sim = cosine_similarity( ref_dst_ptr, dst_ptr, M*N);
    auto max_differ = find_max_diff(ref_dst_ptr, dst_ptr, M, N);

    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

    print_matrix_partially<float>(ref_dst_ptr, M, N);
    print_matrix_partially<float>(dst_ptr, M, N);

    print_vector_partially<float>(ref_dst_ptr);
    print_vector_partially<float>(dst_ptr);

    // EXPECTED SUM : 805306368(GT) VS 805593600.00
    std::cout << "MSE : " << mean_squared_error << ", COS_SIM : " << cos_sim << ", MAX_DIFFER : " << max_differ << ", SUM : " << sum << ", SUM_GT : " << sum_gt << std::endl;

    /* 
        Room for optimization
        
        1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
            - PRO : We can save the time for repacking the weight
            - CON : We need to save the weight in two different formats (q4_K and q4_K_8x8), and such kernel works for specific HWs.
        2. Pre-allocation of runtime quantized activation buffer
        3. ???
    */
}

TEST(nntrainer_cpu_backend_standalone, q4_K_GEMM) {
    ///@note A(M, K) * W.T(N, K) = (M, N)
    ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

    // const unsigned int M = 512; // = sizez
    // const unsigned int K = 512; // = sizex
    // const unsigned int N = 512; // = sizey
    // const unsigned int M = 512; // = sizez
    // const unsigned int K = 1024; // = sizex
    // const unsigned int N = 1536; // = sizey

    const unsigned int M = 1024; // = sizez
    const unsigned int K = 3072; // = sizex
    const unsigned int N = 3072; // = sizey

    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    // std::vector<float> activation = generate_random_positive_vector<float>(M * K);
    // std::vector<float> weight = generate_random_positive_vector<float>(N * K);
    std::vector<float> activation = generate_random_vector<float>(M * K);
    std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is might be an invalid(too huge?) size for weight data. Needs double checking.
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, N, K, /*imatrix*/nullptr);
    // nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, /*imatrix*/nullptr);
    ///@note Step1 is validated with unittest TC : q4_k_quantization
    // print_q4_k_block_partially(offline_qWeight_ptr);

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    std::vector<char> repacked_qWeight = std::vector<char>(data_size); 
    nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,  data_size, /*row*/N, /*col*/K); // W is transposed, so it is N*K ?

    ///@note Needs validation!
    ///@note double-check for : row / col order (since is this function consider a transpoed weight? Or is it just generalized for all matrices?)
    ///@note double-check for data_size (temporally allocated the same size with offline_qWeight, but itself is not validated, yet.)
    ///@note super-quick check with gemm op! (but this might make unclear diagnosis : repacking problem? or gemm kernel problem? ...)

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    
    //####################
    //nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) rhs_ptr, N, dst_ptr, N); // It can be used to multiply float wieghts
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N); // It can be used to multiply quantized wieghts
    //####################

    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemm_q4_K : " << dt.count() << " ns " << std::endl
              << std::endl;
    ///@note Needs validation!

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have a huge quantization error(32bit to 4.12bit), but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp
    ///@note Needs validation!
    auto mean_squared_error = mse<float, float>( ref_dst_ptr, dst_ptr, M*N);
    auto cos_sim = cosine_similarity( ref_dst_ptr, dst_ptr, M*N);
    auto max_differ = find_max_diff(ref_dst_ptr, dst_ptr, M, N);

    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

    print_matrix_partially<float>(ref_dst_ptr, M, N);
    print_matrix_partially<float>(dst_ptr, M, N);

    print_vector_partially<float>(ref_dst_ptr);
    print_vector_partially<float>(dst_ptr);

    // EXPECTED SUM : 805306368(GT) VS 805593600.00
    std::cout << "MSE : " << mean_squared_error << ", COS_SIM : " << cos_sim << ", MAX_DIFFER : " << max_differ << ", SUM : " << sum << ", SUM_GT : " << sum_gt << std::endl;

    /* 
        Room for optimization
        
        1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
            - PRO : We can save the time for repacking the weight
            - CON : We need to save the weight in two different formats (q4_K and q4_K_8x8), and such kernel works for specific HWs.
        2. Pre-allocation of runtime quantized activation buffer
        3. ???
    */
}
TEST(nntrainer_cpu_backend_standalone, q4_K_GEMV_512) {
    ///@note A(M, K) * W.T(N, K) = (M, N)
    ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

    const unsigned int M = 1; // = sizez
    const unsigned int K = 768; // = sizex
    const unsigned int N = 512; // = sizey
    // const unsigned int M = 1; // = sizez
    // const unsigned int K = 3072; // = sizex
    // const unsigned int N = 3072; // = sizey

    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    std::vector<float> activation = generate_random_vector<float>(M * K);
    std::vector<float> weight = generate_random_vector<float>(N * K);
    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is might be an invalid(too huge?) size for weight data. Needs double checking.
    data_size = N * K / q4_k_block_size * q4_k_type_size;
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, /*imatrix*/nullptr);
    ///@note Step1 is validated with unittest TC : q4_k_quantization

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    std::vector<char> repacked_qWeight = std::vector<char>(data_size); 
    nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,  data_size, /*row*/N, /*col*/K); // W is transposed, so it is N*K ?
    ///@note Needs validation!
    ///@note double-check for : row / col order (since is this function consider a transpoed weight? Or is it just generalized for all matrices?)
    ///@note double-check for data_size (temporally allocated the same size with offline_qWeight, but itself is not validated, yet.)
    ///@note super-quick check with gemm op! (but this might make unclear diagnosis : repacking problem? or gemm kernel problem? ...)

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemv_q4_K : " << dt.count()
            << " ns " << std::endl;
    ///@note Needs validation!

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have a huge quantization error(32bit to 4.12bit), but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp
    ///@note Needs validation!
    auto mean_squared_error = mse<float, float>( ref_dst_ptr, dst_ptr, M*N);
    auto cos_sim = cosine_similarity( ref_dst_ptr, dst_ptr, M*N);
    auto max_differ = find_max_diff(ref_dst_ptr, dst_ptr, M, N);
    
    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

    print_matrix_partially<float>(ref_dst_ptr, M, 0); // for it is a vector
    print_matrix_partially<float>(dst_ptr, M, 0);

    print_vector_partially<float>(ref_dst_ptr);
    print_vector_partially<float>(dst_ptr);

    // EXPECTED SUM : 805306368(GT) VS 805593600.00
    std::cout << "MSE : " << mean_squared_error << ", COS_SIM : " << cos_sim << ", MAX_DIFFER : " << max_differ << ", SUM : " << sum << ", SUM_GT : " << sum_gt << std::endl;

    /* 
        Room for optimization
        
        1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
            - PRO : We can save the time for repacking the weight
            - CON : We need to save the weight in two different formats (q4_K and q4_K_8x8), and such kernel works for specific HWs.
        2. Pre-allocation of runtime quantized activation buffer
        3. ???
    */
}
TEST(nntrainer_cpu_backend_standalone, DISABLED_q4_K_GEMV_1024) {
    ///@note A(M, K) * W.T(N, K) = (M, N)
    ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

    const unsigned int M = 1; // = sizez
    const unsigned int K = 768; // = sizex
    const unsigned int N = 1024; // = sizey
    
    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 2.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.0F);
    std::vector<float> activation = generate_random_vector<float>(M * K);
    std::vector<float> weight = generate_random_vector<float>(N * K);
    // std::vector<float> activation = generate_random_positive_vector<float>(M * K);
    // std::vector<float> weight = generate_random_positive_vector<float>(N * K);

    std::vector<float> weight_tmp(N * K);
    std::vector<float> ref_dst(M * N);
    std::vector<float> dst(M * N);

    const float* lhs_ptr = (const float*) activation.data();
    const float* rhs_ptr = (const float*) weight.data();
    float* rhs_ptr_tmp = weight_tmp.data();
    float* ref_dst_ptr = (float*) ref_dst.data();
    float* dst_ptr = (float*) dst.data();

    // GROUND TRUTH TRANSB SGEMM for reference
    auto t1 = high_resolution_clock::now();
    nntrainer::sgemm(/*ROW MAJOR*/0, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, K, 0.F, ref_dst_ptr, N);
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "sgemm : " << dt.count()
            << " ns " << std::endl;

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is might be an invalid(too huge?) size for weight data. Needs double checking.
    data_size = N * K / q4_k_block_size * q4_k_type_size;
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, /*imatrix*/nullptr);
    ///@note Step1 is validated with unittest TC : q4_k_quantization

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    std::vector<char> repacked_qWeight = std::vector<char>(data_size); 
    nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,  data_size, /*row*/N, /*col*/K); // W is transposed, so it is N*K ?
    ///@note Needs validation!
    ///@note double-check for : row / col order (since is this function consider a transpoed weight? Or is it just generalized for all matrices?)
    ///@note double-check for data_size (temporally allocated the same size with offline_qWeight, but itself is not validated, yet.)
    ///@note super-quick check with gemm op! (but this might make unclear diagnosis : repacking problem? or gemm kernel problem? ...)

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) repacked_qWeight.data(), N, dst_ptr, N);
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    std::cout << "gemv_q4_K : " << dt.count()
            << " ns " << std::endl;
    ///@note Needs validation!

    // Step4. Compare quantization error
    // 1. Ground Truth VS Q4_K GEMM
    // 2. Q4_K GEMM on the nntrainer VS Q4_K GEMM on the llama.cpp
    ///@note It is quite obvious to have a huge quantization error(32bit to 4.12bit), but the error is expected to be similar to something we can obtain from llama.cpp benchmark-matmult.cpp
    ///@note Needs validation!
    auto mean_squared_error = mse<float, float>( ref_dst_ptr, dst_ptr, M*N);
    auto cos_sim = cosine_similarity( ref_dst_ptr, dst_ptr, M*N);
    auto max_differ = find_max_diff(ref_dst_ptr, dst_ptr, M, N);
    
    auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
    auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);

    print_matrix_partially<float>(ref_dst_ptr, M, 0); // for it is a vector
    print_matrix_partially<float>(dst_ptr, M, 0);

    print_vector_partially<float>(ref_dst_ptr);
    print_vector_partially<float>(dst_ptr);

    std::cout << "MSE : " << mean_squared_error << ", COS_SIM : " << cos_sim << ", MAX_DIFFER : " << max_differ << ", SUM : " << sum << ", SUM_GT : " << sum_gt << std::endl;

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
#ifdef ENABLE_GGML
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
#else
  result = 0;
#endif
  return result;
}