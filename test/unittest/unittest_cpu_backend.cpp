#include <gtest/gtest.h>
#include "nntrainer_test_util.h"
#include <cpu_backend.h>
#include <vector>
#include <random>
#include <iostream>

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
static inline double find_max_diff(T* src, T* src2, int M, int N){
    float max_diff = 0;
    double err_sum = 0;
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            max_diff = std::max(max_diff, std::abs(src[i * N + j] - src2[i * N + j]));
            err_sum += std::abs(src[i * N + j] - src2[i * N + j]);
        }
    }
    std::cout << "err_sum : " << err_sum << std::endl;
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
} block_q4_K_static;

typedef struct {
    float   d;              // delta
    int8_t  qs[256];       // quants
    int16_t bsums[256/16]; // sum of quants in groups of 16
} block_q8_K_static;


// FP32 to FP16 conversion
static inline uint16_t tmp_nntr_fp32_to_fp16(float value) {
    if (std::isnan(value)) {
        return 0x7E00; // Return canonical NaN
    }

    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    uint16_t sign = (bits & 0x80000000) >> 16;
    int32_t exponent = ((bits & 0x7F800000) >> 23) - 127 + 15;
    uint32_t mantissa = bits & 0x007FFFFF;

    if (exponent <= 0) {
        // Subnormal or zero
        if (exponent < -10) {
            return sign; // Too small becomes zero
        }
        // Convert to subnormal
        mantissa = (mantissa | 0x00800000) >> (1 - exponent);
        if (mantissa & 0x00001000) {
            mantissa += 0x00002000;
        }
        return sign | (mantissa >> 13);
    } else if (exponent == 0x1F) {
        if (mantissa == 0) {
            // Infinity
            return sign | 0x7C00;
        } else {
            // NaN
            return sign | 0x7E00;
        }
    } else if (exponent > 30) {
        // Overflow to infinity
        return sign | 0x7C00;
    } else {
        // Normalized number
        if (mantissa & 0x00001000) {
            mantissa += 0x00002000;
            if (mantissa & 0x00800000) {
                mantissa = 0; // Mantissa overflow
                exponent += 1;
            }
        }
        return sign | (exponent << 10) | (mantissa >> 13);
    }
}
// FP16 to FP32 conversion
static inline float tmp_nntr_fp16_to_fp32(uint16_t half) {
    uint16_t sign = (half & 0x8000) >> 15;
    uint16_t exponent = (half & 0x7C00) >> 10;
    uint16_t mantissa = half & 0x03FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            return sign == 0 ? 0.0f : -0.0f;
        } else {
            // Subnormal number
            float normalized = mantissa / 1024.0f;
            float result = std::ldexp(normalized, -14);
            return sign == 0 ? result : -result;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            // Infinity
            return sign == 0 ? INFINITY : -INFINITY;
        } else {
            // NaN
            return NAN;
        }
    } else {
        // Normalized number
        float normalized = 1.0f + mantissa / 1024.0f;
        float result = std::ldexp(normalized, exponent - 15);
        return sign == 0 ? result : -result;
    }
}

static inline void tmp_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                    uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// static void dequantize_row_q8_K(const void * x, float * y, int64_t k) {
//     // block_q8_K
//     block_q8_K_static* x_casted = (block_q8_K_static * ) x;
//     const int64_t nb = k / 256;
//     for (int i = 0; i < nb; i++) {
//         for (int j = 0; j < 256; ++j) {
//             *y++ = x_casted[i].d * x_casted[i].qs[j];
//         }
//     }
// }

static void dequantize_row_q4_K(const void * x_raw, float * y, int64_t k) {
    block_q4_K_static* x = (block_q4_K_static *) x_raw; // block_q4_K
    const int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = tmp_nntr_fp16_to_fp32(x[i].d);
        const float min = tmp_nntr_fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            tmp_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            tmp_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

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
    // const unsigned int M = 1024;
    // const unsigned int K = 768;
    // const unsigned int N = 512;
    
    ///@note q4_K GEMM is a Row-Major, transB GEMM
    ///@todo Temporally use homogenous matrices. Need to replace with random data after accuracy debugging. Reason why it is set 1.0 and 1.5 is to compare with benchmark-matmult.cpp from llama.cpp
    // std::vector<float> activation = generate_homogeneous_vector<float>(M * K, 1.0f);
    // std::vector<float> weight = generate_homogeneous_vector<float>(N * K, 1.5F);
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
    nntrainer::sgemm(0/*ROW MAJOR*/, false, true, M, N, K, 1.F, lhs_ptr, K, rhs_ptr, N, 0.F, ref_dst_ptr, N);

    // Step0. Allocate a temporary buffer for quantized weight
    int64_t ne0 = N; // row length of the weight matrix
    int64_t q4_k_block_size = 256;
    int64_t q4_k_type_size = sizeof(block_q4_K_static);
    int64_t num_blocks = (K * N) / q4_k_block_size;
    size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
    data_size *= K;
    ///@todo this is an invalid weight data. Needs fix.
    std::vector<char> offline_qWeight = std::vector<char>(data_size); 
    char* offline_qWeight_ptr = (char*) offline_qWeight.data();

    // Step1. Supposed to be an offline Weight quantization from float to q4_K (Zero latency overhead for the model runtime)
    nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight_ptr, K, N, nullptr);
    // nntrainer::quantize_q4_K(rhs_ptr, /*dst quantized vector*/(void*) offline_qWeight, N, K, nullptr);
    ///@todo HOW BIG IS THE WEIGHT BUFFER? (N * K / len(q4_K) * sizeof(q4_K) bytes)?

    /* Step1 is successfully checked!
        /// CHECK : is Step1 valid?
        // print_matrix<float>(weight.data(), N, K);
        dequantize_row_q4_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);
        // print_matrix<float>(rhs_ptr_tmp, N, K);
        auto mean_squared_error = mse<float, float>( weight.data(), rhs_ptr_tmp, N * K);
        auto cos_sim = cosine_similarity( weight.data(), rhs_ptr_tmp, N * K);
        auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);
        std::cout << "mean_squared_error : " << mean_squared_error << std::endl;
        std::cout << "cos_sim : " << cos_sim << std::endl;
        std::cout << "max_differ : " << max_differ << std::endl;
        /// END CHECK : is Step1 valid?
    */

    // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the model weights. It's a one-time operation)
    ///@note do something like : nntrainer::repack_q4_K_to_q4_K_8x8(offline_qWeight_ptr, N, K, q4_K_8x8_weight);

    // Step3. Run GEMM! (Online activation quantization + kernel routine + return float)
    ///@note do something like : nntrainer::gemm_q4_K(M, N, K, lhs_ptr, K, (void*) offline_qWeight_ptr, N, dst_ptr, N);

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