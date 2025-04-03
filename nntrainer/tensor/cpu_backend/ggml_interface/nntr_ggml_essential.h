// SPDX-License-Identifier: Apache-2.0
/**
 * @file	nntr_ggml_essential.h
 * @date	03 April 2025
 * @brief	This is essential ggml codes to use ggml_cpu_impl
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cstdint>
#include <cstring>
#include "ggml-common.h"
#include "ggml_avx_macro.h"
#include <cassert>
#include <cmath>

#ifndef MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#elif defined(__VXE__) || defined(__VXE2__)
#define CACHE_LINE_SIZE 256
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

// enum class
enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_COUNT   = 39,
};

struct ggml_type_traits {
    const char             * type_name;
    int64_t                  blck_size;
    // int64_t                  blck_size_interleave; // interleave elements in blocks
    size_t                   type_size;
    bool                     is_quantized;
    // ggml_to_float_t          to_float;
    // ggml_from_float_t        from_float_ref;
};

// ggm_type_traits array in C++ style
static const ggml_type_traits type_traits[GGML_TYPE_COUNT] = {
    // GGML_TYPE_I8
    { "i8", 1, sizeof(int8_t), false },
    // GGML_TYPE_I16
    { "i16", 1, sizeof(int16_t), false },
    // GGML_TYPE_I32
    { "i32", 1, sizeof(int32_t), false },
    // GGML_TYPE_I64
    { "i64", 1, sizeof(int64_t), false },
    // GGML_TYPE_F64
    { "f64", 1, sizeof(double), false },
    // GGML_TYPE_F32
    { "f32", 1, sizeof(float), false },
    // GGML_TYPE_F16
    { "f16", 1, 0, false }, // Assuming type_size is 0 for F16; adjust as needed
    // GGML_TYPE_Q4_0
    { "q4_0", QK4_0, sizeof(block_q4_0), true },
    // GGML_TYPE_Q4_1
    { "q4_1", QK4_1, sizeof(block_q4_1), true },
    // GGML_TYPE_Q5_0
    { "q5_0", QK5_0, sizeof(block_q5_0), true },
    // GGML_TYPE_Q5_1
    { "q5_1", QK5_1, sizeof(block_q5_1), true },
    // GGML_TYPE_Q8_0
    { "q8_0", QK8_0, sizeof(block_q8_0), true },
    // GGML_TYPE_Q8_1
    { "q8_1", QK8_1, sizeof(block_q8_1), true },
    // GGML_TYPE_Q2_K
    { "q2_K", QK_K, sizeof(block_q2_K), true },
    // GGML_TYPE_Q3_K
    { "q3_K", QK_K, sizeof(block_q3_K), true },
    // GGML_TYPE_Q4_K
    { "q4_K", QK_K, sizeof(block_q4_K), true },
    // GGML_TYPE_Q5_K
    { "q5_K", QK_K, sizeof(block_q5_K), true },
    // GGML_TYPE_Q6_K
    { "q6_K", QK_K, sizeof(block_q6_K), true },
    // GGML_TYPE_IQ2_XXS
    { "iq2_xxs", QK_K, sizeof(block_iq2_xxs), true },
    // GGML_TYPE_IQ2_XS
    { "iq2_xs", QK_K, sizeof(block_iq2_xs), true },
    // GGML_TYPE_IQ3_XXS
    { "iq3_xxs", QK_K, sizeof(block_iq3_xxs), true },
    // GGML_TYPE_IQ3_S
    { "iq3_s", QK_K, sizeof(block_iq3_s), true },
    // GGML_TYPE_IQ2_S
    { "iq2_s", QK_K, sizeof(block_iq2_s), true },
    // GGML_TYPE_IQ1_S
    { "iq1_s", QK_K, sizeof(block_iq1_s), true },
    // GGML_TYPE_IQ1_M
    { "iq1_m", QK_K, sizeof(block_iq1_m), true },
    // GGML_TYPE_IQ4_NL
    { "iq4_nl", QK4_NL, 0, true }, // Assuming type_size is 0; adjust as needed
    // GGML_TYPE_IQ4_XS
    { "iq4_xs", QK_K, 0, true }, // Assuming type_size is 0; adjust as needed
    // GGML_TYPE_Q8_K
    { "q8_K", QK_K, sizeof(block_q8_K), true },
    // GGML_TYPE_BF16
    { "bf16", 1, 0, false }, // Assuming type_size is 0; adjust as needed
    // GGML_TYPE_TQ1_0
    { "tq1_0", QK_K, sizeof(block_tq1_0), true },
    // GGML_TYPE_TQ2_0
    { "tq2_0", QK_K, sizeof(block_tq2_0), true },
    // GGML_TYPE_COUNT (if needed)
    { nullptr, 0, 0, false } // Placeholder for GGML_TYPE_COUNT; adjust as needed
};
struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;
    
    ///@todo Enable multithreading
    // struct ggml_threadpool * threadpool;
};

// FP32 to FP16 conversion
uint16_t nntr_fp32_to_fp16(float value) {
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
float nntr_fp16_to_fp32(uint16_t half) {
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

int64_t ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}

