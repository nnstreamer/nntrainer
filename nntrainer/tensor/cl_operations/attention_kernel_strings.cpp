// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernel_strings.cpp
 * @date	2 April 2025
 * @brief	All attention OpenCL kernel strings
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "attention_kernel_strings.h"

namespace nntrainer {

const std::string &getRotaryEmbClKernel() {
  static const std::string rotary_emb_cl_kernel_ = R"(
  __kernel void rotary_emb_cl(__global float *input,
                                        __global float *output,
                                        __global float *freqs_cos,
                                        __global float *freqs_sin,
                                        __global float *cos_,
                                        __global float *sin_,
                                        unsigned int batch,
                                        unsigned int channel,
                                        unsigned int height,
                                        unsigned int width,
                                        unsigned int dim,
                                        unsigned int half_,
                                        unsigned int max_timestep,
                                        unsigned int from,
                                        unsigned int offsetFreqsSin,
                                        unsigned int offsetSin) {
      __global float *cos_ptr = cos_;
      __global float *sin_ptr = sin_;
  
      float value = 0.0f;
      float transformed_value = 0.0f;
  
      unsigned int b = get_global_id(0);
      unsigned int c = get_global_id(1);
      
      if(b < batch && c < channel){
        for (unsigned int h = 0; h < height; h++) {
          if (from + h < max_timestep) {
            unsigned idx = (from + h)*dim;
            for(unsigned int i = idx; i < idx + dim; i++){
              cos_ptr[i - idx] = freqs_cos[i];
              sin_ptr[i - idx + offsetSin] = freqs_sin[i + offsetFreqsSin];
            }
          }
  
          for (unsigned int w = 0; w < width; w = w + dim) {
            for (unsigned int k = 0; k < dim; k++) {
              unsigned int span = w + k;
              value = input[b * channel * height * width + c * height * width + h * width + span];
              if (k < half_) {
                transformed_value = -1.0f * input[b * channel * height * width + c * height * width + h * width + span + half_];
              } else {
                transformed_value = input[b * channel * height * width + c * height * width + h * width + span - half_];
              }
              value = value * cos_ptr[k] + transformed_value * sin_ptr[k + offsetSin];
              output[b * channel * height * width + c * height * width + h * width + span] = value;
            }
          }
        }
      }
  }
  )";
  return rotary_emb_cl_kernel_;
}

#ifdef ENABLE_FP16

// USING :
// https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-opencl/kernels/flash_attn_f16.cl
const std::string &getFlashAttentionClKernelFP16() {
  static const std::string flash_attention_kernel_ = R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// NOTE(m.wlasiuk) : see ggml/src/ggml-opencl/ggml-opencl.cpp:1353
#define DK 256
#define DV 256
#define BLOCK_M 16
#define BLOCK_N 16

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define DATA_TYPE half
#define DATA_TYPE4 half4
#define CONVERT_ACC4(x) convert_float4(x)
#define CONVERT_DATA4(x) convert_half4(x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)
#define Q1_WG_SIZE 64

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}
__kernel void flash_attn_f16(
    const global void * q_void,
    const global void * k_void,
    const global void * v_void,
    global void * o_void,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1,
    const ulong q_nb2,
    const ulong q_nb3,
    const ulong k_nb1,
    const ulong k_nb2,
    const ulong k_nb3,
    const ulong v_nb1,
    const ulong v_nb2,
    const ulong v_nb3,
    const ulong o_nb1,
    const ulong o_nb2,
    const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void;
    const global char* k_base = (const global char*)k_void;
    const global char* v_base = (const global char*)v_void;
    global char* o_base = (global char*)o_void;

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global DATA_TYPE4* q_ptr = (const global DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = CONVERT_ACC4(q_ptr[i]);
        }
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global DATA_TYPE4*)(k_base + k_row_offset))[col];
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global DATA_TYPE4*)(v_base + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) {
            continue;
        }

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j;
            const int k_row1 = k_start + j + 1;

            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_acc0 = mad(q_priv[k], CONVERT_ACC4(l_k[j][k]), dot_acc0);
                dot_acc1 = mad(q_priv[k], CONVERT_ACC4(l_k[j+1][k]), dot_acc1);
            }
            ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
            ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

            if (is_causal) {
                if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
            }

            if (k_row0 >= n_kv) score0 = -INFINITY;
            if (k_row1 >= n_kv) score1 = -INFINITY;


            if (logit_softcap > 0.0f) {
                score0 = logit_softcap * tanh(score0 / logit_softcap);
                score1 = logit_softcap * tanh(score1 / logit_softcap);
            }

            const ACC_TYPE m_new = max(m_i, max(score0, score1));
            const ACC_TYPE p0 = exp(score0 - m_new);
            const ACC_TYPE p1 = exp(score1 - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_ACC4(l_v[j][i]) + p1 * CONVERT_ACC4(l_v[j+1][i]);
            }
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global DATA_TYPE4 *o_row = (global DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (DATA_TYPE4)(0.0f);
            }
        }
    }
}
  )";

  return flash_attention_kernel_;
}

const std::string &getRotaryEmbClKernelFP16() {
  static const std::string rotary_emb_cl_kernel_fp16_ = R"(
  
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    
  __kernel void rotary_emb_cl_fp16(__global half *input,
                                        __global half *output,
                                        __global float *freqs_cos,
                                        __global float *freqs_sin,
                                        __global float *cos_,
                                        __global float *sin_,
                                        unsigned int batch,
                                        unsigned int channel,
                                        unsigned int height,
                                        unsigned int width,
                                        unsigned int dim,
                                        unsigned int half_,
                                        unsigned int max_timestep,
                                        unsigned int from,
                                        unsigned int offsetFreqsSin,
                                        unsigned int offsetSin) {
      __global float *cos_ptr = cos_;
      __global float *sin_ptr = sin_;
  
      float value = 0.0f;
      float transformed_value = 0.0f;
  
      unsigned int b = get_global_id(0);
      unsigned int c = get_global_id(1);
      
      if(b < batch && c < channel){
        for (unsigned int h = 0; h < height; h++) {
          if (from + h < max_timestep) {
            unsigned idx = (from + h)*dim;
            for(int i = idx; i < idx + dim; i++ ){
              cos_ptr[i - idx] = freqs_cos[i];
              sin_ptr[i - idx + offsetSin] = freqs_sin[i + offsetFreqsSin];
            }
          }
  
          for (unsigned int w = 0; w < width; w = w + dim) {
            for (unsigned int k = 0; k < dim; k++) {
              unsigned int span = w + k;
              value = (float)input[b * channel * height * width + c * height * width + h * width + span];
              if (k < half_) {
                transformed_value = -1.0f * (float)input[b * channel * height * width + c * height * width + h * width + span + half_];
              } else {
                transformed_value = (float)input[b * channel * height * width + c * height * width + h * width + span - half_];
              }
              value = value * cos_ptr[k] + transformed_value * sin_ptr[k + offsetSin];
              output[b * channel * height * width + c * height * width + h * width + span] = (half)value;
            }
          }
        }
      }
  }
  )";
  return rotary_emb_cl_kernel_fp16_;
}
#endif

} // namespace nntrainer
