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
                                        unsigned int from) {
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
              sin_ptr[i - idx] = freqs_sin[i];
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
              value = value * cos_ptr[k] + transformed_value * sin_ptr[k];
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
                                        unsigned int from) {
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
              sin_ptr[i - idx] = freqs_sin[i];
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
              value = value * cos_ptr[k] + transformed_value * sin_ptr[k];
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
