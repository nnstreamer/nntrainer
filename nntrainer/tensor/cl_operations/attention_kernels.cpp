// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernels.cpp
 * @date	28 August 2024
 * @brief	Common attention OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <attention_kernels.h>

namespace nntrainer {
std::string rotary_emb_cl_kernel = R"(
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
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
    unsigned int gid = get_global_id(0);
    unsigned int gws = get_global_size(0);

    __global float *cos_ptr = cos_;
    __global float *sin_ptr = sin_;

    float value = 0.0f;
    float transformed_value = 0.0f;

    for (unsigned int b = 0; b < batch; b++) {
      for (unsigned int c = 0; c < channel; c++) {
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
}
)";

/**
 * @brief defining global kernel objects
 */
opencl::Kernel kernel_rotary_emb;

void rotary_emb_cl(float *in, float *out,
                   std::vector<std::vector<float>> freqs_cos,
                   std::vector<std::vector<float>> freqs_sin,
                   std::vector<float> cos_, std::vector<float> sin_,
                   unsigned int batch, unsigned int channel,
                   unsigned int height, unsigned int width, unsigned int dim,
                   unsigned int from, unsigned int max_timestep,
                   unsigned int in_size, unsigned int out_size,
                   RunLayerContext &context) {
  bool result = false;

  do {
    result = context.clCreateKernel(
      rotary_emb_cl_kernel, context.LayerKernel::ROTARY_EMB, kernel_rotary_emb);
    if (!result) {
      printf("Failed to create kernel for rotary_emb_cl\n");
      break;
    }
    unsigned int cos_dim = cos_.size();
    unsigned int sin_dim = sin_.size();
    unsigned int freqs_cos_dim = freqs_cos.size();
    unsigned int freqs_sin_dim = freqs_sin.size();

    size_t dim1_size = sizeof(float) * in_size;
    size_t dim2_size = sizeof(float) * out_size;
    size_t dim3_size = sizeof(float) * cos_dim;
    size_t dim4_size = sizeof(float) * sin_dim;
    size_t dim5_size =
      sizeof(float) * freqs_cos_dim * dim; // max_timestep * dim
    size_t dim6_size = sizeof(float) * freqs_sin_dim * dim;

    opencl::Buffer inputA(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inOutRes(context.context_inst_, dim2_size, true, nullptr);

    opencl::Buffer cosBuf(context.context_inst_, dim3_size, true, nullptr);

    opencl::Buffer sinBuf(context.context_inst_, dim4_size, true, nullptr);

    opencl::Buffer freqs_cosBuf(context.context_inst_, dim5_size, true,
                                nullptr);

    opencl::Buffer freqs_sinBuf(context.context_inst_, dim6_size, true,
                                nullptr);

    std::vector<float> freqs_cos_flat;
    std::vector<float> freqs_sin_flat;
    for (const auto &row : freqs_cos) {
      freqs_cos_flat.insert(freqs_cos_flat.end(), row.begin(), row.end());
    }
    for (const auto &row : freqs_sin) {
      freqs_sin_flat.insert(freqs_sin_flat.end(), row.begin(), row.end());
    }

    result = inputA.WriteData(context.command_queue_inst_, in);
    if (!result) {
      printf("Failed to write input data\n");
      break;
    }

    result = inOutRes.WriteData(context.command_queue_inst_, out);
    if (!result) {
      printf("Failed to write output data\n");
      break;
    }

    result = freqs_cosBuf.WriteData(context.command_queue_inst_,
                                    freqs_cos_flat.data());
    if (!result) {
      printf("Failed to write freqs cos data\n");
      break;
    }

    result = freqs_sinBuf.WriteData(context.command_queue_inst_,
                                    freqs_sin_flat.data());
    if (!result) {
      printf("Failed to write freqs sin data\n");
      break;
    }

    result = cosBuf.WriteData(context.command_queue_inst_, cos_.data());
    if (!result) {
      printf("Failed to write cos data\n");
      break;
    }

    result = sinBuf.WriteData(context.command_queue_inst_, sin_.data());
    if (!result) {
      printf("Failed to write sin data\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set inputA argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set inOutRes argument\n");
      break;
    }

    result =
      kernel_rotary_emb.SetKernelArguments(2, &freqs_cosBuf, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set freqs_cosBuf argument\n");
      break;
    }

    result =
      kernel_rotary_emb.SetKernelArguments(3, &freqs_sinBuf, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set freqs_sinBuf argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(4, &cosBuf, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set cosBuf argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(5, &sinBuf, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set sinBuf argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(6, &batch, sizeof(int));
    if (!result) {
      printf("Failed to set batch argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(7, &channel, sizeof(int));
    if (!result) {
      printf("Failed to set channel argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(8, &height, sizeof(int));
    if (!result) {
      printf("Failed to set height argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(9, &width, sizeof(int));
    if (!result) {
      printf("Failed to set width argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(10, &dim, sizeof(int));
    if (!result) {
      printf("Failed to set dim argument\n");
      break;
    }
    unsigned int half_ = dim / 2;
    result = kernel_rotary_emb.SetKernelArguments(11, &half_, sizeof(int));
    if (!result) {
      printf("Failed to set half argument\n");
      break;
    }

    result =
      kernel_rotary_emb.SetKernelArguments(12, &max_timestep, sizeof(int));
    if (!result) {
      printf("Failed to set timestamp argument\n");
      break;
    }

    result = kernel_rotary_emb.SetKernelArguments(13, &from, sizeof(int));
    if (!result) {
      printf("Failed to set from argument\n");
      break;
    }

    const int work_groups_count[3] = {1, 1, 1};
    const int work_group_size[3] = {32, 1, 1}; // test-value
    result = context.command_queue_inst_.DispatchCommand(
      kernel_rotary_emb, work_groups_count, work_group_size);
    if (!result) {
      printf("Failed to dispatch command\n");
      break;
    }

    result = inOutRes.ReadData(context.command_queue_inst_, out);
    if (!result) {
      printf("Failed to read data\n");
      break;
    }

  } while (false);
}
} // namespace nntrainer
