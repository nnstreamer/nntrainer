// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernels_fp16.cpp
 * @date	28 August 2024
 * @brief	Common attention OpenCL fp16 kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <attention_kernel_strings.h>
#include <attention_kernels.h>

namespace nntrainer {

void rotary_emb_cl(_FP16 *in, _FP16 *out,
                   std::vector<std::vector<float>> freqs_cos,
                   std::vector<std::vector<float>> freqs_sin,
                   std::vector<float> cos_, std::vector<float> sin_,
                   unsigned int batch, unsigned int channel,
                   unsigned int height, unsigned int width, unsigned int dim,
                   unsigned int from, unsigned int max_timestep,
                   unsigned int in_size, unsigned int out_size) {

  bool result = false;
  ClContext *attention_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClBufferManager &clbuffInstance = ClBufferManager::getInstance();

  do {
    ClContext::SharedPtrClKernel kernel_rotaryEmb_fp16_ptr =
      attention_cc->registerClKernel(getRotaryEmbClKernelFP16(),
                                     "rotary_emb_cl_fp16");
    if (!kernel_rotaryEmb_fp16_ptr) {
      break;
    }

    unsigned int cos_dim = cos_.size();
    unsigned int sin_dim = sin_.size();
    unsigned int freqs_cos_dim = freqs_cos.size();
    unsigned int freqs_sin_dim = freqs_sin.size();

    size_t dim1_size = sizeof(cl_half) * in_size;
    size_t dim2_size = sizeof(cl_half) * out_size;
    size_t dim3_size = sizeof(float) * cos_dim;
    size_t dim4_size = sizeof(float) * sin_dim;
    size_t dim5_size = sizeof(float) * freqs_cos_dim * dim;
    size_t dim6_size = sizeof(float) * freqs_sin_dim * dim;

    std::vector<float> freqs_cos_flat;
    std::vector<float> freqs_sin_flat;
    for (const auto &row : freqs_cos) {
      freqs_cos_flat.insert(freqs_cos_flat.end(), row.begin(), row.end());
    }
    for (const auto &row : freqs_sin) {
      freqs_sin_flat.insert(freqs_sin_flat.end(), row.begin(), row.end());
    }

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      attention_cc->command_queue_inst_, dim1_size, in);
    if (!result) {
      printf("Failed to write input data\n");
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      attention_cc->command_queue_inst_, dim2_size, out);
    if (!result) {
      printf("Failed to write output data\n");
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      attention_cc->command_queue_inst_, dim5_size, freqs_cos_flat.data());
    if (!result) {
      printf("Failed to write freqs cos data\n");
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      attention_cc->command_queue_inst_, dim6_size, freqs_sin_flat.data(), 0,
      dim5_size);
    if (!result) {
      printf("Failed to write freqs sin data\n");
      break;
    }

    result = clbuffInstance.getInBufferC()->WriteDataRegion(
      attention_cc->command_queue_inst_, dim3_size, cos_.data());
    if (!result) {
      printf("Failed to write cos data\n");
      break;
    }

    result = clbuffInstance.getInBufferC()->WriteDataRegion(
      attention_cc->command_queue_inst_, dim4_size, sin_.data(), 0, dim3_size);
    if (!result) {
      printf("Failed to write sin data\n");
      break;
    }

    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      printf("Failed to set inputA argument\n");
      break;
    }

    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(
      1, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      printf("Failed to set inOutRes argument\n");
      break;
    }

    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(
      2, clbuffInstance.getInBufferB(), sizeof(cl_mem));
    if (!result) {
      printf("Failed to set freqs_cosBuf argument\n");
      break;
    }

    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(
      3, clbuffInstance.getInBufferB(), sizeof(cl_mem));
    if (!result) {
      printf("Failed to set freqs_sinBuf argument\n");
      break;
    }

    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(
      4, clbuffInstance.getInBufferC(), sizeof(cl_mem));
    if (!result) {
      printf("Failed to set cosBuf argument\n");
      break;
    }

    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(
      5, clbuffInstance.getInBufferC(), sizeof(cl_mem));
    if (!result) {
      printf("Failed to set sinBuf argument\n");
      break;
    }

    result =
      kernel_rotaryEmb_fp16_ptr->SetKernelArguments(6, &batch, sizeof(int));
    if (!result) {
      printf("Failed to set batch argument\n");
      break;
    }

    result =
      kernel_rotaryEmb_fp16_ptr->SetKernelArguments(7, &channel, sizeof(int));
    if (!result) {
      printf("Failed to set channel argument\n");
      break;
    }

    result =
      kernel_rotaryEmb_fp16_ptr->SetKernelArguments(8, &height, sizeof(int));
    if (!result) {
      printf("Failed to set height argument\n");
      break;
    }

    result =
      kernel_rotaryEmb_fp16_ptr->SetKernelArguments(9, &width, sizeof(int));
    if (!result) {
      printf("Failed to set width argument\n");
      break;
    }

    result =
      kernel_rotaryEmb_fp16_ptr->SetKernelArguments(10, &dim, sizeof(int));
    if (!result) {
      printf("Failed to set dim argument\n");
      break;
    }
    unsigned int half_ = dim / 2;
    result =
      kernel_rotaryEmb_fp16_ptr->SetKernelArguments(11, &half_, sizeof(int));
    if (!result) {
      printf("Failed to set half argument\n");
      break;
    }

    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(12, &max_timestep,
                                                           sizeof(int));
    if (!result) {
      printf("Failed to set timestamp argument\n");
      break;
    }

    result =
      kernel_rotaryEmb_fp16_ptr->SetKernelArguments(13, &from, sizeof(int));
    if (!result) {
      printf("Failed to set from argument\n");
      break;
    }

    unsigned int offsetFreqsSin = freqs_cos_dim * dim;
    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(14, &offsetFreqsSin,
                                                           sizeof(int));
    if (!result) {
      printf("Failed to set offsetFreqsSin argument\n");
      break;
    }

    unsigned int offsetSin = cos_dim;
    result = kernel_rotaryEmb_fp16_ptr->SetKernelArguments(15, &offsetSin,
                                                           sizeof(int));
    if (!result) {
      printf("Failed to set offsetSin argument\n");
      break;
    }

    const int work_groups_count[3] = {(int)batch, (int)channel, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value
    result = attention_cc->command_queue_inst_.DispatchCommand(
      kernel_rotaryEmb_fp16_ptr, work_groups_count, work_group_size);
    if (!result) {
      printf("Failed to dispatch command\n");
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      attention_cc->command_queue_inst_, dim2_size, out);
    if (!result) {
      printf("Failed to read data\n");
      break;
    }

  } while (false);
}
} // namespace nntrainer
