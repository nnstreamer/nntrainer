// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024  Niket Agarwal <niket.a@samsung.com>
 *
 * @file   concat_cl_fp16.cpp
 * @date   2 July 2024
 * @brief  Implementation of Concat Layer for FP16
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cstring>
#include <vector>

#include <concat_cl.h>
#include <iostream>
#include <layer_context.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor_dim.h>
#include <util_func.h>

std::string concat_cl_axis3_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void concat_cl_axis3_fp16(__global const half* in1, 
                                           __global const half* in2, 
                                           __global half* out,
                                           const int batch_size, 
                                           const int channels, 
                                           const int height, 
                                           const int width1, 
                                           const int width2) {
    int global_id = get_global_id(0);
    
    int total_width = width1 + width2;
    
    int width = total_width;

    // 4D space coordinates
    int w = global_id % total_width;
    int h = (global_id / total_width) % height;
    int c = (global_id / (total_width * height)) % channels;
    int b = global_id / (total_width * height * channels);

    int output_index = ((b * channels + c) * height + h) * total_width + w;
    
    // Determining if the index is in in1 or in2
    if (w < width1) {
        // in1 index calculation
        int input1_index = ((b * channels + c) * height + h) * width1 + w;
        out[output_index] = in1[input1_index];
  
    } else {
        // in2 index calculation
        int input2_index = ((b * channels + c) * height + h) * width2 + (w - width1);
        out[output_index] = in2[input2_index];
    }
})";

std::string concat_cl_axis2_kernel_fp16_ =
  R"(__kernel void concat_cl_axis2_fp16(__global const half* in1,
                             __global const half* in2,
                             __global half* out,
                             const int batch_size,
                             const int channels,
                             const int height1,
                             const int height2,
                             const int width) {
    
    int total_height = height1 + height2;
    int global_id = get_global_id(0);
    
    // Calculate the coordinates in the 4D space
    int w = global_id % width;
    int h = (global_id / width) % total_height;
    int c = (global_id / (width * total_height)) % channels;
    int b = global_id / (width * total_height * channels);

    // Calculate the offset for the current batch, channel, and width in the output tensor
    int output_index = ((b * channels + c) * total_height + h) * width + w;

    if (h < height1) {
        // Index within input1
        int input1_index = ((b * channels + c) * height1 + h) * width + w;
        out[output_index] = in1[input1_index];
    } else {
        // Index within input2
        int input2_index = ((b * channels + c) * height2 + (h - height1)) * width + w;
        out[output_index] = in2[input2_index];
    }

})";

std::string concat_cl_axis1_kernel_fp16_ =
  R"(__kernel void concat_cl_axis1_fp16(__global const half* in1, 
                                           __global const half* in2, 
                                           __global half* out,
                                           const int batch_size, 
                                           const int channels1, 
                                           const int channels2, 
                                           const int height, 
                                           const int width) {
    int global_id = get_global_id(0);
    
    int total_channels = channels1 + channels2;

    // Calculate the coordinates in the 4D space
    int w = global_id % width;
    int h = (global_id / width) % height;
    int c = (global_id / (width * height)) % total_channels;
    int b = global_id / (width * height * total_channels);

    // Calculate the offset for the current batch, height, and width in the output tensor
    int output_index = ((b * total_channels + c) * height + h) * width + w;

    if (c < channels1) {
        // Index within input1
        int input1_index = ((b * channels1 + c) * height + h) * width + w;
        out[output_index] = in1[input1_index];
    } else {
        // Index within input2
        int input2_index = ((b * channels2 + (c - channels1)) * height + h) * width + w;
        out[output_index] = in2[input2_index];
    }
})";

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

opencl::Kernel ConcatLayerCl::kernel_concat_axis3_fp16;
opencl::Kernel ConcatLayerCl::kernel_concat_axis2_fp16;
opencl::Kernel ConcatLayerCl::kernel_concat_axis1_fp16;

void ConcatLayerCl::concat_cl_axis3_fp16(const _FP16 *matAdata,
                                         const _FP16 *vecXdata, _FP16 *vecYdata,
                                         unsigned int input1_batch_size,
                                         unsigned int input1_channels,
                                         unsigned int input1_height,
                                         unsigned int input1_width,
                                         unsigned int input2_width) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_concat_ptr =
      cl_context_ref.registerClKernel(concat_cl_axis3_kernel_fp16_,
                                      "concat_cl_axis3_fp16");
    if (!kernel_concat_ptr) {
      break;
    }

    int dim = int(input1_batch_size * input1_channels * input1_height *
                  (input1_width + input2_width));

    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input1_channels *
                            input1_height * input1_width,
                          true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input1_channels *
                            input1_height * input2_width,
                          true, nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input1_channels *
                            input1_height * (input1_width + input2_width),
                          true, nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input2_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

void ConcatLayerCl::concat_cl_axis2_fp16(const _FP16 *matAdata,
                                         const _FP16 *vecXdata, _FP16 *vecYdata,
                                         unsigned int input1_batch_size,
                                         unsigned int input1_channels,
                                         unsigned int input1_width,
                                         unsigned int input1_height,
                                         unsigned int input2_height) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_concat_ptr =
      cl_context_ref.registerClKernel(concat_cl_axis2_kernel_fp16_,
                                      "concat_cl_axis2_fp16");
    if (!kernel_concat_ptr) {
      break;
    }

    int dim = int(input1_batch_size * input1_channels * input1_width *
                  (input1_height + input2_height));

    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input1_channels *
                            input1_height * input1_width,
                          true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input1_channels *
                            input2_height * input1_width,
                          true, nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input1_channels *
                            (input1_height + input2_height) * input1_width,
                          true, nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input2_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

void ConcatLayerCl::concat_cl_axis1_fp16(const _FP16 *matAdata,
                                         const _FP16 *vecXdata, _FP16 *vecYdata,
                                         unsigned int input1_batch_size,
                                         unsigned int input1_height,
                                         unsigned int input1_width,
                                         unsigned int input1_channels,
                                         unsigned int input2_channels) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_concat_ptr =
      cl_context_ref.registerClKernel(concat_cl_axis1_kernel_fp16_,
                                      "concat_cl_axis1_fp16");
    if (!kernel_concat_ptr) {
      break;
    }

    int dim = int(input1_batch_size * input1_width * input1_height *
                  (input1_channels + input2_channels));

    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input1_channels *
                            input1_height * input1_width,
                          true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input2_channels *
                            input1_height * input1_width,
                          true, nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_,
                          sizeof(_FP16) * input1_batch_size * input1_width *
                            input1_height * (input1_channels + input2_channels),
                          true, nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input2_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

} /* namespace nntrainer */
