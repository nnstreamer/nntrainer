// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   reshape_cl_fp16.cpp
 * @date   18 June 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Reshape GPU Layer Implementation for FP16.
 */

#include <iostream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <reshape_cl.h>

std::string copy_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void copy_cl_fp16(__global const half* input, 
                               __global half* output,
                               const int batchsize, 
                               const int channels, 
                               const int height, 
                               const int width) {

    int i= get_global_id(0);
    output[i] = input[i];
    
})";

namespace nntrainer {

opencl::Kernel ReshapeLayerCl::kernel_copy_fp16;

void ReshapeLayerCl::copy_cl_fp16(const _FP16 *input, _FP16 *res,
                                  unsigned int input_batch_size,
                                  unsigned int input_channels,
                                  unsigned int input_height,
                                  unsigned int input_width) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_copy_ptr =
      cl_context_ref.registerClKernel(copy_cl_kernel_fp16_, "copy_cl_fp16");
    if (!kernel_copy_ptr) {
      break;
    }

    size_t dim_size = sizeof(_FP16) * input_batch_size * input_height *
                      input_width * input_channels;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim_size, true,
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, input);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_copy_ptr->SetKernelArguments(2, &input_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_copy_ptr->SetKernelArguments(3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim_size, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_copy_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

} /* namespace nntrainer */
