// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   swiglu_cl_fp16.cpp
 * @date   6th June 2024
 * @brief  Implementation of SwiGLU activation function for FP16
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "swiglu_cl.h"
#include <iostream>

std::string swiglu_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void swiglu_cl_fp16(__global const half *in1, __global const half *in2, __global half *out) {
    int i = get_global_id(0);
    half swish = in1[i] * exp(in1[i]) / (1 + exp(in1[i]));
    out[i] = swish * in2[i];
})";

namespace nntrainer {

void SwiGLULayerCl::swiglu_cl_fp16(const _FP16 *matAdata, const _FP16 *vecXdata,
                                   _FP16 *vecYdata, unsigned int dim1,
                                   unsigned int dim2) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_swiglu_ptr =
      cl_context_ref.registerClKernel(swiglu_cl_kernel_fp16_, "swiglu_cl_fp16");
    if (!kernel_swiglu_ptr) {
      break;
    }

    int dim = int(dim1 * dim2);
    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          sizeof(_FP16) * dim1 * dim2, true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_,
                          sizeof(_FP16) * dim1 * dim2, true, nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_,
                          sizeof(_FP16) * dim1 * dim2, true, nullptr);

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

    result = kernel_swiglu_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_swiglu_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_swiglu_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_swiglu_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

} // namespace nntrainer
