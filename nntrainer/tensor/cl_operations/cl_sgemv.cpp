// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_sgemv.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Experimental SGEMV implementation using OpenCL
 *
 * @note This file is experimental and is kept for testing purpose
 *
 */

#include "cl_sgemv.h"
#include <iostream>
#include <opencl_buffer.h>

#include <nntrainer_log.h>

namespace nntrainer::internal {

template <typename T>
T *GpuCLSgemv::CLSgemv(const T *matAdata, const T *vecXdata, T *vecYdata,
                       T alpha, T beta, unsigned int dim1, unsigned int dim2) {

  ml_logi("GpuCLSgemv::CLSgemv");

  bool result = false;

  do {
    result = Init(sgemv_kernel_, "sgemv");
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(T) * dim1;
    size_t dim2_size = sizeof(T) * dim2;
    opencl::Buffer inputA(context_inst_, dim1_size * dim2_size, true, nullptr);

    opencl::Buffer inputX(context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inOutY(context_inst_, dim2_size, true, nullptr);

    result = inputA.WriteData(command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(3, &alpha, sizeof(T));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(4, &beta, sizeof(T));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(5, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_.SetKernelArguments(6, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = command_queue_inst_.DispatchCommand(kernel_, work_groups_count,
                                                 work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);

  return vecYdata;
}

template float *GpuCLSgemv::CLSgemv<float>(const float *matAdata,
                                           const float *vecXdata,
                                           float *vecYdata, float alpha,
                                           float beta, unsigned int dim1,
                                           unsigned int dim2);

} // namespace nntrainer::internal
