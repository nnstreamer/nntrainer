// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_kernel.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for kernel management
 *
 */

#include "opencl_kernel.hpp"

#include "opencl_loader.hpp"

#include <nntrainer_log.h>

namespace nntrainer::internal {

bool Kernel::CreateKernelFromProgram(Program program,
                                     const std::string &function_name) {
  int error_code;
  cl_program prgm = program.GetProgram();
  kernel_ = clCreateKernel(prgm, function_name.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS) {
    kernel_ = nullptr;
    ml_loge("Failed to create %s. OpenCL error code: %d", function_name.c_str(),
            error_code);
    return false;
  }
  clRetainProgram(prgm);

  return true;
}

bool Kernel::SetKernelArguments(cl_uint arg_index, const void *arg_value,
                                size_t size) {
  int error_code;
  error_code = clSetKernelArg(kernel_, arg_index, size, arg_value);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to set argument. OpenCL error code: %d", error_code);
    return false;
  }

  return true;
}

const cl_kernel Kernel::GetKernel() { return kernel_; }

} // namespace nntrainer::internal