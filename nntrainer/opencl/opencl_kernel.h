// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_kernel.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for kernel management
 *
 */

#ifndef GPU_CL_OPENCL_KERNEL_HPP_
#define GPU_CL_OPENCL_KERNEL_HPP_

#include <string>

#include "opencl_program.h"
#include "third_party/cl.h"

namespace nntrainer::opencl {

/**
 * @class Kernel contains wrappers for managing OpenCL kernels
 * @brief OpenCL kernel wrapper
 *
 */
class Kernel {
  cl_kernel kernel_{nullptr};

public:
  /**
   * @brief Create a Kernel From Program object
   *
   * @param program
   * @param function_name the kernel string name
   * @return true if successful or false otherwise
   */
  bool CreateKernelFromProgram(Program program,
                               const std::string &function_name);

  /**
   * @brief Set the Kernel Arguments
   *
   * @param arg_index index of the argument
   * @param arg_value value of the argument
   * @param size size of the argument
   * @return true if successful or false otherwise
   */
  bool SetKernelArguments(cl_uint arg_index, const void *arg_value,
                          size_t size);

  /**
   * @brief Get the Kernel object
   *
   * @return const cl_kernel
   */
  const cl_kernel GetKernel();
};
} // namespace nntrainer::opencl
#endif // GPU_CL_OPENCL_KERNEL_HPP_
