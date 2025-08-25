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

#ifndef __OPENCL_KERNEL_H__
#define __OPENCL_KERNEL_H__

#include <string>
#include <vector>

#include "CL/cl.h"
#include "opencl_program.h"

namespace nntrainer::opencl {

/**
 * @brief Parameters needed to set OpenCL kernel argument
 */
struct KernelArgument {
  uint32_t index = {};
  const void *value = nullptr;
  uint64_t size = {};
};

/**
 * @brief Parameters needed to set OpenCL kernel SVM argument
 */
struct KernelSVMArgument {
  uint32_t index = {};
  const void *value = nullptr;
};

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
  bool SetKernelArgument(cl_uint arg_index, const void *arg_value,
                         size_t size) const;

  /**
   * @brief Set the Kernel Arguments
   *
   * @param arg_index index of the argument
   * @param arg_value value of the argument
   * @return true if successful or false otherwise
   */
  bool SetKernelSVMArgument(cl_uint arg_index, const void *arg_value) const;

  /**
   * @brief Set the Kernel Arguments
   *
   * @param args list of arguments to set
   * @return true if successful or false if setting of any argument failed
   */
  bool SetKernelArguments(const std::vector<KernelArgument> &args) const;

  /**
   * @brief Set the Kernel Arguments
   *
   * @param args list of arguments to set
   * @return true if successful or false if setting of any argument failed
   */
  bool SetKernelSVMArguments(const std::vector<KernelSVMArgument> &args) const;

  /**
   * @brief Get the Kernel object
   *
   * @return const cl_kernel
   */
  const cl_kernel GetKernel();
};
} // namespace nntrainer::opencl
#endif // __OPENCL_KERNEL_H__
