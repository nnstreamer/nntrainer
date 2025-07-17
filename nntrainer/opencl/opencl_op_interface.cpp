// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_op_interface.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Manage OpenCL operation flow
 *
 * @note This class is experimental and might be deprecated in future
 *
 */

#include "opencl_op_interface.h"
#include <nntrainer_log.h>

namespace nntrainer::opencl {

/**
 * @brief Initialize OpenCL kernel
 *
 * @param kernel_string
 * @param kernel_name
 * @return true if successful or false otherwise
 */
bool GpuCLOpInterface::Init(std::string kernel_string,
                            std::string kernel_name) {
  if (initialized_) {
    ml_logi("Kernel already initialized: %s", kernel_name.c_str());
    return true;
  }

  ml_logi("Kernel initializing: %s", kernel_name.c_str());

  bool result = false;

  do {
    // creating command queue
    result = command_queue_inst_.CreateCommandQueue();
    if (!result) {
      break;
    }

    Program program;

    // creating program
    result = program.CreateCLProgram(context_inst_.GetContext(),
                                     context_inst_.GetDeviceId(), kernel_string,
                                     "-cl-fast-relaxed-math -cl-mad-enable");
    if (!result) {
      break;
    }

    result = kernel_.CreateKernelFromProgram(program, kernel_name);
    if (!result) {
      break;
    }
    initialized_ = true;
  } while (false);

  return result;
}

/**
 * @brief Destroy the GpuCLOpInterface object
 *
 */
GpuCLOpInterface::~GpuCLOpInterface() {
  if (initialized_) {
    // releaseing command queue and context since they are created in
    // GpuCLOpInterface::Init
    command_queue_inst_.ReleaseCommandQueue();
    context_inst_.ReleaseContext();
  }
}
} // namespace nntrainer::opencl
