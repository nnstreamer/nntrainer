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
bool GpuCLOpInterface::Init(std::string kernel_string,
                            std::string kernel_name) {
  if (initialized_) {
    ml_logi("Kernel already initialized: %s", kernel_name.c_str());
    return true;
  }

  ml_logi("Kernel initializing: %s", kernel_name.c_str());

  bool result = false;

  do {
    result = command_queue_inst_.CreateCommandQueue();
    if (!result) {
      break;
    }

    Program program;
    result =
      program.CreateCLProgram(context_inst_.GetContext(),
                              context_inst_.GetDeviceId(), kernel_string, "");
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

GpuCLOpInterface::~GpuCLOpInterface() {
  if (initialized_) {
    command_queue_inst_.ReleaseCommandQueue();
    context_inst_.ReleaseContext();
  }
}
} // namespace nntrainer::opencl
