// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_op_interface.hpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Manage OpenCL operation flow
 *
 */

#ifndef GPU_CL_OP_INTERFACE_HPP_
#define GPU_CL_OP_INTERFACE_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "opencl_command_queue_manager.hpp"
#include "opencl_context_manager.hpp"
#include "opencl_kernel.hpp"
#include "opencl_program.hpp"

namespace nntrainer::internal {
class GpuCLOpInterface {

protected:
  bool initialized_;
  Kernel kernel_;
  ContextManager &context_inst_ = ContextManager::GetInstance();
  CommandQueueManager &command_queue_inst_ = CommandQueueManager::GetInstance();

  bool Init(std::string kernel_string, std::string kernel_name);

  ~GpuCLOpInterface();
};
} // namespace nntrainer::internal

#endif // GPU_CL_OP_INTERFACE_HPP_