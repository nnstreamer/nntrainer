// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_command_queue_manager.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for command queue management
 *
 */

#ifndef GPU_CL_OPENCL_COMMAND_QUEUE_MANAGER_HPP_
#define GPU_CL_OPENCL_COMMAND_QUEUE_MANAGER_HPP_

#include "opencl_kernel.h"
#include "third_party/cl.h"

namespace nntrainer::opencl {
class CommandQueueManager {
  cl_command_queue command_queue_{nullptr};

  CommandQueueManager(){};

public:
  static CommandQueueManager &GetInstance();
  bool CreateCommandQueue();
  void ReleaseCommandQueue();

  bool EnqueueReadBuffer(cl_mem buffer, size_t size_in_bytes, void *data,
                         bool async = false);
  bool EnqueueWriteBuffer(cl_mem buffer, size_t size_in_bytes, const void *data,
                          bool async = false);

  bool DispatchCommand(Kernel kernel, const int (&work_groups_count)[3],
                       const int (&work_group_size)[3],
                       cl_event *event = nullptr);

  const cl_command_queue GetCommandQueue();

  void operator=(CommandQueueManager const &) = delete;
  CommandQueueManager(CommandQueueManager const &) = delete;
  ~CommandQueueManager();
};
} // namespace nntrainer::opencl

#endif // GPU_CL_OPENCL_COMMAND_QUEUE_MANAGER_HPP_
