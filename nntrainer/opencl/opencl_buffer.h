// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_buffer.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for buffer usage
 *
 */

#ifndef GPU_CL_OPENCL_BUFFER_HPP_
#define GPU_CL_OPENCL_BUFFER_HPP_

#include "opencl_command_queue_manager.h"
#include "opencl_context_manager.h"
#include "third_party/cl.h"

namespace nntrainer::opencl {
class Buffer {
  cl_mem mem_buf_{nullptr};
  size_t size_{0};
  void Release();

public:
  Buffer(){};
  Buffer(ContextManager &context_manager, int size_in_bytes, bool read_only,
         void *data);

  // Move only
  Buffer(Buffer &&buffer);
  Buffer &operator=(Buffer &&buffer);
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  ~Buffer();
  cl_mem &GetBuffer();

  bool WriteData(CommandQueueManager &command_queue_inst, const void *data);
  bool ReadData(CommandQueueManager &command_queue_inst, void *data);
};
} // namespace nntrainer::opencl
#endif // GPU_CL_OPENCL_BUFFER_HPP_
