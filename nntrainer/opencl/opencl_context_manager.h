// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_context_manager.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for context management
 *
 */

#ifndef GPU_CL_OPENCL_CONTEXT_MANAGER_HPP_
#define GPU_CL_OPENCL_CONTEXT_MANAGER_HPP_

#include <mutex>

#include "third_party/cl.h"

namespace nntrainer::opencl {
class ContextManager {
  cl_platform_id platform_id_{nullptr};
  cl_device_id device_id_{nullptr};
  cl_context context_{nullptr};

  bool CreateDefaultGPUDevice();
  bool CreateCLContext();

  ContextManager(){};

public:
  static ContextManager &GetInstance();

  const cl_context &GetContext();
  void ReleaseContext();

  const cl_device_id GetDeviceId();

  void operator=(ContextManager const &) = delete;
  ContextManager(ContextManager const &) = delete;
  ~ContextManager();
};
} // namespace nntrainer::opencl
#endif // GPU_CL_OPENCL_CONTEXT_MANAGER_HPP_
