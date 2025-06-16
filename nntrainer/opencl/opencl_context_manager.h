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

#ifndef __OPENCL_CONTEXT_MANAGER_H__
#define __OPENCL_CONTEXT_MANAGER_H__

#include <mutex>

#include "CL/cl.h"

#include "opencl_device.h"

namespace nntrainer::opencl {

/**
 * @class ContextManager contains wrappers for managing OpenCL context
 * @brief OpenCL context wrapper
 *
 */
class ContextManager {
  OpenCLDevice opencl_device_;
  cl_context context_;

  /**
   * @brief Create a default device object
   *
   * @return true if successful or false otherwise
   */
  bool CreateDefaultDevice(cl_device_type type = CL_DEVICE_TYPE_GPU);

  /**
   * @brief Create OpenCL context
   *
   * @return true if successful or false otherwise
   */
  bool CreateCLContext();

  /**
   * @brief Private constructor to prevent object creation
   *
   */
  ContextManager() {};

public:
  /**
   * @brief Get the global instance object
   *
   * @return ContextManager global instance
   */
  static ContextManager &GetInstance();

  /**
   * @brief Get the OpenCL context object
   *
   * @return const cl_context
   */
  const cl_context &GetContext();

  /**
   * @brief Release OpenCL context
   *
   */
  void ReleaseContext();

  /**
   * @brief Get the Device Id object
   *
   * @return const cl_device_id
   */
  const cl_device_id GetDeviceId();

  /**
   * @brief Deleting operator overload
   *
   */
  void operator=(ContextManager const &) = delete;

  /**
   * @brief Deleting copy constructor
   *
   */
  ContextManager(ContextManager const &) = delete;

  /**
   * @brief Destroy the Context Manager object
   *
   */
  ~ContextManager();
};
} // namespace nntrainer::opencl
#endif // __OPENCL_CONTEXT_MANAGER_H__
