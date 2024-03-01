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

/**
 * @class ContextManager contains wrappers for managing OpenCL context
 * @brief OpenCL context wrapper
 *
 */
class ContextManager {
  cl_platform_id platform_id_{nullptr};
  cl_device_id device_id_{nullptr};
  cl_context context_{nullptr};

  /**
   * @brief Create a Default GPU Device object
   *
   * @return true if successful or false otherwise
   */
  bool CreateDefaultGPUDevice();

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
  ContextManager(){};

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
#endif // GPU_CL_OPENCL_CONTEXT_MANAGER_HPP_
