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

#if defined(_WIN32)
#define NNTR_API __declspec(dllexport)
#else
#define NNTR_API
#endif

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
  NNTR_API bool CreateDefaultGPUDevice();

  /**
   * @brief Create OpenCL context
   *
   * @return true if successful or false otherwise
   */
  NNTR_API bool CreateCLContext();

  /**
   * @brief Private constructor to prevent object creation
   *
   */
  NNTR_API ContextManager(){};

public:
  /**
   * @brief Get the global instance object
   *
   * @return ContextManager global instance
   */
  NNTR_API static ContextManager &GetInstance();

  /**
   * @brief Get the OpenCL context object
   *
   * @return const cl_context
   */
  NNTR_API const cl_context &GetContext();

  /**
   * @brief Release OpenCL context
   *
   */
  NNTR_API void ReleaseContext();

  /**
   * @brief Get the Device Id object
   *
   * @return const cl_device_id
   */
  NNTR_API const cl_device_id GetDeviceId();

  /**
   * @brief allocate SVM memory
   *
   * @param size size of the memory to be allocated
   * @return void* pointer to the allocated SVM memory
   */
  NNTR_API void *createSVMRegion(size_t size);

  /**
   * @brief deallocate SVM memory
   *
   * @param svm_ptr pointer to the SVM memory to be deallocated
   */
  NNTR_API void releaseSVMRegion(void *svm_ptr);

  /**
   * @brief Deleting operator overload
   *
   */
  NNTR_API void operator=(ContextManager const &) = delete;

  /**
   * @brief Deleting copy constructor
   *
   */
  NNTR_API ContextManager(ContextManager const &) = delete;

  /**
   * @brief Destroy the Context Manager object
   *
   */
  NNTR_API ~ContextManager();

};
} // namespace nntrainer::opencl
#endif // __OPENCL_CONTEXT_MANAGER_H__
