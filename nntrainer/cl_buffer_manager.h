// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_buffer_manager.h
 * @date    01 Dec 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them
 */

#ifndef __CL_BUFFER_MANAGER_H__
#define __CL_BUFFER_MANAGER_H__

#include <string>

#include <opencl_buffer.h>
#include <opencl_context_manager.h>

#include <nntrainer_log.h>

namespace nntrainer {

/**
 * @class ClBufferManager contains Buffer object management
 * @brief Support for Buffer management
 */

class ClBufferManager {

private:
  /**
   * @brief Private constructor to prevent object creation
   *
   */
  ClBufferManager(){};

  /**
   * @brief OpenCl context global instance
   *
   */
  opencl::ContextManager &context_inst_ = opencl::ContextManager::GetInstance();

  /**
   * @brief Buffer size in bytes preset (256 mebibytes)
   */
  size_t buffer_size_bytes = 8192 * 8192 * sizeof(float);

public:
  /**
   * @brief Get Global ClBufferManager.
   *
   * @return ClBufferManager&
   */
  static ClBufferManager &getInstance();

  opencl::Buffer *readBufferA;
  opencl::Buffer *readBufferB;
  opencl::Buffer *readBufferC;
  opencl::Buffer *writeBufferA;
  opencl::Buffer *writeBufferB;

  /**
   * @brief Initialize Buffer objects.
   */
  void initBuffers();

  /**
   * @brief Destroy Buffer pointers.
   *
   */
  ~ClBufferManager();
};
} // namespace nntrainer

#endif /* __CL_BUFFER_MANAGER_H__ */
