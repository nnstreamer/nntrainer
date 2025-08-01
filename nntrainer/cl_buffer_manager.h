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

#include "utils/singleton.h"

namespace nntrainer {

/**
 * @class ClBufferManager contains Buffer object management
 * @brief Support for Buffer management
 */

class ClBufferManager : public Singleton<ClBufferManager> {

private:
  /**
   * @brief OpenCl context global instance
   *
   */
  opencl::ContextManager &context_inst_ = opencl::ContextManager::Global();

  /**
   * @brief Buffer size in bytes preset (256 mebibytes)
   */
  const size_t buffer_size_bytes = 8192 * 8192 * sizeof(float);

  opencl::Buffer *inBufferA = nullptr;
  opencl::Buffer *inBufferB = nullptr;
  opencl::Buffer *inBufferC = nullptr;
  opencl::Buffer *outBufferA = nullptr;
  opencl::Buffer *outBufferB = nullptr;

public:
  /**
   * @brief Initialize Buffer objects.
   */
  void initBuffers();

  /**
   * @brief Get read only inBufferA.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  opencl::Buffer *getInBufferA() { return inBufferA; }

  /**
   * @brief Get read only inBufferB.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  opencl::Buffer *getInBufferB() { return inBufferB; }

  /**
   * @brief Get read only inBufferC.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  opencl::Buffer *getInBufferC() { return inBufferC; }

  /**
   * @brief Get read-write outBufferA.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  opencl::Buffer *getOutBufferA() { return outBufferA; }

  /**
   * @brief Get read-write outBufferB.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  opencl::Buffer *getOutBufferB() { return outBufferB; }

  /**
   * @brief Destroy Buffer pointers.
   *
   */
  ~ClBufferManager();
};
} // namespace nntrainer

#endif /* __CL_BUFFER_MANAGER_H__ */
