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
  NNTR_API ClBufferManager() :
    inBufferA(nullptr),
    inBufferB(nullptr),
    inBufferC(nullptr),
    outBufferA(nullptr),
    outBufferB(nullptr){};

  /**
   * @brief OpenCl context global instance
   *
   */
  opencl::ContextManager &context_inst_ = opencl::ContextManager::GetInstance();

  /**
   * @brief Buffer size in bytes preset (256 mebibytes)
   */
  const size_t buffer_size_bytes = 8192 * 8192 * sizeof(float);

  opencl::Buffer *inBufferA;
  opencl::Buffer *inBufferB;
  opencl::Buffer *inBufferC;
  opencl::Buffer *outBufferA;
  opencl::Buffer *outBufferB;

public:
  /**
   * @brief Get Global ClBufferManager.
   *
   * @return ClBufferManager&
   */
  NNTR_API static ClBufferManager &getInstance();

  /**
   * @brief Initialize Buffer objects.
   */
  NNTR_API void initBuffers();

  /**
   * @brief Get read only inBufferA.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  NNTR_API opencl::Buffer *getInBufferA() { return inBufferA; }

  /**
   * @brief Get read only inBufferB.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  NNTR_API opencl::Buffer *getInBufferB() { return inBufferB; }

  /**
   * @brief Get read only inBufferC.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  NNTR_API opencl::Buffer *getInBufferC() { return inBufferC; }

  /**
   * @brief Get read-write outBufferA.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  NNTR_API opencl::Buffer *getOutBufferA() { return outBufferA; }

  /**
   * @brief Get read-write outBufferB.
   * @return opencl::Buffer* or nullptr if initBuffers() is not called
   */
  NNTR_API opencl::Buffer *getOutBufferB() { return outBufferB; }

  /**
   * @brief Destroy Buffer pointers.
   *
   */
  NNTR_API ~ClBufferManager();
};
} // namespace nntrainer

#endif /* __CL_BUFFER_MANAGER_H__ */
