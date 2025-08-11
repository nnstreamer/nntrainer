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
  const size_t buffer_size_bytes = 1024 * 8192 * sizeof(float);

  /// @note this size might be changed
  const size_t scale_q4_0_size =
    3072 * (8192 / 32) * 2; /** buffer size of quants */
  const size_t quant_q4_0_size =
    3072 * (8192 / 32) * 16; /** buffer size of scales */

  opencl::Buffer *inBufferA = nullptr;
  opencl::Buffer *inBufferB = nullptr;
  opencl::Buffer *inBufferC = nullptr;
  opencl::Buffer *outBufferA = nullptr;
  opencl::Buffer *outBufferB = nullptr;

  // OpenCL Buffer used for quants & scales in QK_K computation
  opencl::Buffer *scaleBuffer;
  opencl::Buffer *quantBuffer;

  // OpenCL Image used for input & output
  cl_mem input_image = nullptr;  /** created by inBufferC */
  cl_mem output_image = nullptr; /** created by outBufferB */
  cl_mem q_image = nullptr;      /** created by quantBuffer */
  cl_mem d_image = nullptr;      /** created by scaleBuffer */

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
   * @brief Get the Scale Buffer object
   */
  opencl::Buffer *getScaleBuffer() { return scaleBuffer; }

  /**
   * @brief Get the Quant Buffer object
   */
  opencl::Buffer *getQuantBuffer() { return quantBuffer; }

  /**
   * @brief Get the input image mem (backend by inBufferC)
   */
  cl_mem &getInputImage() { return input_image; }

  /**
   * @brief Get the output image mem  (backend by outBufferB)
   */
  cl_mem &getOutputImage() { return output_image; }

  /**
   * @brief Get the input image mem (backend by inBufferC)
   */
  cl_mem &getQuantImage() { return q_image; }

  /**
   * @brief Get the output image mem  (backend by outBufferB)
   */
  cl_mem &getScaleImage() { return d_image; }

  /**
   * @brief Destroy Buffer pointers.
   *
   */
  ~ClBufferManager();
};
} // namespace nntrainer

#endif /* __CL_BUFFER_MANAGER_H__ */
