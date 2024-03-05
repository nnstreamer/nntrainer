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

#ifndef __OPENCL_BUFFER_H__
#define __OPENCL_BUFFER_H__

#include "opencl_command_queue_manager.h"
#include "opencl_context_manager.h"
#include "third_party/cl.h"

namespace nntrainer::opencl {

/**
 * @class Buffer contains wrappers for managing OpenCL buffer
 * @brief OpenCL wrapper for buffer
 */
class Buffer {
  /**
   * @brief cl_mem object to store the buffer
   *
   */
  cl_mem mem_buf_{nullptr};

  size_t size_{0};

  /**
   * @brief Release OpenCL buffer
   *
   */
  void Release();

public:
  /**
   * @brief Default constructor
   *
   */
  Buffer(){};

  /**
   * @brief Construct a new Buffer object
   *
   * @param context_manager reference
   * @param size_in_bytes size of buffer
   * @param read_only flag
   * @param data data for the buffer
   */
  Buffer(ContextManager &context_manager, int size_in_bytes, bool read_only,
         void *data);

  /**
   * @brief Move constructor for buffer by deleting the previous buffer
   *
   * @param buffer
   */
  Buffer(Buffer &&buffer);

  /**
   * @brief Swapping buffer object using operator overload
   *
   * @param buffer
   * @return Buffer&
   */
  Buffer &operator=(Buffer &&buffer);

  /**
   * @brief Deleting copy constructor
   *
   */
  Buffer(const Buffer &) = delete;

  /**
   * @brief Deleting operator overload
   *
   */
  Buffer &operator=(const Buffer &) = delete;

  /**
   * @brief Destroy the Buffer object
   *
   */
  ~Buffer();

  /**
   * @brief Get the Buffer object
   *
   * @return cl_mem& refrence to cl_mem
   */
  cl_mem &GetBuffer();

  /**
   * @brief writing data to buffer
   *
   * @param command_queue_inst reference of command queue instance
   * @param data
   * @return true if successful write or false otherwise
   */
  bool WriteData(CommandQueueManager &command_queue_inst, const void *data);

  /**
   * @brief reading data from the buffer
   *
   * @param command_queue_inst reference of command queue instance
   * @param data
   * @return true if successful read or false otherwise
   */
  bool ReadData(CommandQueueManager &command_queue_inst, void *data);
};
} // namespace nntrainer::opencl
#endif // __OPENCL_BUFFER_H__
