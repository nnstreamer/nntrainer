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

#include "CL/cl.h"
#include "opencl_command_queue_manager.h"
#include "opencl_context_manager.h"
#include "utils/noncopyable.h"

namespace nntrainer::opencl {

/**
 * @class Buffer contains wrappers for managing OpenCL buffer
 * @brief OpenCL wrapper for buffer
 */
class Buffer : public Noncopyable {
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
  Buffer(ContextManager &context_manager, size_t size_in_bytes, bool read_only,
         void *data = nullptr);

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
   * @brief writing data to a buffer region
   *
   * @param command_queue_inst reference of command queue instance
   * @param size_in_bytes size of region
   * @param data pointer of region
   * @param host_origin_offset offset in the host memory region
   * @param buffer_origin_offset offset in the buffer memory region
   * @return true if successful write or false otherwise
   */
  bool WriteDataRegion(CommandQueueManager &command_queue_inst,
                       size_t size_in_bytes, const void *data,
                       size_t host_origin_offset = 0,
                       size_t buffer_origin_offset = 0);

  /**
   * @brief reading data from the buffer
   *
   * @param command_queue_inst reference of command queue instance
   * @param data
   * @return true if successful read or false otherwise
   */
  bool ReadData(CommandQueueManager &command_queue_inst, void *data);

  /**
   * @brief Reading data from a buffer region
   *
   * @param command_queue_inst reference of command queue instance
   * @param size_in_bytes size of region
   * @param data pointer of region
   * @param host_origin_offset offset in the host memory region
   * @param buffer_origin_offset offset in the buffer memory region
   * @return true if successful write or false otherwise
   */
  bool ReadDataRegion(CommandQueueManager &command_queue_inst,
                      size_t size_in_bytes, void *data,
                      size_t host_origin_offset = 0,
                      size_t buffer_origin_offset = 0);

  /**
   * @brief Mapping buffer to host memory
   *
   * @param command_queue_inst reference of command queue instance
   * @param offset_in_bytes offset of the region in the buffer object that is
   * being mapped
   * @param size_in_bytes size of the buffer object that is being mapped
   * @param read_only flag for read only mapping
   * @param async flag for asynchronous operation
   * @return void* pointer to the mapped region
   */
  void *MapBuffer(CommandQueueManager &command_queue_inst,
                  size_t offset_in_bytes, size_t size_in_bytes, bool read_only,
                  bool async = false);

  /**
   * @brief Un-mapping buffer from host memeory
   *
   * @param command_queue_inst reference of command queue instance
   * @param mapped_ptr pointer to the mapped region
   * @return true if unmap is successful
   */
  bool UnMapBuffer(CommandQueueManager &command_queue_inst, void *mapped_ptr);
};
} // namespace nntrainer::opencl
#endif // __OPENCL_BUFFER_H__
