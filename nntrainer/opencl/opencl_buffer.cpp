// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_buffer.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for buffer usage
 *
 */

#include "opencl_buffer.h"

#include "opencl_loader.h"

#include <nntrainer_log.h>

namespace nntrainer::opencl {

/**
 * @brief Construct a new Buffer object
 *
 * @param context_manager reference
 * @param size_in_bytes size of buffer
 * @param read_only flag
 * @param data data for the buffer
 */
Buffer::Buffer(ContextManager &context_manager, int size_in_bytes,
               bool read_only, void *data) {
  cl_context context = context_manager.GetContext();
  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  if (data) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }
  cl_int error_code;

  // clCreateBuffer returns NULL with error code if fails
  mem_buf_ = clCreateBuffer(context, flags, size_in_bytes, data, &error_code);
  size_ = size_in_bytes;
  if (!mem_buf_) {
    size_ = 0;
    ml_loge("Failed to allocate device memory (clCreateBuffer). OpenCL error "
            "code: %d",
            error_code);
  }
}

/**
 * @brief Move constructor for buffer by deleting the previous buffer
 *
 * @param buffer
 */
Buffer::Buffer(Buffer &&buffer) :
  mem_buf_(buffer.mem_buf_), size_(buffer.size_) {
  buffer.mem_buf_ = nullptr;
  buffer.size_ = 0;
}

/**
 * @brief Swapping buffer object using operator overload
 *
 * @param buffer
 * @return Buffer&
 */
Buffer &Buffer::operator=(Buffer &&buffer) {
  if (this != &buffer) {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(mem_buf_, buffer.mem_buf_);
  }
  return *this;
}

Buffer::~Buffer() { Release(); }

/**
 * @brief Get the Buffer object
 *
 * @return cl_mem& refrence to cl_mem
 */
cl_mem &Buffer::GetBuffer() { return mem_buf_; }

/**
 * @brief writing data to buffer
 *
 * @param command_queue_inst reference of command queue instance
 * @param data
 * @return true if successful write or false otherwise
 */
bool Buffer::WriteData(CommandQueueManager &command_queue_inst,
                       const void *data) {
  return command_queue_inst.EnqueueWriteBuffer(mem_buf_, size_, data);
}

/**
 * @brief reading data from the buffer
 *
 * @param command_queue_inst reference of command queue instance
 * @param data
 * @return true if successful read or false otherwise
 */
bool Buffer::ReadData(CommandQueueManager &command_queue_inst, void *data) {
  return command_queue_inst.EnqueueReadBuffer(mem_buf_, size_, data);
}

/**
 * @brief Release OpenCL buffer
 *
 */
void Buffer::Release() {
  if (mem_buf_) {
    clReleaseMemObject(mem_buf_);
    mem_buf_ = nullptr;
  }
  size_ = 0;
}

} // namespace nntrainer::opencl
