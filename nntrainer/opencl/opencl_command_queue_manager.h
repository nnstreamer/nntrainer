// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_command_queue_manager.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for command queue management
 *
 */

#ifndef __OPENCL_COMMAND_QUEUE_MANAGER_H__
#define __OPENCL_COMMAND_QUEUE_MANAGER_H__

#include "opencl_kernel.h"
#include "third_party/cl.h"
#include <memory>

namespace nntrainer::opencl {

/**
 * @class CommandQueueManager contains wrappers for managing OpenCL command
 * queue
 * @brief OpenCL command queue wrapper
 *
 */
class CommandQueueManager {

  /**
   * @brief cl_command_queue instance
   *
   */
  cl_command_queue command_queue_{nullptr};

  /**
   * @brief Private constructor to prevent object creation
   *
   */
  CommandQueueManager(){};

public:
  /**
   * @brief Get the global instance
   *
   * @return CommandQueueManager global instance
   */
  static CommandQueueManager &GetInstance();

  /**
   * @brief Create a Command Queue object
   *
   * @return true if creation is successful or false otherwise
   */
  bool CreateCommandQueue();

  /**
   * @brief Release th OpenCL command queue instance
   *
   */
  void ReleaseCommandQueue();

  /**
   * @brief Reading buffer object. Used from Buffer class
   *
   * @param buffer cl_mem buffer object
   * @param size_in_bytes size of data
   * @param data getting the data stored in buffer
   * @param async flag for asynchronous operation
   * @return true if reading is successful or false otherwise
   */
  bool EnqueueReadBuffer(cl_mem buffer, size_t size_in_bytes, void *data,
                         bool async = false);

  /**
   * @brief Writing buffer object. Used from Buffer class
   *
   * @param buffer cl_mem buffer object
   * @param size_in_bytes size of data
   * @param data to be enqueued into the buffer
   * @param async flag for asynchronous operation
   * @return true if writing is successful or false otherwise
   */
  bool EnqueueWriteBuffer(cl_mem buffer, size_t size_in_bytes, const void *data,
                          bool async = false);

  /**
   * @brief Mapping a region of a buffer object into the host address space
   *
   * @param buffer cl_mem buffer object
   * @param offset_in_bytes offset of the region in the buffer object that is
   * being mapped
   * @param size_in_bytes size of the buffer object that is being mapped
   * @param read_only flag for read only mapping
   * @param async flag for asynchronous operation
   * @param event Object that identifies this command and can be used to query
   * or wait for this command to complete
   * @return void* pointer to the mapped region
   */
  void *EnqueueMapBuffer(cl_mem buffer, size_t offset_in_bytes,
                         size_t size_in_bytes, bool read_only,
                         bool async = false, cl_event *event = nullptr);

  /**
   * @brief Un-mapping a buffer object from the host address space
   *
   * @param buffer cl_mem buffer object
   * @param mapped_ptr pointer to the mapped region
   * @param event Object that identifies this command and can be used to query
   * or wait for this command to complete
   * @return true if unmap is successful
   */
  bool EnqueueUnmapMemObject(cl_mem buffer, void *mapped_ptr,
                             cl_event *event = nullptr);

  /**
   * @brief Function to initiate execution of the command queue.
   *
   * @param kernel OpenCL kernel
   * @param work_groups_count Total number of work items that will execute the
   * kernel function
   * @param work_group_size Number of work items that make up a work group
   * @param event Object that identifies this command and can be used to query
   * or wait for this command to complete
   * @return true if command queue execution is successful or false otherwise
   */
  bool DispatchCommand(Kernel kernel, const int (&work_groups_count)[3],
                       const int (&work_group_size)[3],
                       cl_event *event = nullptr);

  /**
   * @brief Overloaded function to initiate execution of the command queue.
   *
   * @param kernel_ptr reference of OpenCL kernel shared_ptr
   * @param work_groups_count Total number of work items that will execute the
   * kernel function
   * @param work_group_size Number of work items that make up a work group
   * @param event Object that identifies this command and can be used to query
   * or wait for this command to complete
   * @return true if command queue execution is successful or false otherwise
   */
  bool DispatchCommand(const std::shared_ptr<Kernel> &kernel_ptr,
                       const int (&work_groups_count)[3],
                       const int (&work_group_size)[3],
                       cl_event *event = nullptr);

  /**
   * @brief Get the OpenCL Command Queue object
   *
   * @return const cl_command_queue
   */
  const cl_command_queue GetCommandQueue();

  /**
   * @brief Deleting operator overload
   *
   */
  void operator=(CommandQueueManager const &) = delete;

  /**
   * @brief Deleting copy constructor
   *
   */
  CommandQueueManager(CommandQueueManager const &) = delete;

  /**
   * @brief Destroy the Command Queue Manager object
   *
   */
  ~CommandQueueManager();
};
} // namespace nntrainer::opencl

#endif // __OPENCL_COMMAND_QUEUE_MANAGER_H__
