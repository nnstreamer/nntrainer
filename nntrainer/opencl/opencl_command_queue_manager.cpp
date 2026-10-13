// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_command_queue_manager.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for command queue management
 *
 */

#include "opencl_command_queue_manager.h"

#include "opencl_context_manager.h"
#include "opencl_loader.h"

#include <atomic>
#include <cstdlib>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

/** @brief Return the CommandQueueManager singleton. */
template <>
NNTRAINER_SINGLETON_API opencl::CommandQueueManager &
Singleton<opencl::CommandQueueManager>::Global() {
  static opencl::CommandQueueManager instance;
  instance.initializeOnce();
  return instance;
}

} // namespace nntrainer

namespace nntrainer::opencl {

namespace {

/**
 * @brief Whether to drain the queue after every dispatch that touches shared
 *        virtual memory.
 *
 * Off by default; NNTR_XE3_SYNC=1 turns it on. It used to default on for every
 * device without fine-grain SVM (Xe3), where it cost a clFinish per
 * shared-memory dispatch -- 218 of 461 per decoded token on a 1.5B model. Two
 * things were measured before turning it off.
 *
 * Kernel to kernel, the in-order queue already orders coarse-grain SVM: with
 * the drain off, a 1.5B dense model and a 2B per-layer-embedding model gave the
 * same tokens as with it, run after run.
 *
 * The one ordering the drain was really providing is a shared-memory unmap
 * enqueued behind a kernel that is still running. With the drain off, Gemma-4
 * produced different text run to run until that unmap waited for the queue
 * (enqueueSVMUnmap below). With that in place the drain is redundant, and it
 * stays here only as a diagnostic.
 *
 * Resolved once per process.
 */
bool needsCoarseSVMDrain() {
  static const bool drain = []() {
    const char *e = std::getenv("NNTR_XE3_SYNC");
    const bool on = e != nullptr && std::atoi(e) != 0;
    if (on)
      ml_logi("NNTR_XE3_SYNC=%s: draining the queue after every "
              "shared-memory dispatch",
              e);
    return on;
  }();
  return drain;
}

/**
 * @brief Whether the device's shared virtual memory is coarse-grain only.
 *
 * Neither fine-grain capability (buffer or system) is reported. The unmap
 * ordering below is a correctness dependency, not a throughput lever, so no
 * environment variable turns it off. Unknown device info answers true and is
 * not latched -- the cost of a wrong "true" is one clFinish on an idle queue.
 */
bool coarseGrainSVMOnly() {
  static std::atomic<int> state{-1};
  const int s = state.load(std::memory_order_relaxed);
  if (s >= 0)
    return s != 0;
  const auto *device_info = ContextManager::Global().getDeviceInfo();
  if (!device_info)
    return true;
  const cl_device_svm_capabilities svm =
    device_info->getDeviceSVMCapabilities();
  const bool coarse = (svm & (CL_DEVICE_SVM_FINE_GRAIN_BUFFER |
                              CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) == 0;
  state.store(coarse ? 1 : 0, std::memory_order_relaxed);
  return coarse;
}

/**
 * @brief True while a kernel enqueued on the queue may still be running: set by
 * every NDRange enqueue that was not followed by a drain here, cleared by a
 * drain here or by a blocking map (which waits for everything before it).
 * Another clFinish elsewhere leaves it set; the cost of that is one clFinish
 * on an idle queue before the next shared-memory unmap, never a missed one.
 */
std::atomic<bool> s_kernels_in_flight{false};

} // namespace

/**
 * @brief Create a Command Queue object
 *
 * @return true if creation is successful or false otherwise
 */
bool CommandQueueManager::CreateCommandQueue() {
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Retained command queue");
    // increments the command_queue reference count
    clRetainCommandQueue(command_queue_);
    return true;
  }

  int error_code;
  ContextManager &context_instance = ContextManager::Global();

  // OpenCL context is created
  cl_context context = context_instance.GetContext();

  // If context is invalid, return false
  if (context == nullptr) {
    return false;
  }

  // getting GPU device ID
  cl_device_id device_id = context_instance.GetDeviceId();

  // In-order queue. No kernel dispatch in this tree enqueues a wait list (the
  // only event waits are on asynchronous weight uploads, before their host
  // staging is freed), so out-of-order execution bought no overlap between
  // kernels; what it did do is let the device reorder two kernels that hand
  // off through a shared buffer, which is exactly how consecutive layers on
  // the OpenCL memory allocator communicate. Submission order is the only
  // ordering guarantee those handoffs have.
  command_queue_ = clCreateCommandQueue(context, device_id, 0, &error_code);
  if (!command_queue_) {
    ml_loge("Failed to create a command queue. OpenCL error code: %d : ",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  ml_logi("opencl_command_queue_manager: Created command queue");
  // increments the command_queue reference count
  clRetainCommandQueue(command_queue_);
  ml_logi("opencl_command_queue_manager: Retained command queue");
  return true;
}

/**
 * @brief Release th OpenCL command queue instance
 *
 */
void CommandQueueManager::ReleaseCommandQueue() {
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Released command queue");
    clReleaseCommandQueue(command_queue_);
  }
}

/**
 * @brief Destroy the Command Queue Manager object
 *
 */
CommandQueueManager::~CommandQueueManager() {
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Destroyed command queue");
    // decrements the command_queue reference count
    clReleaseCommandQueue(command_queue_);
    command_queue_ = nullptr;

    // releasing OpenCL context since it has been created by
    // CommandQueueManager::CreateCommandQueue
    ContextManager::Global().ReleaseContext();
  }
}

/**
 * @brief Get the OpenCL Command Queue object
 *
 * @return const cl_command_queue
 */
const cl_command_queue CommandQueueManager::GetCommandQueue() {
  return command_queue_;
}

/**
 * @brief Reading buffer object. Used from Buffer class
 *
 * @param buffer cl_mem buffer object
 * @param size_in_bytes size of data
 * @param data getting the data stored in buffer
 * @param async flag for asynchronous operation
 * @return true if reading is successful or false otherwise
 */
bool CommandQueueManager::EnqueueReadBuffer(cl_mem buffer, size_t size_in_bytes,
                                            void *data, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // returns NULL with error code if fails
  auto error_code =
    clEnqueueReadBuffer(command_queue_, buffer, blocking, 0, size_in_bytes,
                        data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to read data from GPU (clEnqueueReadBuffer). OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

bool CommandQueueManager::EnqueueReadBufferRegion(
  cl_mem buffer, size_t size_in_bytes, void *data, size_t host_origin_offset,
  size_t buffer_origin_offset, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  // (x, y, z) offset in the memory region associated with buffer
  const size_t buffer_origin[] = {buffer_origin_offset, 0, 0};
  // (x, y, z) offset in the memory region associated with host
  const size_t host_origin[] = {host_origin_offset, 0, 0};
  // region defines the (width in bytes, height in rows, depth in slices)
  const size_t region[] = {size_in_bytes, 1, 1};
  // length of each row in bytes
  size_t row_pitch = region[0];
  // length of each 2D slice in bytes
  size_t slice_pitch = region[0] * region[1];

  // Buffer and host data are interpreted as 1D in this case
  // hence row and slice pitch are same for both
  cl_int error_code = clEnqueueReadBufferRect(
    command_queue_, buffer, blocking, buffer_origin, host_origin, region,
    row_pitch, slice_pitch, row_pitch, slice_pitch, data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to write data region to GPU (clEnqueueReadBufferRect). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Writing buffer object. Used from Buffer class
 *
 * @param buffer cl_mem buffer object
 * @param size_in_bytes size of data
 * @param data to be enqueued into the buffer
 * @param async flag for asynchronous operation
 * @return true if writing is successful or false otherwise
 */
bool CommandQueueManager::EnqueueWriteBuffer(cl_mem buffer,
                                             size_t size_in_bytes,
                                             const void *data, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // returns NULL with error code if fails
  auto error_code =
    clEnqueueWriteBuffer(command_queue_, buffer, blocking, 0, size_in_bytes,
                         data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to upload data to GPU (clEnqueueWriteBuffer). OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

bool CommandQueueManager::EnqueueWriteBufferRegion(
  cl_mem buffer, size_t size_in_bytes, const void *data,
  size_t host_origin_offset, size_t buffer_origin_offset, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  // (x, y, z) offset in the memory region associated with buffer
  const size_t buffer_origin[] = {buffer_origin_offset, 0, 0};
  // (x, y, z) offset in the memory region associated with host
  const size_t host_origin[] = {host_origin_offset, 0, 0};
  // region defines the (width in bytes, height in rows, depth in slices)
  const size_t region[] = {size_in_bytes, 1, 1};
  // length of each row in bytes
  size_t row_pitch = region[0];
  // length of each 2D slice in bytes
  size_t slice_pitch = region[0] * region[1];

  // Buffer and host data are interpreted as 1D in this case
  // hence row and slice pitch are same for both
  cl_int error_code = clEnqueueWriteBufferRect(
    command_queue_, buffer, blocking, buffer_origin, host_origin, region,
    row_pitch, slice_pitch, row_pitch, slice_pitch, data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to write data region to GPU (clEnqueueWriteBufferRect). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

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
void *CommandQueueManager::EnqueueMapBuffer(cl_mem buffer,
                                            size_t offset_in_bytes,
                                            size_t size_in_bytes,
                                            bool read_only, bool async,
                                            cl_event *event) {
  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // managing read/write flags
  const cl_map_flags map_flag = read_only ? CL_MAP_READ : CL_MAP_WRITE;

  cl_int error_code;

  void *host_mem_buf = clEnqueueMapBuffer(
    command_queue_, buffer, blocking, map_flag, offset_in_bytes, size_in_bytes,
    0, nullptr, event, &error_code);

  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to map buffer to host memory(clEnqueueMapBuffer). OpenCL error "
      "code: %d : %s",
      error_code, OpenCLErrorCodeToString(error_code));
    return nullptr;
  }
  return host_mem_buf;
}

/**
 * @brief Mapping a region of a buffer object into the host address space
 *
 * @param buffer cl_mem buffer object
 * @param mapped_ptr pointer to the mapped region
 * @param event Object that identifies this command and can be used to query
 * or wait for this command to complete
 * @return true if unmap is successful
 */
bool CommandQueueManager::EnqueueUnmapMemObject(cl_mem buffer, void *mapped_ptr,
                                                cl_event *event) {
  cl_int error_code = clEnqueueUnmapMemObject(command_queue_, buffer,
                                              mapped_ptr, 0, nullptr, event);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to unmap buffer from host memory(clEnqueueUnmapMemObject). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  return true;
}

bool CommandQueueManager::enqueueSVMMap(void *svm_ptr, size_t size,
                                        bool read_only, cl_event *event,
                                        bool async) {
  // managing read/write flags
  const cl_map_flags map_flag = read_only ? CL_MAP_READ : CL_MAP_WRITE;

  // async = true makes the map non-blocking. That is sound only on an in-order
  // queue, where the map is ordered ahead of the next operation's unmap, and
  // only when nothing on the host touches the region before that next GPU op
  // runs -- which is exactly the GPU-to-GPU handoff the quantized GEMM below
  // performs. It removes a host stall that otherwise drains the queue to idle
  // between two device operations. The default keeps the blocking behaviour
  // every existing caller has.
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  // The event out-parameter was accepted and then dropped on the floor here,
  // so a caller could never wait on the map it just enqueued. Pass it through.
  cl_int error_code = clEnqueueSVMMap(command_queue_, blocking, map_flag,
                                      svm_ptr, size, 0, nullptr, event);

  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to map SVM memory (clEnqueueSVMMap). OpenCL error code: %d : %s",
      error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  // A blocking map returns only after every earlier command has completed.
  if (blocking == CL_TRUE)
    s_kernels_in_flight.store(false, std::memory_order_relaxed);
  return true;
}

bool CommandQueueManager::enqueueSVMUnmap(void *svm_ptr, cl_event *event) {
  // On a coarse-grain device, a shared-memory unmap enqueued behind kernels
  // that are still running is not ordered after them in practice, although
  // the queue is in-order: measured on Xe3, the residual add that hands a
  // host-written operand to the device (unmap, copy kernel, map) read
  // different bytes run to run whenever the kernel before it (the per-layer
  // projection norm) had not finished, and identical bytes when the queue
  // was drained first or the unmap was left out. The per-dispatch drain
  // (needsCoarseSVMDrain) hid this by keeping the queue idle at every unmap;
  // this is the one ordering that drain was providing, kept on its own so the
  // drain can be turned off without it.
  if (s_kernels_in_flight.load(std::memory_order_relaxed) &&
      coarseGrainSVMOnly()) {
    clFinish(command_queue_);
    Kernel::clearUndrainedSvmPlanes();
    s_kernels_in_flight.store(false, std::memory_order_relaxed);
  }
  cl_int error_code =
    clEnqueueSVMUnmap(command_queue_, svm_ptr, 0, nullptr, event);

  if (error_code != CL_SUCCESS) {
    // CL_INVALID_VALUE on a real pointer is the runtime saying the region is
    // not mapped (Adreno reports it; Intel accepts the call). Callers unmap
    // every shared operand before a dispatch whether or not the host holds
    // it, so this is routine there, and logging it as an error cost an
    // FP32-activation decode a log write per operand. Still reported to the
    // caller.
    if (error_code == CL_INVALID_VALUE && svm_ptr != nullptr)
      ml_logd("clEnqueueSVMUnmap: %p is not mapped", svm_ptr);
    else
      ml_loge(
        "Failed to unmap SVM memory (clEnqueueSVMUnmap). OpenCL error code: "
        "%d : %s",
        error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  return true;
}

bool CommandQueueManager::kernelsMayBeInFlight() const {
  return s_kernels_in_flight.load(std::memory_order_relaxed);
}

void CommandQueueManager::drainAndResetInFlight() {
  clFinish(command_queue_);
  // Every enqueued command has completed, so nothing is in flight any more --
  // and the undrained-plane set is dispatch-epoch state the finished queue
  // has retired. Reset both: see the header for what a stale flag does to the
  // next model in the process.
  Kernel::clearUndrainedSvmPlanes();
  s_kernels_in_flight.store(false, std::memory_order_relaxed);
}

bool CommandQueueManager::coarseGrainSVM() const {
  return coarseGrainSVMOnly();
}

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
bool CommandQueueManager::DispatchCommand(
  Kernel kernel, const int (&work_groups_count)[3],
  const int (&work_group_size)[3], cl_event *event,
  std::vector<cl_event> events_to_wait) {

  // work_dim of 3 has been hardcoded, might be modified later based on
  // requirements

  // setting the local_work_size referred to as the size of the
  // work-group
  const size_t local[3] = {static_cast<size_t>(work_group_size[0]),
                           static_cast<size_t>(work_group_size[1]),
                           static_cast<size_t>(work_group_size[2])};

  // setting the global_work_size that describe the number of global work-items
  const size_t global[3] = {static_cast<size_t>(work_groups_count[0]),
                            static_cast<size_t>(work_groups_count[1]),
                            static_cast<size_t>(work_groups_count[2])};

  cl_kernel kernel_ = kernel.GetKernel();

  // The risk condition, asked per dispatch. If any shared-memory pointer bound
  // for this dispatch lands inside a plane an earlier undrained dispatch wrote,
  // that write has to be made visible BEFORE this one is enqueued, not after:
  // the consumer is the reader. On a hit the whole queue drains, which retires
  // every outstanding plane, so the set is cleared inside the call. Asked
  // unconditionally so the per-dispatch bound-pointer list is consumed even
  // when the set is empty.
  if (Kernel::takeDispatchSvmHazard())
    clFinish(command_queue_);

  // returns NULL with error code if fails
  const int error_code =
    clEnqueueNDRangeKernel(command_queue_, kernel_, 3, nullptr, global, local,
                           events_to_wait.size(), events_to_wait.data(), event);
  // Always consume the flag, so it never leaks onto the next dispatch.
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();
  // One write-log entry per NDRange, before the error check: a dispatch that
  // failed to enqueue still has to consume its pending declaration, and an
  // entry that says nothing is the safe answer for it.
  Kernel::commitDispatch();
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  if (touched_svm && needsCoarseSVMDrain()) {
    clFinish(command_queue_);
    Kernel::clearUndrainedSvmPlanes();
    s_kernels_in_flight.store(false, std::memory_order_relaxed);
  } else {
    s_kernels_in_flight.store(true, std::memory_order_relaxed);
  }

  return true;
}

bool CommandQueueManager::DispatchCommand(
  const std::shared_ptr<Kernel> &kernel_ptr, const int (&work_groups_count)[3],
  const int (&work_group_size)[3], cl_event *event,
  std::vector<cl_event> events_to_wait) {

  // work_dim of 3 has been hardcoded, might be modified later based on
  // requirements

  // setting the local_work_size referred to as the size of the
  // work-group
  const size_t local[3] = {static_cast<size_t>(work_group_size[0]),
                           static_cast<size_t>(work_group_size[1]),
                           static_cast<size_t>(work_group_size[2])};

  // setting the global_work_size that describe the number of global work-items
  const size_t global[3] = {static_cast<size_t>(work_groups_count[0]),
                            static_cast<size_t>(work_groups_count[1]),
                            static_cast<size_t>(work_groups_count[2])};

  cl_kernel kernel_ = kernel_ptr->GetKernel();

  // The risk condition, asked per dispatch. If any shared-memory pointer bound
  // for this dispatch lands inside a plane an earlier undrained dispatch wrote,
  // that write has to be made visible BEFORE this one is enqueued, not after:
  // the consumer is the reader. On a hit the whole queue drains, which retires
  // every outstanding plane, so the set is cleared inside the call. Asked
  // unconditionally so the per-dispatch bound-pointer list is consumed even
  // when the set is empty.
  if (Kernel::takeDispatchSvmHazard())
    clFinish(command_queue_);

  // returns NULL with error code if fails
  const int error_code =
    clEnqueueNDRangeKernel(command_queue_, kernel_, 3, nullptr, global, local,
                           events_to_wait.size(), events_to_wait.data(), event);
  // Always consume the flag, so it never leaks onto the next dispatch.
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();
  // One write-log entry per NDRange, before the error check: a dispatch that
  // failed to enqueue still has to consume its pending declaration, and an
  // entry that says nothing is the safe answer for it.
  Kernel::commitDispatch();
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  if (touched_svm && needsCoarseSVMDrain()) {
    clFinish(command_queue_);
    Kernel::clearUndrainedSvmPlanes();
    s_kernels_in_flight.store(false, std::memory_order_relaxed);
  } else {
    s_kernels_in_flight.store(true, std::memory_order_relaxed);
  }

  return true;
}

void CommandQueueManager::enqueueKernel(const cl_kernel kernel,
                                        const cl_uint work_dim,
                                        const size_t *global_work_size,
                                        const size_t *local_work_size,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event *event_wait_list,
                                        cl_event *event) {

  // The risk condition, asked per dispatch. If any shared-memory pointer bound
  // for this dispatch lands inside a plane an earlier undrained dispatch wrote,
  // that write has to be made visible BEFORE this one is enqueued, not after:
  // the consumer is the reader. On a hit the whole queue drains, which retires
  // every outstanding plane, so the set is cleared inside the call. Asked
  // unconditionally so the per-dispatch bound-pointer list is consumed even
  // when the set is empty.
  if (Kernel::takeDispatchSvmHazard())
    clFinish(command_queue_);

  const auto error_code = clEnqueueNDRangeKernel(
    command_queue_, kernel, work_dim, nullptr, global_work_size,
    local_work_size, num_events_in_wait_list, event_wait_list, event);

  // Always consume the flag, so it never leaks onto the next dispatch.
  const bool touched_svm = Kernel::takeDispatchTouchedSVM();
  Kernel::commitDispatch();

  NNTR_THROW_IF(error_code != CL_SUCCESS, std::runtime_error)
    << "clEnqueueNDRangeKernel failed. OpenCL error code: " << error_code
    << ", error: " << OpenCLErrorCodeToString(error_code);

  // The attention and rotary-embedding kernels come through here rather than
  // DispatchCommand, and they are the shared-memory producers and consumers,
  // so the flush that keeps their handoff coherent has to live here too.
  if (touched_svm && needsCoarseSVMDrain()) {
    clFinish(command_queue_);
    Kernel::clearUndrainedSvmPlanes();
    s_kernels_in_flight.store(false, std::memory_order_relaxed);
  } else {
    s_kernels_in_flight.store(true, std::memory_order_relaxed);
  }
}

} // namespace nntrainer::opencl
