// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_loader.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Load required OpenCL functions
 *
 */

#ifndef GPU_CL_OPENCL_LOADER_HPP_
#define GPU_CL_OPENCL_LOADER_HPP_

#include "third_party/cl.h"

#define CL_API_ENTRY
#define CL_API_CALL
#define CL_CALLBACK

namespace nntrainer::opencl {

bool LoadOpenCL();

typedef cl_int(CL_API_CALL *PFN_clGetPlatformIDs)(
  cl_uint /* num_entries */, cl_platform_id * /* platforms */,
  cl_uint * /* num_platforms */);

typedef cl_int(CL_API_CALL *PFN_clGetDeviceIDs)(
  cl_platform_id /* platform */, cl_device_type /* device_type */,
  cl_uint /* num_entries */, cl_device_id * /* devices */,
  cl_uint * /* num_devices */);

typedef cl_context(CL_API_CALL *PFN_clCreateContext)(
  const cl_context_properties * /* properties */, cl_uint /* num_devices */,
  const cl_device_id * /* devices */,
  void(CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t,
                                       void *),
  void * /* user_data */, cl_int * /* errcode_ret */);

typedef cl_command_queue(CL_API_CALL *PFN_clCreateCommandQueue)(
  cl_context /* context */, cl_device_id /* device */,
  cl_command_queue_properties /* properties */, cl_int * /* errcode_ret */);

typedef cl_mem(CL_API_CALL *PFN_clCreateBuffer)(cl_context /* context */,
                                                cl_mem_flags /* flags */,
                                                size_t /* size */,
                                                void * /* host_ptr */,
                                                cl_int * /* errcode_ret */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueWriteBuffer)(
  cl_command_queue /* command_queue */, cl_mem /* buffer */,
  cl_bool /* blocking_write */, size_t /* offset */, size_t /* size */,
  const void * /* ptr */, cl_uint /* num_events_in_wait_list */,
  const cl_event * /* event_wait_list */, cl_event * /* event */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueReadBuffer)(
  cl_command_queue /* command_queue */, cl_mem /* buffer */,
  cl_bool /* blocking_read */, size_t /* offset */, size_t /* size */,
  void * /* ptr */, cl_uint /* num_events_in_wait_list */,
  const cl_event * /* event_wait_list */, cl_event * /* event */);

typedef cl_program(CL_API_CALL *PFN_clCreateProgramWithSource)(
  cl_context /* context */, cl_uint /* count */, const char ** /* strings */,
  const size_t * /* lengths */, cl_int * /* errcode_ret */);

typedef cl_int(CL_API_CALL *PFN_clBuildProgram)(
  cl_program /* program */, cl_uint /* num_devices */,
  const cl_device_id * /* device_list */, const char * /* options */,
  void(CL_CALLBACK * /* pfn_notify */)(cl_program /* program */,
                                       void * /* user_data */),
  void * /* user_data */);

typedef cl_int(CL_API_CALL *PFN_clGetProgramBuildInfo)(
  cl_program /* program */, cl_device_id /* device */,
  cl_program_build_info /* param_name */, size_t /* param_value_size */,
  void * /* param_value */, size_t * /* param_value_size_ret */);

typedef cl_int(CL_API_CALL *PFN_clRetainProgram)(cl_program /* program */);

typedef cl_kernel(CL_API_CALL *PFN_clCreateKernel)(
  cl_program /* program */, const char * /* kernel_name */,
  cl_int * /* errcode_ret */);

typedef cl_int(CL_API_CALL *PFN_clSetKernelArg)(cl_kernel /* kernel */,
                                                cl_uint /* arg_index */,
                                                size_t /* arg_size */,
                                                const void * /* arg_value */);

typedef cl_int(CL_API_CALL *PFN_clEnqueueNDRangeKernel)(
  cl_command_queue /* command_queue */, cl_kernel /* kernel */,
  cl_uint /* work_dim */, const size_t * /* global_work_offset */,
  const size_t * /* global_work_size */, const size_t * /* local_work_size */,
  cl_uint /* num_events_in_wait_list */, const cl_event * /* event_wait_list */,
  cl_event * /* event */);

typedef cl_int(CL_API_CALL *PFN_clGetEventProfilingInfo)(
  cl_event /* event */, cl_profiling_info /* param_name */,
  size_t /* param_value_size */, void * /* param_value */,
  size_t * /* param_value_size_ret */);

typedef cl_int(CL_API_CALL *PFN_clRetainContext)(cl_context /* context */);

typedef cl_int(CL_API_CALL *PFN_clReleaseContext)(cl_context /* context */);

typedef cl_int(CL_API_CALL *PFN_clRetainCommandQueue)(
  cl_command_queue /* command_queue */);

typedef cl_int(CL_API_CALL *PFN_clReleaseCommandQueue)(
  cl_command_queue /* command_queue */);

typedef cl_int(CL_API_CALL *PFN_clReleaseMemObject)(cl_mem /* memobj */);

extern PFN_clGetPlatformIDs clGetPlatformIDs;
extern PFN_clGetDeviceIDs clGetDeviceIDs;
extern PFN_clCreateContext clCreateContext;
extern PFN_clCreateCommandQueue clCreateCommandQueue;
extern PFN_clCreateBuffer clCreateBuffer;
extern PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
extern PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
extern PFN_clCreateProgramWithSource clCreateProgramWithSource;
extern PFN_clBuildProgram clBuildProgram;
extern PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
extern PFN_clRetainProgram clRetainProgram;
extern PFN_clCreateKernel clCreateKernel;
extern PFN_clSetKernelArg clSetKernelArg;
extern PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
extern PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
extern PFN_clRetainContext clRetainContext;
extern PFN_clReleaseContext clReleaseContext;
extern PFN_clRetainCommandQueue clRetainCommandQueue;
extern PFN_clReleaseCommandQueue clReleaseCommandQueue;
extern PFN_clReleaseMemObject clReleaseMemObject;

} // namespace nntrainer::opencl

#endif // GPU_CL_OPENCL_LOADER_HPP_
