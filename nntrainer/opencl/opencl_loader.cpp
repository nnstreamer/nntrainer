// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_loader.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Load required OpenCL functions
 *
 */

#include "opencl_loader.h"

#include <dynamic_library_loader.h>
#include <nntrainer_log.h>
#include <string>

namespace nntrainer::opencl {

#define LoadFunction(function)                                                 \
  function = reinterpret_cast<PFN_##function>(                                 \
    DynamicLibraryLoader::loadSymbol(libopencl, #function));

/**
 * @brief Declaration of loading function for OpenCL APIs
 *
 * @param libopencl
 */
void LoadOpenCLFunctions(void *libopencl);

static bool open_cl_initialized = false;

/**
 * @brief Loading OpenCL libraries and required function
 *
 * @return true if successfull or false otherwise
 */
bool LoadOpenCL() {
  // check if already loaded
  if (open_cl_initialized) {
    return true;
  }

  void *libopencl = nullptr;

#if defined(_WIN32)
  static const char *kClLibName = "OpenCL.dll";
#else
  static const char *kClLibName = "libOpenCL.so";
#endif

  libopencl =
    DynamicLibraryLoader::loadLibrary(kClLibName, RTLD_NOW | RTLD_LOCAL);
  if (libopencl) {
    LoadOpenCLFunctions(libopencl);
    open_cl_initialized = true;
    return true;
  }

  // record error
  std::string error(DynamicLibraryLoader::getLastError());
  ml_loge("Can not open OpenCL library on this device - %s", error.c_str());
  return false;
}

/**
 * @brief Retrieves string representation of OpenCL status code
 *
 * @return OpenCL status code as string
 */
const char *OpenCLErrorCodeToString(const cl_int code) {
#define SWITCH_CASE_RETURN(ENUM)                                               \
  case ENUM:                                                                   \
    return #ENUM

  switch (code) {
    SWITCH_CASE_RETURN(CL_SUCCESS);
    SWITCH_CASE_RETURN(CL_DEVICE_NOT_FOUND);
    SWITCH_CASE_RETURN(CL_DEVICE_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_COMPILER_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    SWITCH_CASE_RETURN(CL_OUT_OF_RESOURCES);
    SWITCH_CASE_RETURN(CL_OUT_OF_HOST_MEMORY);
    SWITCH_CASE_RETURN(CL_PROFILING_INFO_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_MEM_COPY_OVERLAP);
    SWITCH_CASE_RETURN(CL_IMAGE_FORMAT_MISMATCH);
    SWITCH_CASE_RETURN(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    SWITCH_CASE_RETURN(CL_BUILD_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_MAP_FAILURE);
#ifdef CL_VERSION_1_1
    SWITCH_CASE_RETURN(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    SWITCH_CASE_RETURN(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
#ifdef CL_VERSION_1_2
    SWITCH_CASE_RETURN(CL_COMPILE_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_LINKER_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_LINK_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_DEVICE_PARTITION_FAILED);
    SWITCH_CASE_RETURN(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
    SWITCH_CASE_RETURN(CL_INVALID_VALUE);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_TYPE);
    SWITCH_CASE_RETURN(CL_INVALID_PLATFORM);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE);
    SWITCH_CASE_RETURN(CL_INVALID_CONTEXT);
    SWITCH_CASE_RETURN(CL_INVALID_QUEUE_PROPERTIES);
    SWITCH_CASE_RETURN(CL_INVALID_COMMAND_QUEUE);
    SWITCH_CASE_RETURN(CL_INVALID_HOST_PTR);
    SWITCH_CASE_RETURN(CL_INVALID_MEM_OBJECT);
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_SAMPLER);
    SWITCH_CASE_RETURN(CL_INVALID_BINARY);
    SWITCH_CASE_RETURN(CL_INVALID_BUILD_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_PROGRAM);
    SWITCH_CASE_RETURN(CL_INVALID_PROGRAM_EXECUTABLE);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_NAME);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_DEFINITION);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_INDEX);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_VALUE);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_ARGS);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_DIMENSION);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_GROUP_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_ITEM_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_GLOBAL_OFFSET);
    SWITCH_CASE_RETURN(CL_INVALID_EVENT_WAIT_LIST);
    SWITCH_CASE_RETURN(CL_INVALID_EVENT);
    SWITCH_CASE_RETURN(CL_INVALID_OPERATION);
    SWITCH_CASE_RETURN(CL_INVALID_GL_OBJECT);
    SWITCH_CASE_RETURN(CL_INVALID_BUFFER_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_MIP_LEVEL);
    SWITCH_CASE_RETURN(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_1
    SWITCH_CASE_RETURN(CL_INVALID_PROPERTY);
#endif
#ifdef CL_VERSION_1_2
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_DESCRIPTOR);
    SWITCH_CASE_RETURN(CL_INVALID_COMPILER_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_LINKER_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#ifdef CL_VERSION_2_0
    SWITCH_CASE_RETURN(CL_INVALID_PIPE_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_QUEUE);
#endif
#ifdef CL_VERSION_2_2
    SWITCH_CASE_RETURN(CL_INVALID_SPEC_ID);
    SWITCH_CASE_RETURN(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif
  default:
    return "(unknown)";
  }
#undef SWITCH_CASE_RETURN
}

/**
 * @brief Utility to load the required OpenCL APIs
 *
 * @param libopencl
 */
void LoadOpenCLFunctions(void *libopencl) {
  LoadFunction(clGetPlatformIDs);
  LoadFunction(clGetDeviceIDs);
  LoadFunction(clGetDeviceInfo);
  LoadFunction(clCreateContext);
  LoadFunction(clCreateCommandQueue);
  LoadFunction(clCreateBuffer);
  LoadFunction(clCreateSubBuffer);
  LoadFunction(clCreateImage);
  LoadFunction(clEnqueueWriteBuffer);
  LoadFunction(clEnqueueReadBuffer);
  LoadFunction(clEnqueueMapBuffer);
  LoadFunction(clEnqueueUnmapMemObject);
  LoadFunction(clEnqueueWriteBufferRect);
  LoadFunction(clEnqueueReadBufferRect);
  LoadFunction(clCreateProgramWithSource);
  LoadFunction(clCreateProgramWithBinary);
  LoadFunction(clBuildProgram);
  LoadFunction(clGetProgramInfo);
  LoadFunction(clGetProgramBuildInfo);
  LoadFunction(clRetainProgram);
  LoadFunction(clCreateKernel);
  LoadFunction(clSetKernelArg);
  LoadFunction(clEnqueueNDRangeKernel);
  LoadFunction(clGetEventProfilingInfo);
  LoadFunction(clRetainContext);
  LoadFunction(clReleaseContext);
  LoadFunction(clRetainCommandQueue);
  LoadFunction(clReleaseCommandQueue);
  LoadFunction(clReleaseMemObject);
  LoadFunction(clFlush);
  LoadFunction(clFinish);
  LoadFunction(clSVMAlloc);
  LoadFunction(clSVMFree);
  LoadFunction(clEnqueueSVMMap);
  LoadFunction(clEnqueueSVMUnmap);
  LoadFunction(clSetKernelArgSVMPointer);
  LoadFunction(clWaitForEvents);
}

PFN_clGetPlatformIDs clGetPlatformIDs;
PFN_clGetDeviceIDs clGetDeviceIDs;
PFN_clGetDeviceInfo clGetDeviceInfo;
PFN_clCreateContext clCreateContext;
PFN_clCreateCommandQueue clCreateCommandQueue;
PFN_clCreateBuffer clCreateBuffer;
PFN_clCreateSubBuffer clCreateSubBuffer;
PFN_clCreateImage clCreateImage;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
PFN_clEnqueueMapBuffer clEnqueueMapBuffer;
PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect;
PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect;
PFN_clCreateProgramWithSource clCreateProgramWithSource;
PFN_clCreateProgramWithBinary clCreateProgramWithBinary;
PFN_clBuildProgram clBuildProgram;
PFN_clGetProgramInfo clGetProgramInfo;
PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
PFN_clRetainProgram clRetainProgram;
PFN_clCreateKernel clCreateKernel;
PFN_clSetKernelArg clSetKernelArg;
PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
PFN_clRetainContext clRetainContext;
PFN_clReleaseContext clReleaseContext;
PFN_clRetainCommandQueue clRetainCommandQueue;
PFN_clReleaseCommandQueue clReleaseCommandQueue;
PFN_clReleaseMemObject clReleaseMemObject;
PFN_clFlush clFlush;
PFN_clFinish clFinish;
PFN_clSVMAlloc clSVMAlloc;
PFN_clSVMFree clSVMFree;
PFN_clEnqueueSVMMap clEnqueueSVMMap;
PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap;
PFN_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
PFN_clWaitForEvents clWaitForEvents;
} // namespace nntrainer::opencl
