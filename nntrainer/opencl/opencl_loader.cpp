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

#include <dlfcn.h>

#include <nntrainer_log.h>
#include <string>

namespace nntrainer::opencl {

#define LoadFunction(function) \
  function = reinterpret_cast<PFN_##function>(dlsym(libopencl, #function));

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
  static const char *kClLibName = "libOpenCL.so";

  libopencl = dlopen(kClLibName, RTLD_NOW | RTLD_LOCAL);
  if (libopencl) {
    LoadOpenCLFunctions(libopencl);
    open_cl_initialized = true;
    return true;
  }

  // record error
  std::string error(dlerror());
  ml_loge("Can not open OpenCL library on this device - %s", error.c_str());
  return false;
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
  LoadFunction(clEnqueueWriteBuffer);
  LoadFunction(clEnqueueReadBuffer);
  LoadFunction(clEnqueueMapBuffer);
  LoadFunction(clEnqueueUnmapMemObject);
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
}

PFN_clGetPlatformIDs clGetPlatformIDs;
PFN_clGetDeviceIDs clGetDeviceIDs;
PFN_clGetDeviceInfo clGetDeviceInfo;
PFN_clCreateContext clCreateContext;
PFN_clCreateCommandQueue clCreateCommandQueue;
PFN_clCreateBuffer clCreateBuffer;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
PFN_clEnqueueMapBuffer clEnqueueMapBuffer;
PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
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

} // namespace nntrainer::opencl
