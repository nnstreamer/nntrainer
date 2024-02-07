// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_loader.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Load required OpenCL functions
 *
 */

#include "opencl_loader.hpp"

#include <dlfcn.h>

#include <nntrainer_log.h>
#include <string>

namespace nntrainer::internal {

#define LoadFunction(function) \
  function = reinterpret_cast<PFN_##function>(dlsym(libopencl, #function));

void LoadOpenCLFunctions(void *libopencl);

static bool open_cl_initialized = false;

bool LoadOpenCL() {
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

void LoadOpenCLFunctions(void *libopencl) {
  LoadFunction(clGetPlatformIDs);
  LoadFunction(clGetDeviceIDs);
  LoadFunction(clCreateContext);
  LoadFunction(clCreateCommandQueue);
  LoadFunction(clCreateBuffer);
  LoadFunction(clEnqueueWriteBuffer);
  LoadFunction(clEnqueueReadBuffer);
  LoadFunction(clCreateProgramWithSource);
  LoadFunction(clBuildProgram);
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
PFN_clCreateContext clCreateContext;
PFN_clCreateCommandQueue clCreateCommandQueue;
PFN_clCreateBuffer clCreateBuffer;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
PFN_clCreateProgramWithSource clCreateProgramWithSource;
PFN_clBuildProgram clBuildProgram;
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

} // namespace nntrainer::internal
