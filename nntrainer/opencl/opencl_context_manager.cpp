// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_context_manager.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for context management
 *
 */

#include "opencl_context_manager.hpp"

#include <vector>

#include "opencl_loader.hpp"

#include <nntrainer_log.h>

namespace nntrainer::internal {

ContextManager &ContextManager::GetInstance() {
  static ContextManager instance;
  return instance;
}

const cl_context &ContextManager::GetContext() {
  LoadOpenCL();

  if (context_) {
    clRetainContext(context_);
    return context_;
  }

  bool result = true;

  do {
    result = CreateDefaultGPUDevice();
    if (!result) {
      break;
    }

    result = CreateCLContext();
    if (!result) {
      break;
    }

    clRetainContext(context_);

  } while (false);

  if (!result) {
    ml_loge("Failed to create OpenCL Context");
    context_ = nullptr;
  }

  return context_;
}

void ContextManager::ReleaseContext() {
  if (context_) {
    clReleaseContext(context_);
  }
}

const cl_device_id ContextManager::GetDeviceId() { return device_id_; }

ContextManager::~ContextManager() {
  if (context_) {
    clReleaseContext(context_);
    context_ = nullptr;
  }
}

bool ContextManager::CreateDefaultGPUDevice() {
  cl_uint num_platforms;
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d", status);
    return false;
  }
  if (num_platforms == 0) {
    ml_loge("No supported OpenCL platform.");
    return false;
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d", status);
    return false;
  }

  cl_platform_id platform_id_ = platforms[0];

  cl_uint num_devices;
  status =
    clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceIDs returned %d", status);
    return false;
  }
  if (num_devices == 0) {
    ml_loge("No GPU on current platform.");
    return false;
  }

  std::vector<cl_device_id> devices(num_devices);
  status = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, num_devices,
                          devices.data(), nullptr);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceIDs returned %d", status);
    return false;
  }

  device_id_ = devices[0];
  platform_id_ = platform_id_;

  return true;
}

bool ContextManager::CreateCLContext() {
  int error_code;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_id_, 0};
  context_ =
    clCreateContext(properties, 1, &device_id_, nullptr, nullptr, &error_code);
  if (!context_) {
    ml_loge("Failed to create a compute context. OpenCL error code: %d",
            error_code);
    return false;
  }

  return true;
}
} // namespace nntrainer::internal