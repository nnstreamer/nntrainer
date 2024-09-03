// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_context_manager.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for context management
 *
 */

#include "opencl_context_manager.h"

#include <vector>

#include "opencl_loader.h"

#include <nntrainer_log.h>

namespace nntrainer::opencl {

/**
 * @brief Get the global instance object
 *
 * @return ContextManager global instance
 */
ContextManager &ContextManager::GetInstance() {
  static ContextManager instance;
  return instance;
}

/**
 * @brief Get the OpenCL context object
 *
 * @return const cl_context
 */
const cl_context &ContextManager::GetContext() {
  // loading the OpenCL library and required functions
  LoadOpenCL();

  if (context_) {
    // increments the context reference count
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

    // increments the context reference count
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
    // decrements the context reference count
    clReleaseContext(context_);
  }
}

/**
 * @brief Get the Device Id object
 *
 * @return const cl_device_id
 */
const cl_device_id ContextManager::GetDeviceId() { return device_id_; }

/**
 * @brief Destroy the Context Manager object
 *
 */
ContextManager::~ContextManager() {
  if (context_) {
    // decrements the context reference count
    clReleaseContext(context_);
    context_ = nullptr;
  }
}

/**
 * @brief Create a Default GPU Device object
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateDefaultGPUDevice() {
  cl_uint num_platforms;

  // returns number of OpenCL supported platforms
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d", status);
    return false;
  }
  if (num_platforms == 0) {
    ml_loge("No supported OpenCL platform.");
    return false;
  }

  // getting the platform IDs
  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d", status);
    return false;
  }

  // platform is a specific OpenCL implementation, for instance ARM
  cl_platform_id platform_id_ = platforms[0];

  cl_uint num_devices;

  // getting available GPU devices
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

  // getting the GPU device IDs
  std::vector<cl_device_id> devices(num_devices);
  status = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, num_devices,
                          devices.data(), nullptr);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceIDs returned %d", status);
    return false;
  }

  // setting the first GPU ID and platform (ARM)
  device_id_ = devices[0];
  platform_id_ = platform_id_;

#ifdef ENABLE_FP16
  // check for fp16 (half) support available on device
  // getting extensions
  size_t extension_size;
  status =
    clGetDeviceInfo(device_id_, CL_DEVICE_EXTENSIONS, 0, NULL, &extension_size);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceInfo returned %d", status);
    return false;
  }

  char *extensions = new char[extension_size];
  status = clGetDeviceInfo(device_id_, CL_DEVICE_EXTENSIONS, extension_size,
                           extensions, NULL);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceInfo returned %d", status);
    return false;
  }

  if (std::string(extensions).find("cl_khr_fp16") == std::string::npos) {
    ml_loge("fp16 (half) is not supported by device");
    return false;
  }
  delete[] extensions;
#endif

  return true;
}

/**
 * @brief Create OpenCL context
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateCLContext() {
  int error_code;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_id_, 0};

  // creating valid ARM GPU OpenCL context, will return NULL with error code if
  // fails
  context_ =
    clCreateContext(properties, 1, &device_id_, nullptr, nullptr, &error_code);
  if (!context_) {
    ml_loge("Failed to create a compute context. OpenCL error code: %d",
            error_code);
    return false;
  }

  return true;
}
} // namespace nntrainer::opencl
