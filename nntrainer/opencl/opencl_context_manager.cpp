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

#include <iostream>
#include <vector>

#include "opencl_loader.h"

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
    auto error_code = clRetainContext(context_);
    if (error_code != CL_SUCCESS) {
      ml_loge("Failed to specify the OpenCL context to retain. OpenCL error "
              "code: %d",
              error_code);
    }

    return context_;
  }

  bool result = true;

  do {
    result = CreateDefaultDevice();
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
const cl_device_id ContextManager::GetDeviceId() {
  return opencl_device_.GetHandle();
}

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
 * @brief Create a default device object
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateDefaultDevice(cl_device_type type) {
  for (const auto &device : OpenCLDevice::GetDeviceList()) {
    if (device.GetInfoValue<cl_device_type>(CL_DEVICE_TYPE) == type) {
      auto name = device.GetInfoString(CL_DEVICE_NAME);
      auto vendor = device.GetInfoString(CL_DEVICE_VENDOR);

      ml_logi("Found suitable device : %s (%s)", name.c_str(), vendor.c_str());

      opencl_device_ = device;

      return true;
    }
  }

#ifdef ENABLE_FP16
  /// @note This is working incorrectly. For CUDA devices, cl_khr_fp16 is not
  /// listed, but compilation with half-precision works.

  // check for fp16 (half) support available on device
  // getting extensions
  // size_t extension_size;
  // status =
  //   clGetDeviceInfo(device_id_, CL_DEVICE_EXTENSIONS, 0, NULL,
  //   &extension_size);
  // if (status != CL_SUCCESS) {
  //   ml_loge("clGetDeviceInfo returned %d", status);
  //   return false;
  // }

  // std::vector<char> extensions(extension_size);
  // status = clGetDeviceInfo(device_id_, CL_DEVICE_EXTENSIONS, extension_size,
  //                          extensions.data(), NULL);
  // if (status != CL_SUCCESS) {
  //   ml_loge("clGetDeviceInfo returned %d", status);
  //   return false;
  // }

  // if (std::string(extensions.data()).find("cl_khr_fp16") ==
  // std::string::npos) {
  //   ml_loge("fp16 (half) is not supported by device");
  //   return false;
  // }
#endif

  return false;
}

/**
 * @brief Create OpenCL context
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateCLContext() {
  cl_platform_id platform =
    opencl_device_.GetInfoValue<cl_platform_id>(CL_DEVICE_PLATFORM);
  cl_device_id device = opencl_device_.GetHandle();

  int error_code;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform, 0};

  // creating valid ARM GPU OpenCL context, will return NULL with error code if
  // fails
  context_ =
    clCreateContext(properties, 1, &device, nullptr, nullptr, &error_code);
  if (!context_) {
    ml_loge("Failed to create a compute context. OpenCL error code: %d",
            error_code);
    return false;
  }

  return true;
}
} // namespace nntrainer::opencl
