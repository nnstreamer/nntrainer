// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 *
 * @file    opencl_device.cpp
 * @date    15 Jun 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for device management
 *
 */

#include "opencl_device.h"

namespace nntrainer::opencl {
/**
 * @brief Retrieves OpenCL devices
 *
 * @return List of all OpenCL devices of any type available on system
 */
OpenCLDevice::OpenCLDeviceList OpenCLDevice::GetDeviceList() {
  OpenCLDevice::OpenCLDeviceList result;

  cl_uint platform_count = 0;
  CL_CHECK(clGetPlatformIDs(0, nullptr, &platform_count));

  std::vector<cl_platform_id> platforms(platform_count, nullptr);
  CL_CHECK(clGetPlatformIDs(platform_count, platforms.data(), nullptr));

  for (const auto &platform : platforms) {
    cl_uint device_count = 0;
    CL_CHECK(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count));

    std::vector<cl_device_id> deviecs(device_count, nullptr);
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count,
                            deviecs.data(), nullptr));

    for (const auto &device : deviecs) {
      result.push_back(OpenCLDevice(device));
    }
  }

  return result;
}
} // namespace nntrainer::opencl
