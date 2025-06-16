// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 *
 * @file    opencl_device.h
 * @date    15 Jun 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for device management
 *
 */

#ifndef __OPENCL_DEVICE_H__
#define __OPENCL_DEVICE_H__

#include "CL/cl.h"

#include <cstring>
#include <string>
#include <vector>

#include <nntrainer_log.h>

#define CL_CHECK(expression)                                                   \
  do {                                                                         \
    if (cl_int expression_result = (expression)) {                             \
      ml_loge("Expression '%s' failed with error code : %d", #expression,      \
              expression_result);                                              \
    }                                                                          \
  } while (false)

namespace nntrainer::opencl {
/**
 * @class Wrapper for OpenCL device retrieval, creation and info
 * @brief OpenCL device wrapper
 *
 */
class OpenCLDevice {
private:
  using OpenCLDeviceList = std::vector<OpenCLDevice>;

private:
  cl_device_id handle_ = nullptr;

private:
  OpenCLDevice(const cl_device_id device) : handle_(device) {}

public:
  OpenCLDevice() = default;
  ~OpenCLDevice() = default;

  /**
   * @brief Retrieves OpenCL devices
   *
   * @return List of all OpenCL devices of any type available on system
   */
  static OpenCLDeviceList GetDeviceList();

  /**
   * @brief Retrieves OpenCL devices
   *
   * @return List of all OpenCL devices of any type available on system
   */
  inline cl_device_id GetHandle() const { return handle_; }

  inline std::string GetInfoString(const cl_device_info info) const {
    char buffer[256] = {};
    std::memset(buffer, 0x00, sizeof(buffer));
    CL_CHECK(clGetDeviceInfo(handle_, info, sizeof(buffer), buffer, nullptr));

    return std::string(buffer);
  }

  template <typename T> inline T GetInfoValue(const cl_device_info info) const {
    T result = {};
    CL_CHECK(clGetDeviceInfo(handle_, info, sizeof(T), &result, nullptr));

    return result;
  }
};
} // namespace nntrainer::opencl
#endif // __OPENCL_DEVICE_H__
