// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_program.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for program management
 *
 */

#include "opencl_program.h"

#include <cstring>
#include <string>
#include <vector>

#include "opencl_loader.h"

#include <nntrainer_log.h>

#define stringify(s) stringify2(s)
#define stringify2(s) #s

namespace nntrainer::opencl {

// defining DEFAULT_KERNEL_PATH
const std::string Program::DEFAULT_KERNEL_PATH = stringify(OPENCL_KERNEL_PATH);

/**
 * @brief Build OpenCL program
 *
 * @param device_id OpenCL device id
 * @param compiler_options string compiler options
 * @param binaryCreated true if binary is already present false otherwise
 * @return true if successful or false otherwise
 */
bool Program::BuildProgram(cl_device_id device_id,
                           const std::string &compiler_options,
                           bool binaryCreated) {
  // clBuildProgram returns NULL with error code if fails
  const int error_code = clBuildProgram(
    program_, 0, nullptr, compiler_options.c_str(), nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to build program executable. OpenCL error code: %d : %s. %s",
      error_code, OpenCLErrorCodeToString(error_code),
      (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());
    return false;
  }

  return true;
}

/**
 * @brief Utility to get program binary
 *
 * @param device_id OpenCL device id
 * @return vector of bytes if successful, empty if there was an error
 */
std::vector<std::byte> Program::GetProgramBinary(cl_device_id device_id) {
  cl_int error_code = CL_SUCCESS;

  size_t binary_size;
  error_code = clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES,
                                sizeof(size_t), &binary_size, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to get program binary size. OpenCL error code: %d : %s. %s",
            error_code, OpenCLErrorCodeToString(error_code),
            (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());
    return {};
  }

  // Read the binary
  std::vector<std::byte> binary(binary_size);
  std::byte *binary_data = binary.data();
  error_code = clGetProgramInfo(program_, CL_PROGRAM_BINARIES,
                                sizeof(&binary_data), &binary_data, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to get program binary data. OpenCL error code: %d : %s. %s",
            error_code, OpenCLErrorCodeToString(error_code),
            (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());

    return {};
  }

  return binary;
}

/**
 * @brief Get the information on the program build
 *
 * @param device_id OpenCL device id
 * @param info flag for the information to fetch
 * @return std::string
 */
std::string Program::GetProgramBuildInfo(cl_device_id device_id,
                                         cl_program_build_info info) {
  size_t size;

  // getting the size of the string informationz
  cl_int error_code =
    clGetProgramBuildInfo(program_, device_id, info, 0, nullptr, &size);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to GetProgramBuildInfo. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return "";
  }

  // getting the actual information
  std::string result(size - 1, 0);
  error_code =
    clGetProgramBuildInfo(program_, device_id, info, size, &result[0], nullptr);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to GetProgramBuildInfo. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return "";
  }
  return result;
}

std::string Program::GetDefaultCompilerOptions() const {
  return "-cl-std=CL3.0 -cl-mad-enable -cl-unsafe-math-optimizations "
         "-cl-finite-math-only -cl-fast-relaxed-math ";
}

/**
 * @brief Create OpenCL program from source
 *
 * @param context OpenCL context
 * @param device_id
 * @param code kernel source code string
 * @param compiler_options
 * @return true if successful or false otherwise
 */
bool Program::CreateCLProgram(const cl_context &context,
                              const cl_device_id &device_id,
                              const std::string &code,
                              const std::string &compiler_options) {
  int error_code;
  const char *source = code.c_str();

  // returns NULL with error code if fails
  program_ =
    clCreateProgramWithSource(context, 1, &source, nullptr, &error_code);
  if (!program_ || error_code != CL_SUCCESS) {
    ml_loge("Failed to create compute program. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  // building the created program
  return BuildProgram(device_id,
                      GetDefaultCompilerOptions() + compiler_options);
}

/**
 * @brief Create OpenCL program from pre compiled binary
 *
 * @param context OpenCL context
 * @param device_id OpenCL device id
 * @param size binary file size
 * @param binary data saved as binary
 * @param binary_name name of binary file for logging
 * @param compiler_options string compiler options
 * @return true if successful or false otherwise
 */
bool Program::CreateCLProgramWithBinary(const cl_context &context,
                                        const cl_device_id &device_id,
                                        const std::vector<std::byte> &binary,
                                        std::string binary_name,
                                        const std::string &compiler_options) {

  int error_code;
  int binary_status;
  const size_t size = binary.size();
  const unsigned char *binary_data =
    reinterpret_cast<const unsigned char *>(binary.data());

  program_ = clCreateProgramWithBinary(
    context, 1, &device_id, &size, &binary_data, &binary_status, &error_code);
  if (!program_ || error_code != CL_SUCCESS) {
    ml_loge("Failed to create compute program. OpenCL error code: %d : %s, "
            "binary status: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code), binary_status,
            OpenCLErrorCodeToString(binary_status));
    return false;
  }

  ml_logi("Loaded program from binary for: %s", binary_name.c_str());

  return BuildProgram(device_id, compiler_options, true);
}

std::size_t Program::GetKernelHash(const std::string &code,
                                   const std::string &compiler_options) {
  return std::hash<std::string>{}(GetDefaultCompilerOptions() +
                                  compiler_options + code);
}

/**
 * @brief Get the Program object
 *
 * @return const cl_program
 */
const cl_program &Program::GetProgram() { return program_; }

} // namespace nntrainer::opencl
