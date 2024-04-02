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
#include <fstream>
#include <string>

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
    ml_loge("Failed to build program executable. OpenCL error code: %d. %s",
            error_code,
            (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());
    return false;
  }

  // saving kernel binary
  if (!binaryCreated)
    return GetProgramInfo(device_id);

  return true;
}

/**
 * @brief Utility to get program info and save kernel binaries
 *
 * @param device_id OpenCL device id
 * @return true if successful or false otherwise
 */
bool Program::GetProgramInfo(cl_device_id device_id) {
  // since only one GPU is being used
  unsigned int num_devices = 1;

  cl_int error_code = CL_SUCCESS;

  // Read the binary size
  size_t binaries_size[num_devices];
  error_code =
    clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t) * num_devices, binaries_size, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to get program binary size. OpenCL error code: %d. %s",
            error_code,
            (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());
    return false;
  }

  // Read the kernel name size
  size_t kernel_names_size;
  error_code = clGetProgramInfo(program_, CL_PROGRAM_KERNEL_NAMES, 0, nullptr,
                                &kernel_names_size);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to get program kernel name size. OpenCL error code: %d. %s",
            error_code,
            (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());
    return false;
  }

  // getting the kernel names
  char kernel_names[kernel_names_size];
  error_code = clGetProgramInfo(program_, CL_PROGRAM_KERNEL_NAMES,
                                kernel_names_size, kernel_names, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to get program kernel names. OpenCL error code: %d. %s",
            error_code,
            (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());
    return false;
  } else {
    ml_logi("Saving kernel binary for: %s", std::string(kernel_names).c_str());
  }

  // Read the binary
  size_t binaries_ptr_alloc_size = sizeof(unsigned char *) * num_devices;
  unsigned char *binaries_ptr[num_devices];

  for (unsigned int i = 0; i < num_devices; ++i) {
    binaries_ptr[i] = new unsigned char[binaries_size[i]];
  }

  error_code = clGetProgramInfo(program_, CL_PROGRAM_BINARIES,
                                binaries_ptr_alloc_size, binaries_ptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to get program binary data. OpenCL error code: %d. %s",
            error_code,
            (GetProgramBuildInfo(device_id, CL_PROGRAM_BUILD_LOG)).c_str());

    // cleanup
    for (unsigned int i = 0; i < num_devices; ++i) {
      delete[] binaries_ptr[i];
    }
    return false;
  }

  // Write the binary to file
  // All kernels in the program will be saved in the binary file
  for (unsigned int i = 0; i < num_devices; ++i) {
    std::ofstream fs(Program::DEFAULT_KERNEL_PATH + "/" +
                       std::string(kernel_names) + "_kernel.bin",
                     std::ios::out | std::ios::binary | std::ios::app);
    if (!fs) {
      ml_loge(
        "opencl_program: could not find directory to save kernel binary - %s",
        Program::DEFAULT_KERNEL_PATH.c_str());

      // cleanup
      for (unsigned int i = 0; i < num_devices; ++i) {
        delete[] binaries_ptr[i];
      }
      return false;
    }
    fs.write((char *)binaries_ptr[i], binaries_size[i]);
    fs.close();
  }

  // cleanup
  for (unsigned int i = 0; i < num_devices; ++i) {
    delete[] binaries_ptr[i];
  }

  return true;
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
    ml_loge("Failed to GetProgramBuildInfo. OpenCL error code: %d", error_code);
    return "";
  }

  // getting the actual information
  std::string result(size - 1, 0);
  error_code =
    clGetProgramBuildInfo(program_, device_id, info, size, &result[0], nullptr);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to GetProgramBuildInfo. OpenCL error code: %d", error_code);
    return "";
  }
  return result;
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
    ml_loge("Failed to create compute program. OpenCL error code: %d",
            error_code);
    return false;
  }

  // building the created program
  return BuildProgram(device_id, compiler_options);
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
                                        size_t size, unsigned char *binary,
                                        std::string binary_name,
                                        const std::string &compiler_options) {

  int error_code;
  const cl_device_id device_list[] = {device_id};
  const size_t lengths[] = {size};
  const unsigned char *binaries[] = {binary};

  program_ = clCreateProgramWithBinary(context, 1, device_list, lengths,
                                       binaries, NULL, &error_code);
  if (!program_ || error_code != CL_SUCCESS) {
    ml_loge("Failed to create compute program. OpenCL error code: %d",
            error_code);
    return false;
  }

  ml_logi("Loaded program from binary for: %s", binary_name.c_str());

  return BuildProgram(device_id, compiler_options, true);
}

/**
 * @brief Get the Program object
 *
 * @return const cl_program
 */
const cl_program &Program::GetProgram() { return program_; }

} // namespace nntrainer::opencl
