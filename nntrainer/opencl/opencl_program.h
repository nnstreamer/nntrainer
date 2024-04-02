// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_program.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for program management
 *
 */

#ifndef __OPENCL_PROGRAM_H__
#define __OPENCL_PROGRAM_H__

#include <string>

#include "third_party/cl.h"

namespace nntrainer::opencl {

/**
 * @class Program contains wrapper to manage OpenCL program
 * @brief Weapper for OpenCL program
 *
 */
class Program {
  cl_program program_{nullptr};

  /**
   * @brief Build OpenCL program
   *
   * @param device_id OpenCL device id
   * @param compiler_options string compiler options
   * @param binaryCreated true if binary is already present false otherwise
   * @return true if successful or false otherwise
   */
  bool BuildProgram(cl_device_id device_id, const std::string &compiler_options,
                    bool binaryCreated = false);

  /**
   * @brief Utility to get program info and save kernel binaries
   *
   * @param device_id OpenCL device id
   * @return true if successful or false otherwise
   */
  bool GetProgramInfo(cl_device_id device_id);

  /**
   * @brief Get the information on the program build
   *
   * @param device_id OpenCL device id
   * @param info flag for the information to fetch
   * @return std::string
   */
  std::string GetProgramBuildInfo(cl_device_id device_id,
                                  cl_program_build_info info);

public:
  static const std::string DEFAULT_KERNEL_PATH;

  /**
   * @brief Create OpenCL program from source
   *
   * @param context OpenCL context
   * @param device_id OpenCL device id
   * @param code kernel source code string
   * @param compiler_options string compiler options
   * @return true if successful or false otherwise
   */
  bool CreateCLProgram(const cl_context &context, const cl_device_id &device_id,
                       const std::string &code,
                       const std::string &compiler_options);

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
  bool CreateCLProgramWithBinary(const cl_context &context,
                                 const cl_device_id &device_id, size_t size,
                                 unsigned char *binary, std::string binary_name,
                                 const std::string &compiler_options);

  /**
   * @brief Get the Program object
   *
   * @return const cl_program
   */
  const cl_program &GetProgram();
};
} // namespace nntrainer::opencl
#endif // __OPENCL_PROGRAM_H__
