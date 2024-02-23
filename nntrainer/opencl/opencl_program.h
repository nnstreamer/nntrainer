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

#ifndef GPU_CL_OPENCL_PROGRAM_HPP_
#define GPU_CL_OPENCL_PROGRAM_HPP_

#include <string>

#include "third_party/cl.h"

namespace nntrainer::opencl {
class Program {
  cl_program program_{nullptr};

  bool BuildProgram(cl_device_id device_id,
                    const std::string &compiler_options);
  std::string GetProgramBuildInfo(cl_device_id device_id,
                                  cl_program_build_info info);

public:
  bool CreateCLProgram(const cl_context &context, const cl_device_id &device_id,
                       const std::string &code,
                       const std::string &compiler_options);
  const cl_program &GetProgram();
};
} // namespace nntrainer::opencl
#endif // GPU_CL_OPENCL_PROGRAM_HPP_
