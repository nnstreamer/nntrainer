// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_interface.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Interface for GPU tensor operations
 *
 */

#ifndef __CL_INTERFACE_H_
#define __CL_INTERFACE_H_

#include "cl_operations/cl_sgemv.hpp"

namespace nntrainer {

/**
 * @brief     sgemv computation on GPU : Y = alpha*A*X + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float number
 * @param[in] beta float number
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 */
void gpu_sgemv(const float *A, const float *X, float *Y, float alpha,
               float beta, unsigned int rows, unsigned int cols) {
  static internal::GpuCLSgemv cl_gpu_sgemv;
  cl_gpu_sgemv.CLSgemv(A, X, Y, alpha, beta, rows, cols);
}
} // namespace nntrainer

#endif
