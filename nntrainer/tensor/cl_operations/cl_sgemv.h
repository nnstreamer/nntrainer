// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_sgemv.h
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Experimental SGEMV implementation using OpenCL
 *
 * @note This file is experimental and is kept for testing purpose
 *
 */

#ifndef GPU_CL_SGEMV_HPP_
#define GPU_CL_SGEMV_HPP_

#include <opencl_op_interface.h>

namespace nntrainer::internal {
class GpuCLSgemv : public nntrainer::opencl::GpuCLOpInterface {
  std::string sgemv_kernel_ =
    R"(__kernel void sgemv(const __global float* A, const __global float* X,
                      __global float* Y, float alpha, float beta, unsigned int M, unsigned int N) {
        const int row = get_global_id(0);
        Y[row] = Y[row] * beta;
        for (unsigned int j = 0; j < N; j++){
            Y[row] += alpha * A[row * N + j] * X[j];
        }
    })";

public:
  template <typename T>
  T *CLSgemv(const T *matAdata, const T *vecXdata, T *vecYdata, T alpha, T beta,
             unsigned int dim1, unsigned int dim2);
};
} // namespace nntrainer::internal

#endif // GPU_CL_SGEMV_HPP_
