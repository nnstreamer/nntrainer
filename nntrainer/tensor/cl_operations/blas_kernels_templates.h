// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 *
 * @file	blas_kernels_templates.hpp
 * @date	07 July 2025
 * @brief	Common blas OpenCL kernels (common templates used by
 * blas_kernels_fp16.cpp and blas_kernels.cpp)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @author	Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_TEMPLATES_H__
#define __BLAS_KERNELS_TEMPLATES_H__

#include <blas_kernel_strings.h>
#include <blas_kernels.h>

namespace nntrainer {
template <typename T>
inline static void sgemv_cl_internal(ClContext::SharedPtrClKernel kernel,
                                     const T *matAdata, const T *vecXdata,
                                     T *vecYdata, unsigned int dim1,
                                     unsigned int dim2, unsigned int lda) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t dim1_size = sizeof(T) * dim1;
  size_t dim2_size = sizeof(T) * dim2;
  size_t dim1_dim2_size = sizeof(T) * dim1 * dim2;

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim1_dim2_size, matAdata);
  if (!result) {
    return;
  }

  result = clbuffInstance.getInBufferB()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim2_size, vecXdata);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim1_size, vecYdata);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, clbuffInstance.getInBufferB(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(2, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &dim2, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(4, &lda, sizeof(int));
  if (!result) {
    return;
  }

  const int work_groups_count[3] = {(int)dim1, 1, 1};
  const int work_group_size[3] = {1, 1, 1};

  result = opencl::CommandQueueManager::Global().DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, dim1_size, vecYdata);
  if (!result) {
    return;
  }
}

template <typename T>
T dot_cl_internal(ClContext::SharedPtrClKernel kernel, const T *vecAdata,
                  const T *vecXdata, unsigned int dim1) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  T cl_ret = 0;

  do {
    size_t dim1_size = sizeof(T) * dim1;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecXdata);
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(1, clbuffInstance.getInBufferB(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel->SetKernelArguments(3, clbuffInstance.getOutBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {1, 1, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, sizeof(T), &cl_ret);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

template <typename T>
inline static void
sgemm_cl_internal(ClContext::SharedPtrClKernel kernel, bool TransA, bool TransB,
                  const T *A, const T *B, T *C, unsigned int M, unsigned int N,
                  unsigned int K, unsigned int lda, unsigned int ldb,
                  unsigned int ldc) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // sizes will be same for transpose
  size_t m_k_size = M * K * sizeof(T);
  size_t k_n_size = K * N * sizeof(T);
  size_t m_n_size = M * N * sizeof(T);

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, m_k_size, A);
  if (!result) {
    return;
  }

  result = clbuffInstance.getInBufferB()->WriteDataRegion(
    blas_cc->command_queue_inst_, k_n_size, B);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, m_n_size, C);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, clbuffInstance.getInBufferB(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(2, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &M, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(4, &N, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(5, &K, sizeof(int));
  if (!result) {
    return;
  }

  const int tiled_size = 16;
  const int work_groups_count[3] = {
    (int)((N + tiled_size - 1) / tiled_size) * tiled_size,
    (int)((M + tiled_size - 1) / tiled_size) * tiled_size, 1}; // test-value

  const int work_group_size[3] = {tiled_size, tiled_size, 1}; // test-value

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, m_n_size, C);
  if (!result) {
    return;
  }
}

template <typename T>
inline static void
addition_cl_internal(ClContext::SharedPtrClKernel kernel, const T *input,
                     T *res, unsigned int size_input, unsigned int size_res) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t dim1_size = sizeof(T) * size_input;
  size_t dim2_size = sizeof(T) * size_res;

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim1_size, input);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim2_size, res);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(2, &size_input, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &size_res, sizeof(int));
  if (!result) {
    return;
  }

  const int work_groups_count[3] = {(int)size_res, 1, 1};
  /// @todo: create a group size by device & input
  const int work_group_size[3] = {1, 1, 1}; // test-value
  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, dim2_size, res);

  if (!result) {
    return;
  }
}

template <typename T>
inline static void sscal_cl_internal(ClContext::SharedPtrClKernel kernel, T *X,
                                     const unsigned int N, const float alpha) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t x_size = N * sizeof(T);

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, x_size, X);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, &alpha, sizeof(float));
  if (!result) {
    return;
  }

  const int work_groups_count[3] = {(int)N, 1, 1};
  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, x_size, X);
  if (!result) {
    return;
  }
}

template <typename T>
inline static void transpose_cl_axis_internal(
  ClContext::SharedPtrClKernel kernel, const T *in, T *res,
  unsigned int input_batch_size, unsigned int input_channels,
  unsigned int input_height, unsigned int input_width, unsigned int axis) {

  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t dim_size =
    sizeof(T) * input_batch_size * input_height * input_width * input_channels;

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim_size, in);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, dim_size, res);
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(0, clbuffInstance.getInBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(1, clbuffInstance.getOutBufferA(),
                                      sizeof(cl_mem));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(2, &input_batch_size, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(3, &input_channels, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(4, &input_height, sizeof(int));
  if (!result) {
    return;
  }

  result = kernel->SetKernelArguments(5, &input_width, sizeof(int));
  if (!result) {
    return;
  }

  int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
  if (axis == 2)
    work_groups_count[0] = (int)input_channels;

  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel, work_groups_count, work_group_size);
  if (!result) {
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, dim_size, res);
  if (!result) {
    return;
  }
}

} // namespace nntrainer

#endif
