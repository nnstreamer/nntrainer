// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.cpp
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_strings.h>
#include <blas_kernels.h>

#define CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED // to disable deprecation warnings
#include "clblast.h"

namespace nntrainer {

void sgemv_q6_k_cl(const void *matAdata, const float *vecXdata, float *vecYdata,
                   unsigned int M, unsigned int N) {
  bool result = false;

  ClContext::SharedPtrClKernel kernel_q6_k_sgemv_ptr;

  kernel_q6_k_sgemv_ptr =
    blas_cc->registerClKernel(getQ6KSgemvClKernel(), "kernel_mul_mv_q6_K_f32");

  if (!kernel_q6_k_sgemv_ptr) {
    ml_loge("Failed to register kernel_q6_k_sgemv_ptr");
    return;
  }

  int q6k_size = 210 * M * N / 256;

  result = clbuffInstance.getInBufferA()->WriteDataRegion(
    blas_cc->command_queue_inst_, q6k_size, matAdata);
  if (!result) {
    ml_loge("Failed to write data to input buffer A for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = clbuffInstance.getInBufferB()->WriteDataRegion(
    blas_cc->command_queue_inst_, M * sizeof(float), vecXdata);
  if (!result) {
    ml_loge("Failed to write data to input buffer B for kernel_q6_k_sgemv_ptr");
    return;
  }

  int ne00 = M; // number of rows in matrix X
  int ne01 = N; // number of columns in matrix X
  int ne02 = 1; // number of channels in matrix X
  int ne10 = M; // number of rows in vector A
  int ne11 = 1; // number of columns in vector A
  int ne12 = 1; // number of channels in vector A
  int ne13 = 1; // number of channels in vector A (Need to check)
  int ne0 = N;  // number of rows in output vector Y
  int ne1 = 1;  // number of columns in output vector Y

  int r2 = 1; // number of batches in vector A
  int r3 = 1; // number of batches in matrix X

  int nth0 = 2;
  int nth1 = 16;

  cl_ulong offset0 = 0;
  cl_ulong offset1 = 0;
  cl_ulong offsetd = 0;

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(
    0, clbuffInstance.getInBufferA(), sizeof(cl_mem));

  if (!result) {
    ml_loge("Failed to set kernel argument 0 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(1, &offset0, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 1 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(
    2, clbuffInstance.getInBufferB(), sizeof(cl_mem));

  if (!result) {
    ml_loge("Failed to set kernel argument 2 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(3, &offset1, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 3 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(
    4, clbuffInstance.getOutBufferA(), sizeof(cl_mem));

  if (!result) {
    ml_loge("Failed to set kernel argument 4 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(5, &offsetd, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 5 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(6, &ne00, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 6 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(7, &ne01, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 7 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(8, &ne02, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 8 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(9, &ne10, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 9 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(10, &ne12, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 10 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(11, &ne0, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 11 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(12, &ne1, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 12 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(13, &r2, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 13 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(14, &r3, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 14 for kernel_q6_k_sgemv_ptr");
    return;
  }

#define N_SIMDWIDTH 16
#define N_SIMDGROUP 2

  const int work_groups_count[3] = {((ne0 + N_SIMDGROUP - 1) / N_SIMDGROUP) *
                                      (N_SIMDGROUP * N_SIMDWIDTH),
                                    ne1, 1};
  /// @todo: create a group size by device & input
  const int work_group_size[3] = {32, 1, 1};

  result = opencl::CommandQueueManager::GetInstance().DispatchCommand(
    kernel_q6_k_sgemv_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel q6_k_sgemv");
    return;
  }

  result = clbuffInstance.getOutBufferA()->ReadDataRegion(
    blas_cc->command_queue_inst_, N * sizeof(float), vecYdata);
  if (!result) {
    ml_loge(
      "Failed to read data from the output buffer for kernel_q6_k_sgemv_ptr");

    return;
  }
}

void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemv_ptr;

    if (TransA) {
      kernel_sgemv_ptr =
        blas_cc->registerClKernel(getSgemvClKernel(), "sgemv_cl");
    } else {
      kernel_sgemv_ptr = blas_cc->registerClKernel(getSgemvClNoTransKernel(),
                                                   "sgemv_cl_noTrans");
    }

    if (!kernel_sgemv_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;
    size_t dim2_size = sizeof(float) * dim2;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1 * dim2 * sizeof(float), matAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim2_size, vecXdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(
      1, clbuffInstance.getInBufferB(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(
      2, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(3, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1};

    result = opencl::CommandQueueManager::GetInstance().DispatchCommand(
      kernel_sgemv_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, dim1_size, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1) {

  bool result = false;

  float cl_ret = 0;

  do {
    ClContext::SharedPtrClKernel kernel_dot_ptr =
      blas_cc->registerClKernel(getDotClKernel(), "dot_cl");
    if (!kernel_dot_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;

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

    result = kernel_dot_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(
      1, clbuffInstance.getInBufferB(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(
      3, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_dot_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, sizeof(float), &cl_ret);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc) {

  std::string kernel_func_;
  std::string sgemm_cl_kernel_;

  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans";
    sgemm_cl_kernel_ = getSgemmClNoTransKernel();
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA";
    sgemm_cl_kernel_ = getSgemmClTransAKernel();
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB";
    sgemm_cl_kernel_ = getSgemmClTransBKernel();
  } else {
    kernel_func_ = "sgemm_cl_transAB";
    sgemm_cl_kernel_ = getSgemmClTransABKernel();
  }

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemm_ptr =
      blas_cc->registerClKernel(sgemm_cl_kernel_, kernel_func_);
    if (!kernel_sgemm_ptr) {
      break;
    }

    // sizes will be same for transpose
    size_t m_k_size = M * K * sizeof(float);
    size_t k_n_size = K * N * sizeof(float);
    size_t m_n_size = M * N * sizeof(float);

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, m_k_size, A);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      blas_cc->command_queue_inst_, k_n_size, B);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, m_n_size, C);
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(
      1, clbuffInstance.getInBufferB(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(
      2, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(3, &M, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(4, &N, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(5, &K, sizeof(int));
    if (!result) {
      break;
    }
    const int tiled_size = 16;
    const int work_groups_count[3] = {
      (int)((N + tiled_size - 1) / tiled_size) * tiled_size,
      (int)((M + tiled_size - 1) / tiled_size) * tiled_size, 1}; // test-value

    const int work_group_size[3] = {tiled_size, tiled_size, 1}; // test-value

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_sgemm_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, m_n_size, C);
    if (!result) {
      break;
    }

  } while (false);
}

void addition_cl(const float *input, float *res, unsigned int size_input,
                 unsigned int size_res) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_addition_ptr =
      blas_cc->registerClKernel(getAdditionClKernel(), "addition_cl");
    if (!kernel_addition_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * size_input;
    size_t dim2_size = sizeof(float) * size_res;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim1_size, input);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim2_size, res);
    if (!result) {
      break;
    }

    result = kernel_addition_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }
    result = kernel_addition_ptr->SetKernelArguments(
      1, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_addition_ptr->SetKernelArguments(2, &size_input, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_addition_ptr->SetKernelArguments(3, &size_res, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)size_res, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value
    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_addition_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, dim2_size, res);

    if (!result) {
      break;
    }

  } while (false);
}

void sscal_cl(float *X, const unsigned int N, const float alpha) {
  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_ptr =
      blas_cc->registerClKernel(getSscalClKernel(), "sscal_cl");

    if (!kernel_ptr) {
      break;
    }

    size_t x_size = N * sizeof(float);

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, x_size, X);
    if (!result) {
      break;
    }

    result = kernel_ptr->SetKernelArguments(0, clbuffInstance.getOutBufferA(),
                                            sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_ptr->SetKernelArguments(1, &alpha, sizeof(float));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)N, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, x_size, X);
    if (!result) {
      break;
    }

  } while (false);
}

void transpose_cl_axis(const float *in, float *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_transpose_ptr;
    switch (axis) {
    case 0:
      kernel_transpose_ptr = blas_cc->registerClKernel(
        getTransposeClKernelAxis0(), "transpose_cl_axis0");
      break;
    case 1:
      kernel_transpose_ptr = blas_cc->registerClKernel(
        getTransposeClKernelAxis1(), "transpose_cl_axis1");
      break;
    case 2:
      kernel_transpose_ptr = blas_cc->registerClKernel(
        getTransposeClKernelAxis2(), "transpose_cl_axis2");
      break;
    default:
      throw std::invalid_argument("failed to register CL kernel");
      break;
    }
    if (!kernel_transpose_ptr) {
      break;
    }

    size_t dim_size = sizeof(float) * input_batch_size * input_height *
                      input_width * input_channels;

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim_size, in);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      blas_cc->command_queue_inst_, dim_size, res);
    if (!result) {
      break;
    }

    result = kernel_transpose_ptr->SetKernelArguments(
      0, clbuffInstance.getInBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_transpose_ptr->SetKernelArguments(
      1, clbuffInstance.getOutBufferA(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_transpose_ptr->SetKernelArguments(2, &input_batch_size,
                                                      sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
    if (axis == 2)
      work_groups_count[0] = (int)input_channels;

    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_transpose_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      blas_cc->command_queue_inst_, dim_size, res);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
