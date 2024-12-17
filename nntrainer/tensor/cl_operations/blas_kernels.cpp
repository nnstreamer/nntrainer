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
#include <opencl_loader.h>

namespace nntrainer {

void fused_sgemv_cl_rms(const float *matAdata, const float *vecXdata,
                        float *vecYdata, const float *gdata, const float *bdata,
                        bool isAdditionPossible, float epsilon, bool TransA,
                        bool isbias, unsigned int dim1, unsigned int dim2,
                        unsigned int bias_dim1, unsigned int lda, int b, int c,
                        int h, int w) {

  bool result = false;

  do {
    // printf("Starting with sgemv dotcl\n");
    auto tt1 = std::chrono::high_resolution_clock::now();
    ClContext::SharedPtrClKernel kernel_sgemv_ptr;

    if (TransA) {
      kernel_sgemv_ptr =
        cl_context_ref.registerClKernel(sgemv_cl_kernel_, "sgemv_cl");
    } else {
      kernel_sgemv_ptr = cl_context_ref.registerClKernel(
        sgemv_cl_noTrans_kernel_, "sgemv_cl_noTrans");
    }

    if (!kernel_sgemv_ptr) {
      printf("Failed to register sgemv kernel\n");
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;
    size_t dim2_size = sizeof(float) * dim2;
    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          dim1 * dim2 * sizeof(float), true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim2_size, true,
                          nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_, dim1_size, true,
                          nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, matAdata);
    if (!result) {
      printf("Failed to write inputA data ind dotcl sgemv\n");
      break;
    }

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
    if (!result) {
      printf("Failed to write inputX data in dotcl sgemv\n");
      break;
    }

    result = inOutY.WriteData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      printf("Failed to write inOutY data in dotcl sgemv\n");
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for inputA in dotcl sgemv\n");
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for inputX in dotcl sgemv\n");
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for inOutY in dotcl sgemv\n");
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(3, &dim2, sizeof(int));
    if (!result) {
      printf("Failed to set argument for dim2 in dotcl sgemv\n");
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      printf("Failed to set argument for lda in dotcl sgemv\n");
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sgemv_ptr, work_groups_count, work_group_size);

    if (!result) {
      printf("Failed to dispatch sgemv kernel\n");
      break;
    }

    // printf("Done with sgemv dotcl\n");
    //  const cl_event *ptr_sgemv = &sgemv_event;
    //  result = nntrainer::opencl::clWaitForEvents(1, ptr_sgemv);
    //  if (!result) {
    //    throw std::runtime_error("Failed to wait for SGEMV kernel event");
    //  }

    if (isbias) {
      if (isAdditionPossible) {
        //  cl_event add_event;
        ClContext::SharedPtrClKernel kernel_addition_ptr =
          cl_context_ref.registerClKernel(addition_cl_kernel_, "addition_cl");

        if (!kernel_addition_ptr) {
          printf("Failed to register addition kernel\n");
          break;
        }

        size_t bias_size = sizeof(float) * bias_dim1;
        // size_t dim2_size = sizeof(float) * size_res; // result size -> dim1

        opencl::Buffer inputC(cl_context_ref.context_inst_, bias_size, true,
                              nullptr);

        result = inputC.WriteData(cl_context_ref.command_queue_inst_, bdata);
        if (!result) {
          printf("Failed to write inputC data in addition\n");
          break;
        }

        result =
          kernel_addition_ptr->SetKernelArguments(0, &inputC, sizeof(cl_mem));
        if (!result) {
          printf("Failed to set argument for inputC in addition\n");
          break;
        }

        result =
          kernel_addition_ptr->SetKernelArguments(1, &inOutY, sizeof(cl_mem));
        if (!result) {
          printf("Failed to set argument for inOutY in addition\n");
          break;
        }

        result =
          kernel_addition_ptr->SetKernelArguments(2, &bias_dim1, sizeof(int));
        if (!result) {
          printf("Failed to set argument for bias_dim1 in addition\n");
          break;
        }

        result = kernel_addition_ptr->SetKernelArguments(3, &dim1, sizeof(int));
        if (!result) {
          printf("Failed to set argument for dim1 in addition\n");
          break;
        }

        const int work_groups_count_add[3] = {(int)bias_dim1, 1, 1};
        const int work_group_size_add[3] = {32, 32, 1}; // test-value

        result = cl_context_ref.command_queue_inst_.DispatchCommand(
          kernel_addition_ptr, work_groups_count_add, work_group_size_add);

        if (!result) {
          printf("Failed to dispatch addition kernel\n");
          break;
        }
      } else {
        // throw std::invalid_argument(
        //   "Error: Broadcasting not supported for these dimensions!");
        printf("Broadcasting not supported for these dimensions!\n");
      }
    }

    //  cl_event rms_event;
    ClContext::SharedPtrClKernel kernel_rmsnorm_ptr =
      cl_context_ref.registerClKernel(rmsnorm_cl_kernel_new, "rmsnorm_cl");

    if (!kernel_rmsnorm_ptr) {
      printf("Failed to register rmsnorm kernel\n");
      break;
    }

    // for this the input is nothing but the result from above kernels, which is
    // result only
    opencl::Buffer gammabuf(cl_context_ref.context_inst_, w * sizeof(float),
                            true, nullptr);

    opencl::Buffer resultbuf(
      cl_context_ref.context_inst_, dim1_size, true,
      nullptr); // to store the data of the dot, add and rms

    result = gammabuf.WriteData(cl_context_ref.command_queue_inst_, gdata);
    if (!result) {
      printf("Failed to write gamma data in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(0, &inOutY, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for inputA in rmsnorm\n");
      break;
    }

    result =
      kernel_rmsnorm_ptr->SetKernelArguments(1, &resultbuf, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for inOutY in rmsnorm\n");
      break;
    }

    result =
      kernel_rmsnorm_ptr->SetKernelArguments(2, &gammabuf, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for gammabuf in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(4, &b, sizeof(int));
    if (!result) {
      printf("Failed to set argument for b in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(3, &epsilon, sizeof(float));
    if (!result) {
      printf("Failed to set argument for epsilon in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(5, &c, sizeof(int));
    if (!result) {
      printf("Failed to set argument for c in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(6, &h, sizeof(int));
    if (!result) {
      printf("Failed to set argument for h in rmsnorm\n");
      break;
    }
    result = kernel_rmsnorm_ptr->SetKernelArguments(7, &w, sizeof(int));
    if (!result) {
      printf("Failed to set argument for w in rmsnorm\n");
      break;
    }

    const int work_groups_count_rms[3] = {b * c, h, 1};
    const int work_group_size_rms[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_rmsnorm_ptr, work_groups_count_rms, work_group_size_rms);

    if (!result) {
      printf("Failed to dispatch rmsnorm kernel\n");
      break;
    }

    // printf("Getting the output finally after dot, add, && rms sgemv!!\n");
    result = resultbuf.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      printf("Failed to read result data in the end\n");
      break;
    }
  } while (false);
}

void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemv_ptr;

    if (TransA) {
      kernel_sgemv_ptr =
        cl_context_ref.registerClKernel(sgemv_cl_kernel_, "sgemv_cl");
    } else {
      kernel_sgemv_ptr = cl_context_ref.registerClKernel(
        sgemv_cl_noTrans_kernel_, "sgemv_cl_noTrans");
    }

    if (!kernel_sgemv_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;
    size_t dim2_size = sizeof(float) * dim2;
    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          dim1 * dim2 * sizeof(float), true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim2_size, true,
                          nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_, dim1_size, true,
                          nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemv_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
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
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sgemv_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

// void fused_dot_cl_rms(const float *vecAdata, const float *vecXdata, float
// *rdata, const float *gdata, float epsilon, unsigned int dim1){
//   bool result = false;

//   float cl_ret = 0;
//   do {
//     ClContext::SharedPtrClKernel kernel_dot_ptr =
//       cl_context_ref.registerClKernel(dot_cl_kernel_, "dot_cl");
//     if (!kernel_dot_ptr) {
//       break;
//     }

//     size_t dim1_size = sizeof(float) * dim1;

//     opencl::Buffer inputA(cl_context_ref.context_inst_, dim1_size, true,
//                           nullptr);

//     opencl::Buffer inputX(cl_context_ref.context_inst_, dim1_size, true,
//                           nullptr);

//     opencl::Buffer dotResult(cl_context_ref.context_inst_, sizeof(float),
//     true,
//                              &cl_ret);

//     result = inputA.WriteData(cl_context_ref.command_queue_inst_, vecAdata);
//     if (!result) {
//       break;
//     }

//     result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
//     if (!result) {
//       break;
//     }

//     result = kernel_dot_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
//     if (!result) {
//       break;
//     }

//     result = kernel_dot_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
//     if (!result) {
//       break;
//     }

//     result = kernel_dot_ptr->SetKernelArguments(2, &dim1, sizeof(int));
//     if (!result) {
//       break;
//     }

//     result = kernel_dot_ptr->SetKernelArguments(3, &dotResult,
//     sizeof(cl_mem)); if (!result) {
//       break;
//     }

//     const int work_groups_count[3] = {(int)dim1, 1, 1};
//     const int work_group_size[3] = {32, 32, 1}; // test-value

//     result = cl_context_ref.command_queue_inst_.DispatchCommand(
//       kernel_dot_ptr, work_groups_count, work_group_size);
//     if (!result) {
//       break;
//     }

//     *rdata += cl_ret;

//     result = dotResult.ReadData(cl_context_ref.command_queue_inst_, &cl_ret);
//     if (!result) {
//       break;
//     }

//   } while (false);
// }

float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1) {

  bool result = false;

  float cl_ret = 0;

  do {
    ClContext::SharedPtrClKernel kernel_dot_ptr =
      cl_context_ref.registerClKernel(dot_cl_kernel_, "dot_cl");
    if (!kernel_dot_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim1_size, true,
                          nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_, dim1_size, true,
                          nullptr);

    opencl::Buffer dotResult(cl_context_ref.context_inst_, sizeof(float), true,
                             &cl_ret);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, vecAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_dot_ptr->SetKernelArguments(3, &dotResult, sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_dot_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = dotResult.ReadData(cl_context_ref.command_queue_inst_, &cl_ret);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}

void printMatrix(float *matrix, unsigned int rows, unsigned int cols) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      printf("%f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
}

void fused_sgemm_cl_rms(bool TransA, bool TransB, const float *A,
                        const float *B, float *C, const float *gdata,
                        const float *bdata, bool isAdditionPossible,
                        float epsilon, bool isbias, unsigned int M,
                        unsigned int N, unsigned int K, unsigned int lda,
                        unsigned int ldb, unsigned int ldc,
                        unsigned int bias_dim1, int b, int c, int h, int w) {

  bool result = false;

  do {
    std::string kernel_func_;
    std::string sgemm_cl_kernel_;

    if (!TransA && !TransB) {
      kernel_func_ = "sgemm_cl_noTrans";
      sgemm_cl_kernel_ = sgemm_cl_noTrans_kernel_;
    } else if (TransA && !TransB) {
      kernel_func_ = "sgemm_cl_transA";
      sgemm_cl_kernel_ = sgemm_cl_transA_kernel_;
    } else if (!TransA && TransB) {
      kernel_func_ = "sgemm_cl_transB";
      sgemm_cl_kernel_ = sgemm_cl_transB_kernel_;
    } else {
      kernel_func_ = "sgemm_cl_transAB";
      sgemm_cl_kernel_ = sgemm_cl_transAB_kernel_;
    }

    ClContext::SharedPtrClKernel kernel_sgemm_ptr =
      cl_context_ref.registerClKernel(sgemm_cl_kernel_, kernel_func_);
    if (!kernel_sgemm_ptr) {
      printf("Failed to register sgemm kernel\n");
      break;
    }

    // sizes will be same for transpose
    size_t m_k_size = M * K * sizeof(float);
    size_t k_n_size = K * N * sizeof(float);
    size_t m_n_size = M * N * sizeof(float);
    unsigned int dim1 = M * N; // result size

    opencl::Buffer inputA(cl_context_ref.context_inst_, m_k_size, true,
                          nullptr);

    opencl::Buffer inputB(cl_context_ref.context_inst_, k_n_size, true,
                          nullptr);

    opencl::Buffer inOutC(cl_context_ref.context_inst_, m_n_size, true,
                          nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, A);
    if (!result) {
      printf("Failed to write inputA data ind sgemm\n");
      break;
    }

    result = inputB.WriteData(cl_context_ref.command_queue_inst_, B);
    if (!result) {
      printf("Failed to write inputB data in sgemm\n");
      break;
    }

    result = inOutC.WriteData(cl_context_ref.command_queue_inst_, C);
    if (!result) {
      printf("Failed to write inOutY data in sgemm\n");
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument inputA in sgemm\n");
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(1, &inputB, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument inputB in sgemm\n");
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(2, &inOutC, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument inOutY in sgemm\n");
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(3, &K, sizeof(int));
    if (!result) {
      printf("Failed to set argument K in sgemm\n");
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      printf("Failed to set argument lda in sgemm\n");
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(5, &ldb, sizeof(int));
    if (!result) {
      printf("Failed to set argument ldb in sgemm\n");
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(6, &ldc, sizeof(int));
    if (!result) {
      printf("Failed to set argument ldc in sgemm\n");
      break;
    }

    const int work_groups_count[3] = {(int)M, (int)N, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sgemm_ptr, work_groups_count, work_group_size);

    if (!result) {
      printf("Failed to dispatch sgemm kernel\n");
      break;
    }

    result = inOutC.ReadData(cl_context_ref.command_queue_inst_, C);
    if (!result) {
      printf("Failed to read result data after dptCL\n");
      break;
    }
    // printMatrix(C, M, N);
    // printf("Done with sgemm dotcl\n");

    if (isbias) {
      if (isAdditionPossible) {
        //  cl_event add_event;
        ClContext::SharedPtrClKernel kernel_addition_ptr =
          cl_context_ref.registerClKernel(addition_cl_kernel_, "addition_cl");

        if (!kernel_addition_ptr) {
          printf("Failed to register addition kernel\n");
          break;
        }

        size_t bias_size = sizeof(float) * bias_dim1;
        // size_t dim2_size = sizeof(float) * size_res; // result size -> dim1

        opencl::Buffer inputC(cl_context_ref.context_inst_, bias_size, true,
                              nullptr);

        result = inputC.WriteData(cl_context_ref.command_queue_inst_, bdata);
        if (!result) {
          printf("Failed to write inputC data in addition\n");
          break;
        }

        result =
          kernel_addition_ptr->SetKernelArguments(0, &inputC, sizeof(cl_mem));
        if (!result) {
          printf("Failed to set argument for inputC in addition\n");
          break;
        }

        result =
          kernel_addition_ptr->SetKernelArguments(1, &inOutC, sizeof(cl_mem));
        if (!result) {
          printf("Failed to set argument for inOutY in addition\n");
          break;
        }

        result =
          kernel_addition_ptr->SetKernelArguments(2, &bias_dim1, sizeof(int));
        if (!result) {
          printf("Failed to set argument for bias_dim1 in addition\n");
          break;
        }

        result = kernel_addition_ptr->SetKernelArguments(3, &dim1, sizeof(int));
        if (!result) {
          printf("Failed to set argument for dim1 in addition\n");
          break;
        }

        const int work_groups_count_add[3] = {(int)bias_dim1, 1, 1};
        const int work_group_size_add[3] = {32, 32, 1}; // test-value
        result = cl_context_ref.command_queue_inst_.DispatchCommand(
          kernel_addition_ptr, work_groups_count_add, work_group_size_add);

        if (!result) {
          printf("Failed to dispatch addition kernel\n");
          break;
        }
      } else {
        // throw std::invalid_argument(
        //   "Error: Broadcasting not supported for these dimensions!");
        printf("Broadcasting not supported for these dimensions!\n");
      }
    }

    //  cl_event rms_event;
    ClContext::SharedPtrClKernel kernel_rmsnorm_ptr =
      cl_context_ref.registerClKernel(rmsnorm_cl_kernel_new, "rmsnorm_cl");

    if (!kernel_rmsnorm_ptr) {
      printf("Failed to register rmsnorm kernel\n");
      break;
    }

    // for this the input is nothing but the result from above kernels, which is
    // result only
    opencl::Buffer gammabuf(cl_context_ref.context_inst_, w * sizeof(float),
                            true, nullptr);

    opencl::Buffer resultbuf(
      cl_context_ref.context_inst_, m_n_size, true,
      nullptr); // to store the data of the dot, add and rms

    result = gammabuf.WriteData(cl_context_ref.command_queue_inst_, gdata);
    if (!result) {
      printf("Failed to write gamma data in rmsnorm\n");
      break;
    }
    result = kernel_rmsnorm_ptr->SetKernelArguments(0, &inOutC, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for inputA in rmsnorm\n");
      break;
    }

    result =
      kernel_rmsnorm_ptr->SetKernelArguments(1, &resultbuf, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for inOutY in rmsnorm\n");
      break;
    }

    result =
      kernel_rmsnorm_ptr->SetKernelArguments(2, &gammabuf, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set argument for gammabuf in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(4, &b, sizeof(int));
    if (!result) {
      printf("Failed to set argument for b in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(3, &epsilon, sizeof(float));
    if (!result) {
      printf("Failed to set argument for epsilon in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(5, &c, sizeof(int));
    if (!result) {
      printf("Failed to set argument for c in rmsnorm\n");
      break;
    }

    result = kernel_rmsnorm_ptr->SetKernelArguments(6, &h, sizeof(int));
    if (!result) {
      printf("Failed to set argument for h in rmsnorm\n");
      break;
    }
    result = kernel_rmsnorm_ptr->SetKernelArguments(7, &w, sizeof(int));
    if (!result) {
      printf("Failed to set argument for w in rmsnorm\n");
      break;
    }
    const int work_groups_count_rms[3] = {b * c, h, 1};
    const int work_group_size_rms[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_rmsnorm_ptr, work_groups_count_rms, work_group_size_rms);
    if (!result) {
      printf("Failed to dispatch rmsnorm kernel\n");
      break;
    }

    result = resultbuf.ReadData(cl_context_ref.command_queue_inst_, C);
    if (!result) {
      printf("Failed to read result data in the end\n");
      break;
    }
    // printMatrix(C, M, N);
  } while (false);
}

void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc) {

  std::string kernel_func_;
  std::string sgemm_cl_kernel_;

  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans";
    sgemm_cl_kernel_ = sgemm_cl_noTrans_kernel_;
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA";
    sgemm_cl_kernel_ = sgemm_cl_transA_kernel_;
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB";
    sgemm_cl_kernel_ = sgemm_cl_transB_kernel_;
  } else {
    kernel_func_ = "sgemm_cl_transAB";
    sgemm_cl_kernel_ = sgemm_cl_transAB_kernel_;
  }

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_sgemm_ptr =
      cl_context_ref.registerClKernel(sgemm_cl_kernel_, kernel_func_);
    if (!kernel_sgemm_ptr) {
      break;
    }

    // sizes will be same for transpose
    size_t m_k_size = M * K * sizeof(float);
    size_t k_n_size = K * N * sizeof(float);
    size_t m_n_size = M * N * sizeof(float);

    opencl::Buffer inputA(cl_context_ref.context_inst_, m_k_size, true,
                          nullptr);

    opencl::Buffer inputB(cl_context_ref.context_inst_, k_n_size, true,
                          nullptr);

    opencl::Buffer inOutC(cl_context_ref.context_inst_, m_n_size, true,
                          nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, A);
    if (!result) {
      break;
    }

    result = inputB.WriteData(cl_context_ref.command_queue_inst_, B);
    if (!result) {
      break;
    }

    result = inOutC.WriteData(cl_context_ref.command_queue_inst_, C);
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(1, &inputB, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(2, &inOutC, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(3, &K, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(4, &lda, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(5, &ldb, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_sgemm_ptr->SetKernelArguments(6, &ldc, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)M, (int)N, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_sgemm_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutC.ReadData(cl_context_ref.command_queue_inst_, C);
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
      cl_context_ref.registerClKernel(addition_cl_kernel_, "addition_cl");
    if (!kernel_addition_ptr) {
      break;
    }

    size_t dim1_size = sizeof(float) * size_input;
    size_t dim2_size = sizeof(float) * size_res;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim1_size, true,
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim2_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, input);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result =
      kernel_addition_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_addition_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
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
    const int work_group_size[3] = {32, 32, 1}; // test-value
    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_addition_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void sscal_cl(float *X, const unsigned int N, const float alpha) {
  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_ptr =
      cl_context_ref.registerClKernel(sscal_cl_kernel_, "sscal_cl");

    if (!kernel_ptr) {
      break;
    }

    size_t x_size = N * sizeof(float);

    opencl::Buffer inputX(cl_context_ref.context_inst_, x_size, false, nullptr);

    result = inputX.WriteData(cl_context_ref.command_queue_inst_, X);
    if (!result) {
      break;
    }

    result = kernel_ptr->SetKernelArguments(0, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_ptr->SetKernelArguments(1, &alpha, sizeof(float));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)N, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inputX.ReadData(cl_context_ref.command_queue_inst_, X);
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
      kernel_transpose_ptr = cl_context_ref.registerClKernel(
        transpose_cl_kernel_axis0, "transpose_cl_axis0");
      break;
    case 1:
      kernel_transpose_ptr = cl_context_ref.registerClKernel(
        transpose_cl_kernel_axis1, "transpose_cl_axis1");
      break;
    case 2:
      kernel_transpose_ptr = cl_context_ref.registerClKernel(
        transpose_cl_kernel_axis2, "transpose_cl_axis2");
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

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim_size, true,
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, in);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_transpose_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
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

    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_transpose_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}
} // namespace nntrainer
