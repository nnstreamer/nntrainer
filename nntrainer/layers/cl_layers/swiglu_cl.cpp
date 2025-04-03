// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   swiglu_cl.cpp
 * @date   6th June 2024
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "swiglu_cl.h"
#include <blas_kernel_strings.h>
#include <iostream>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

void SwiGLULayerCl::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SwiGLULayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);
  swigluProcess(in1, in2, out);
}

void SwiGLULayerCl::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  swigluProcess(in1, in2, out);
}

void SwiGLULayerCl::swigluProcess(Tensor const &in1, Tensor const &in2,
                                  Tensor &result) {

  unsigned int dim1, dim2;
  dim1 = in1.batch() * in1.channel() * in1.height();
  dim2 = in1.width();

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data1 = in1.getData();
    const float *data2 = in2.getData();
    float *rdata = result.getData();
    swiglu_cl(data1, data2, rdata, dim1, dim2);
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data1 = in1.getData<_FP16>();
    const _FP16 *data2 = in2.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    swiglu_cl_fp16(data1, data2, rdata, dim1, dim2);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

bool SwiGLULayerCl::registerClKernels() {
  // check if the kernels are already registered.
  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for swiglu_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_swiglu_ptr = nullptr;

    kernel_swiglu_ptr =
      global_cl_context->registerClKernel(getSwiGluClKernel(), "swiglu_cl");

    if (!kernel_swiglu_ptr) {
      ml_loge("OpenCL Error: Fail to register swiglu_cl kernel");
      break;
    }

#ifdef ENABLE_FP16
    layer_kernel_ptrs.emplace_back(kernel_swiglu_ptr);
    kernel_swiglu_ptr = global_cl_context->registerClKernel(
      getSwiGluClKernelFP16(), "swiglu_cl_fp16");

    if (!kernel_swiglu_ptr) {
      ml_loge("OpenCL Error: Fail to register swiglu_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_swiglu_ptr);
#endif

    return true;
  } while (false);

  // clear all registered kernels if any error occurs during registration
  layer_kernel_ptrs.clear();

  return false;
}

void SwiGLULayerCl::swiglu_cl(const float *matAdata, const float *vecXdata,
                              float *vecYdata, unsigned int dim1,
                              unsigned int dim2) {

  bool result = false;

  do {

    auto kernel_swiglu_ptr = layer_kernel_ptrs[Kernels::SWIGLU_CL];

    int dim = int(dim1 * dim2);
    opencl::Buffer inputA(global_cl_context->context_inst_,
                          sizeof(float) * dim1 * dim2, true, nullptr);

    opencl::Buffer inputX(global_cl_context->context_inst_,
                          sizeof(float) * dim1 * dim2, true, nullptr);

    opencl::Buffer inOutY(global_cl_context->context_inst_,
                          sizeof(float) * dim1 * dim2, true, nullptr);

    result = inputA.WriteData(global_cl_context->command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(global_cl_context->command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(global_cl_context->command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_swiglu_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_swiglu_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_swiglu_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_swiglu_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(global_cl_context->command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

#ifdef ENABLE_FP16
void SwiGLULayerCl::swiglu_cl_fp16(const _FP16 *matAdata, const _FP16 *vecXdata,
                                   _FP16 *vecYdata, unsigned int dim1,
                                   unsigned int dim2) {

  bool result = false;

  do {

    auto kernel_swiglu_ptr = layer_kernel_ptrs[Kernels::SWIGLU_CL_FP16];

    int dim = int(dim1 * dim2);
    opencl::Buffer inputA(global_cl_context->context_inst_,
                          sizeof(_FP16) * dim1 * dim2, true, nullptr);

    opencl::Buffer inputX(global_cl_context->context_inst_,
                          sizeof(_FP16) * dim1 * dim2, true, nullptr);

    opencl::Buffer inOutY(global_cl_context->context_inst_,
                          sizeof(_FP16) * dim1 * dim2, true, nullptr);

    result = inputA.WriteData(global_cl_context->command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(global_cl_context->command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(global_cl_context->command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = kernel_swiglu_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_swiglu_ptr->SetKernelArguments(1, &inputX, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_swiglu_ptr->SetKernelArguments(2, &inOutY, sizeof(cl_mem));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_swiglu_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(global_cl_context->command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}
#endif

void SwiGLULayerCl::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

void SwiGLULayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, swiglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[SwigluLayerCl] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} // namespace nntrainer
