// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   swiglu_cl.cpp
 * @date   6th June 2024
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "swiglu_cl.h"
#include <iostream>

std::string swiglu_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void swiglu_cl_fp16(__global const half *in1, __global const half *in2, __global half *out) {
    int i = get_global_id(0);
    half swish = in1[i] * exp(in1[i]) / (1 + exp(in1[i]));
    out[i] = swish * in2[i];
})";

std::string swiglu_cl_kernel_ =
  R"(__kernel void swiglu_cl(__global const float *in1, __global const float *in2, __global float *out) {
    int i = get_global_id(0);
    float swish = in1[i] * exp(in1[i]) / (1 + exp(in1[i]));
    out[i] = swish * in2[i];
})";

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

opencl::Kernel SwiGLULayerCl::kernel_swiglu;
opencl::Kernel SwiGLULayerCl::kernel_swiglu_fp16;

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

void SwiGLULayerCl::swiglu_cl(const float *matAdata, const float *vecXdata,
                              float *vecYdata, unsigned int dim1,
                              unsigned int dim2) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_swiglu_ptr =
      cl_context_ref.registerClKernel(swiglu_cl_kernel_, "swiglu_cl");
    if (!kernel_swiglu_ptr) {
      break;
    }

    int dim = int(dim1 * dim2);
    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          sizeof(float) * dim1 * dim2, true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_,
                          sizeof(float) * dim1 * dim2, true, nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_,
                          sizeof(float) * dim1 * dim2, true, nullptr);

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

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_swiglu_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

void SwiGLULayerCl::swiglu_cl_fp16(const __fp16 *matAdata,
                                   const __fp16 *vecXdata, __fp16 *vecYdata,
                                   unsigned int dim1, unsigned int dim2) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_swiglu_ptr =
      cl_context_ref.registerClKernel(swiglu_cl_kernel_fp16_, "swiglu_cl_fp16");
    if (!kernel_swiglu_ptr) {
      break;
    }

    int dim = int(dim1 * dim2);
    opencl::Buffer inputA(cl_context_ref.context_inst_,
                          sizeof(__fp16) * dim1 * dim2, true, nullptr);

    opencl::Buffer inputX(cl_context_ref.context_inst_,
                          sizeof(__fp16) * dim1 * dim2, true, nullptr);

    opencl::Buffer inOutY(cl_context_ref.context_inst_,
                          sizeof(__fp16) * dim1 * dim2, true, nullptr);

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

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_swiglu_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(cl_context_ref.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

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

#ifdef PLUGGABLE

Layer *create_swiglu_layer_cl() {
  auto layer = new SwiGLULayerCl();
  return layer;
}

void destroy_swiglu_layer_cl(Layer *layer) { delete layer; }

extern "C" {
LayerPluggable ml_train_layer_pluggable{create_swiglu_layer_cl,
                                        destroy_swiglu_layer_cl};
}

#endif
} // namespace nntrainer
