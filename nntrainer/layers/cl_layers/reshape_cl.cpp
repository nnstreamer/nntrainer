// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   reshape_cl.cpp
 * @date   18 June 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Reshape GPU Layer Implementation
 */

#include <iostream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <reshape_cl.h>

std::string copy_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void copy_cl_fp16(__global const half* input, 
                               __global half* output,
                               const int batchsize, 
                               const int channels, 
                               const int height, 
                               const int width) {

    int i= get_global_id(0);
    output[i] = input[i];
    
})";

std::string copy_cl_kernel_ =
  R"(__kernel void copy_cl(__global const float* input, 
                               __global float* output,
                               const int batchsize, 
                               const int channels, 
                               const int height, 
                               const int width) {
    
    int i= get_global_id(0);
    output[i] = input[i];

})";

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapeLayerCl::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Reshape only supports 1 input for now";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  auto &target_shape = std::get<props::TargetShape>(reshape_props);
  NNTR_THROW_IF(target_shape.empty(), std::invalid_argument)
    << "Reshape layer must be provided with target shape";
  TensorDim out_dim = target_shape.get();

  if ((int)out_dim.getDataLen() == -1) {
    out_dim.height(1);
    out_dim.channel(1);
    out_dim.width(in_dim.getFeatureLen());
  } else if (out_dim.getFeatureLen() != in_dim.getFeatureLen()) {
    throw std::invalid_argument(
      "Target and input size mismatch for reshape layer");
  }

  out_dim.batch(in_dim.batch());

  context.setOutputDimensions({out_dim});
}

void ReshapeLayerCl::forwarding(RunLayerContext &context, bool training) {
  if (!context.getInPlace()) {
    Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
    const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
    ReshapeProcess(input, output);
  }
}

void ReshapeLayerCl::incremental_forwarding(RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  if (!context.getInPlace()) {
    Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
    const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
    if (from) {
      NNTR_THROW_IF(to - from != 1, std::invalid_argument)
        << "incremental step size is not 1";
      from = 0;
      to = 1;
    }
    ReshapeProcess(input, output);
  }
}

opencl::Kernel ReshapeLayerCl::kernel_copy;
opencl::Kernel ReshapeLayerCl::kernel_copy_fp16;

void ReshapeLayerCl::ReshapeProcess(Tensor const &input, Tensor &output) {

  unsigned int input_batch_size, input_height, input_width, input_channels;

  input_batch_size = input.batch();
  input_height = input.height();
  input_width = input.width();
  input_channels = input.channel();

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    float *rdata = output.getData();
    copy_cl(data, rdata, input_batch_size, input_channels, input_height,
            input_width);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = input.getData<_FP16>();
    _FP16 *rdata = output.getData<_FP16>();
    copy_cl_fp16(data, rdata, input_batch_size, input_channels, input_height,
                 input_width);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void ReshapeLayerCl::copy_cl_fp16(const __fp16 *input, __fp16 *res,
                                  unsigned int input_batch_size,
                                  unsigned int input_channels,
                                  unsigned int input_height,
                                  unsigned int input_width) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_copy_ptr =
      cl_context_ref.registerClKernel(copy_cl_kernel_fp16_, "copy_cl_fp16");
    if (!kernel_copy_ptr) {
      break;
    }

    size_t dim_size = sizeof(__fp16) * input_batch_size * input_height *
                      input_width * input_channels;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim_size, true,
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, input);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_copy_ptr->SetKernelArguments(2, &input_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_copy_ptr->SetKernelArguments(3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim_size, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_copy_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void ReshapeLayerCl::copy_cl(const float *input, float *res,
                             unsigned int input_batch_size,
                             unsigned int input_channels,
                             unsigned int input_height,
                             unsigned int input_width) {

  bool result = false;

  do {
    ClContext::SharedPtrClKernel kernel_copy_ptr =
      cl_context_ref.registerClKernel(copy_cl_kernel_, "copy_cl");
    if (!kernel_copy_ptr) {
      break;
    }

    size_t dim_size = sizeof(float) * input_batch_size * input_height *
                      input_width * input_channels;

    opencl::Buffer inputA(cl_context_ref.context_inst_, dim_size, true,
                          nullptr);

    opencl::Buffer inOutRes(cl_context_ref.context_inst_, dim_size, true,
                            nullptr);

    result = inputA.WriteData(cl_context_ref.command_queue_inst_, input);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_copy_ptr->SetKernelArguments(2, &input_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_copy_ptr->SetKernelArguments(3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result = kernel_copy_ptr->SetKernelArguments(5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim_size, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_copy_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(cl_context_ref.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void ReshapeLayerCl::calcDerivative(RunLayerContext &context) {
  if (!context.getInPlace()) {
    context.getOutgoingDerivative(SINGLE_INOUT_IDX)
      .copyData(context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void ReshapeLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, reshape_props);
  if (!remain_props.empty()) {
    std::string msg = "[ReshapeLayer] Unknown Layer Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

void ReshapeLayerCl::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  exporter.saveResult(reshape_props, method, this);
}

} /* namespace nntrainer */
