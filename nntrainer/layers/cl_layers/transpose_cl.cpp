// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   transpose_cl.cpp
 * @date   31 July 2024
 * @brief  Implementation of transpose layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "transpose_cl.h"
#include <iostream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

std::string transpose_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void transpose_cl_fp16(__global const half* in, 
                               __global half* output,
                               const int N, 
                               const int C, 
                               const int H, 
                               const int W) {

    int input_size = H * W;
    int output_size = H * W;

    // Calculate n, c, h, w from the global ids
    int h = get_global_id(0);
    int w = get_global_id(1);

    if (h < H && w < W) {
        for (int c = 0; c < C; ++c) {
            for (int n = 0; n < N; ++n) {
                // Calculate the input and output indices
                int input_index = n * (C * input_size) + c * input_size + h * W + w;
                int output_index = c * (N * output_size) + n * output_size + h * W + w;

                // Transpose batch and channel, copying data from input to output
                output[output_index] = in[input_index];
            }
        }
    }

})";

std::string transpose_cl_kernel_ =
  R"(__kernel void transpose_cl(__global const float* in, 
                               __global float* output,
                               const int N, 
                               const int C, 
                               const int H, 
                               const int W) {

    int input_size = H * W;
    int output_size = H * W;

    // Calculate n, c, h, w from the global ids
    int h = get_global_id(0);
    int w = get_global_id(1);

    if (h < H && w < W) {
        for (int c = 0; c < C; ++c) {
            for (int n = 0; n < N; ++n) {
                // Calculate the input and output indices
                int input_index = n * (C * input_size) + c * input_size + h * W + w;
                int output_index = c * (N * output_size) + n * output_size + h * W + w;

                // Transpose batch and channel, copying data from input to output
                output[output_index] = in[input_index];
            }
        }
    }

})";

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void TransposeLayerCl::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel());
      dim[i].height(dim[i].height());
      dim[i].width(dim[i].width());
    }
  }

  context.setOutputDimensions(dim);
}

void TransposeLayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  TransposeProcess(in, out, context);
}

void TransposeLayerCl::incremental_forwarding(RunLayerContext &context,
                                              unsigned int from,
                                              unsigned int to, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }
  TransposeProcess(in, out, context);
}

opencl::Kernel TransposeLayerCl::kernel_transpose;
opencl::Kernel TransposeLayerCl::kernel_transpose_fp16;

void TransposeLayerCl::TransposeProcess(Tensor const &in, Tensor &result,
                                        RunLayerContext &context) {

  unsigned int input_batch_size, input_height, input_width, input_channels;

  input_batch_size = in.batch();
  input_height = in.height();
  input_width = in.width();
  input_channels = in.channel();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = in.getData();
    float *rdata = result.getData();
    transpose_cl(data, rdata, input_batch_size, input_channels, input_height,
                 input_width, context);
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = in.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    transpose_cl_fp16(data, rdata, input_batch_size, input_channels,
                      input_height, input_width, context);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void TransposeLayerCl::transpose_cl(const float *in, float *res,
                                    unsigned int input_batch_size,
                                    unsigned int input_channels,
                                    unsigned int input_height,
                                    unsigned int input_width,
                                    RunLayerContext &context) {

  bool result = false;

  do {
    result = context.clCreateKernel(transpose_cl_kernel_,
                                    context.LayerKernel::TRANSPOSE,
                                    TransposeLayerCl::kernel_transpose);
    if (!result) {
      break;
    }

    size_t dim_size = sizeof(float) * input_batch_size * input_height *
                      input_width * input_channels;

    opencl::Buffer inputA(context.context_inst_, dim_size, true, nullptr);

    opencl::Buffer inOutRes(context.context_inst_, dim_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, in);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose.SetKernelArguments(
      0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose.SetKernelArguments(
      1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose.SetKernelArguments(
      2, &input_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose.SetKernelArguments(
      3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose.SetKernelArguments(
      4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose.SetKernelArguments(
      5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      TransposeLayerCl::kernel_transpose, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void TransposeLayerCl::transpose_cl_fp16(const __fp16 *in, __fp16 *res,
                                         unsigned int input_batch_size,
                                         unsigned int input_channels,
                                         unsigned int input_height,
                                         unsigned int input_width,
                                         RunLayerContext &context) {

  bool result = false;

  do {
    result = context.clCreateKernel(transpose_cl_kernel_fp16_,
                                    context.LayerKernel::TRANSPOSE_FP16,
                                    TransposeLayerCl::kernel_transpose_fp16);
    if (!result) {
      break;
    }

    size_t dim_size = sizeof(__fp16) * input_batch_size * input_height *
                      input_width * input_channels;

    opencl::Buffer inputA(context.context_inst_, dim_size, true, nullptr);

    opencl::Buffer inOutRes(context.context_inst_, dim_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, in);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose_fp16.SetKernelArguments(
      0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose_fp16.SetKernelArguments(
      1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose_fp16.SetKernelArguments(
      2, &input_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose_fp16.SetKernelArguments(
      3, &input_channels, sizeof(int));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose_fp16.SetKernelArguments(
      4, &input_height, sizeof(int));
    if (!result) {
      break;
    }

    result = TransposeLayerCl::kernel_transpose_fp16.SetKernelArguments(
      5, &input_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)input_height, (int)input_width, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      TransposeLayerCl::kernel_transpose_fp16, work_groups_count,
      work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void TransposeLayerCl::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

void TransposeLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, transpose_props);
  if (!remain_props.empty()) {
    std::string msg = "[TransposeLayerCl] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

#ifdef PLUGGABLE

Layer *create_transpose_layer_cl() {
  auto layer = new TransposeLayerCl();
  return layer;
}

void destroy_transpose_layer_cl(Layer *layer) { delete layer; }

extern "C" {
LayerPluggable ml_train_layer_pluggable{create_transpose_layer_cl,
                                        destroy_transpose_layer_cl};
}

#endif

} // namespace nntrainer
