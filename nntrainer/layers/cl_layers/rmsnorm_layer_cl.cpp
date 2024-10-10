// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Thummala Pallavi <t.pallavi@samsung.com>
 *
 * @file        rmsnorm_layer_cl.cpp
 * @date        8 June 2024
 * @brief       This is RMSNorm Layer Class for Neural Network with
 * OpenCl implementation
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Thummala Pallavi <t.pallavi@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <rmsnorm_layer_cl.h>
#include <util_func.h>

std::string rmsnorm_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void rmsnorm_cl_fp16(
    __global const half *input,  // Input tensor
    __global half *output,    // Output tensor
    __global const half *alpha,  // Alpha values (one for each width)
    half epsilon,
    int B,                  // Number of batches
    int C,                  // Number of channels
    int H,                  // Height of feature map
    int W                   // Width of feature map
) {
    int global_id = get_global_id(0);  // Get the global work item index

    // Compute the corresponding batch, height, and channel indices
    int n = global_id / C;       // Batch index
    int c = global_id % C;                    // Height index
    int h = get_global_id(1);                    // Channel index
    int index = ((n * C + c) * H + h) * W;

    // Calculate RMS norm for the current channel, height, and batch
    half sum_squares = 0.0f;
    for (int j = 0; j < W; ++j) {
        sum_squares += input[index+j] * input[index+j];
    }
    sum_squares /= W;
    half rms_norm = sqrt(sum_squares + epsilon);
    // Each work item processes all width elements for its specific n, h, c
    for (int w = 0; w < W; ++w) {
        output[index+w] = (input[index+w] / rms_norm) * alpha[w];
    } 
}
)";

std::string rmsnorm_cl_kernel_ =
  R"(__kernel void rmsnorm_cl(
    __global const float *input,  // Input tensor
    __global float *output,    // Output tensor
    __global const float *alpha,  // Alpha values (one for each width)
    float epsilon,
    int B,                  // Number of batches
    int C,                  // Number of channels
    int H,                  // Height of feature map
    int W                   // Width of feature map
) {
    // Compute the corresponding batch, height, and channel indices
    int n = get_global_id(0) / C;
    int c = get_global_id(0) % C;
    int h = get_global_id(1);
    int index = ((n * C + c) * H + h) * W;
    // Calculate RMS norm for the current channel, height, and batch
    float sum_squares = 0.0f;
    for (int j = 0; j < W; ++j) {
        sum_squares += input[index+j] * input[index+j];
    }
    sum_squares /= W;
    float rms_norm = sqrt(sum_squares + epsilon);
    // Each work item processes all width elements for its specific n, h, c
    for (int w = 0; w < W; ++w) {
        output[index+w] = (input[index+w] / rms_norm) * alpha[w];
    }
}
)";

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum RMSParams { gamma };

RMSNormLayerCl::RMSNormLayerCl() : LayerImpl() { wt_idx.fill(0); }

void RMSNormLayerCl::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  auto &rmsparams_gamma =
    std::get<props::RMS_NORM_GAMMA_INIT_GPU>(rmsnorm_props);

  TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()));
  wt_idx[RMSParams::gamma] =
    context.requestWeight(gamma_dim, rmsparams_gamma, WeightRegularizer::NONE,
                          1.0f, 0.0f, "gamma", false);
}

void RMSNormLayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  auto &epsilon = std::get<props::Epsilon>(rmsnorm_props).get();
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    rmsnormProcess(in, out, gamma, epsilon);
  } else {
#ifdef ENABLE_FP16
    rmsnormProcess_fp16(in, out, gamma, epsilon);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

opencl::Kernel RMSNormLayerCl::kernel_rmsnorm;
opencl::Kernel RMSNormLayerCl::kernel_rmsnorm_fp16;

void RMSNormLayerCl::rmsnormProcess(Tensor const &input, Tensor &result,
                                    Tensor const &gamma, const float epsilon) {
  bool ret = false;
  int dim1 = input.batch() * input.height() * input.width() * input.channel();
  CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                       input.width(), input.getTensorType());
  int b = input.batch();
  int c = input.channel();
  int h = input.height();
  int w = input.width();

  do {
    ClContext::SharedPtrClKernel kernel_rmsnorm_ptr =
      cl_context_ref.registerClKernel(rmsnorm_cl_kernel_, "rmsnorm_cl");
    if (!kernel_rmsnorm_ptr) {
      break;
    }

    opencl::Buffer inputbuf(cl_context_ref.context_inst_, dim1 * sizeof(float),
                            true, nullptr);

    opencl::Buffer gammabuf(cl_context_ref.context_inst_,
                            input.width() * sizeof(float), true, nullptr);
    opencl::Buffer resultbuf(cl_context_ref.context_inst_, dim1 * sizeof(float),
                             true, nullptr);

    const float *data = input.getData();
    float *rdata = result.getData();
    const float *gdata = gamma.getData();
    ret = inputbuf.WriteData(cl_context_ref.command_queue_inst_, data);
    if (!ret) {
      break;
    }

    ret = gammabuf.WriteData(cl_context_ref.command_queue_inst_, gdata);
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(0, &inputbuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(1, &resultbuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(2, &gammabuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(4, &b, sizeof(int));

    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(3, &epsilon, sizeof(float));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(5, &c, sizeof(int));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(6, &h, sizeof(int));
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(7, &w, sizeof(int));
    if (!ret) {
      break;
    }
    const int work_groups_count[3] = {b * c, h, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    ret = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_rmsnorm_ptr, work_groups_count, work_group_size);
    if (!ret) {
      break;
    }

    ret = resultbuf.ReadData(cl_context_ref.command_queue_inst_, rdata);
    if (!ret) {
      break;
    }

  } while (false);
}

void RMSNormLayerCl::rmsnormProcess_fp16(Tensor const &input, Tensor &result,
                                         Tensor const &gamma,
                                         const float epsilon) {

  bool ret = false;
  int dim1 = input.batch() * input.height() * input.width() * input.channel();
  CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                       input.width(), input.getTensorType());
  int b = input.batch();
  int c = input.channel();
  int h = input.height();
  int w = input.width();
  do {
    ClContext::SharedPtrClKernel kernel_rmsnorm_ptr =
      cl_context_ref.registerClKernel(rmsnorm_cl_kernel_fp16_,
                                      "rmsnorm_cl_fp16");
    if (!kernel_rmsnorm_ptr) {
      break;
    }
    opencl::Buffer inputbuf(cl_context_ref.context_inst_,
                            dim1 * sizeof(cl_half), true, nullptr);

    opencl::Buffer gammabuf(cl_context_ref.context_inst_,
                            input.width() * sizeof(cl_half), true, nullptr);
    opencl::Buffer resultbuf(cl_context_ref.context_inst_,
                             dim1 * sizeof(cl_half), true, nullptr);

    const __fp16 *data = input.getData<__fp16>();
    __fp16 *rdata = result.getData<__fp16>();
    const __fp16 *gdata = gamma.getData<__fp16>();
    ret = inputbuf.WriteData(cl_context_ref.command_queue_inst_, data);
    if (!ret) {
      break;
    }

    ret = gammabuf.WriteData(cl_context_ref.command_queue_inst_, gdata);
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(0, &inputbuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(1, &resultbuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(2, &gammabuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(4, &b, sizeof(int));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(3, &epsilon, sizeof(cl_half));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(5, &c, sizeof(int));
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(6, &h, sizeof(int));
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(7, &w, sizeof(int));
    if (!ret) {
      break;
    }
    const int work_groups_count[3] = {b * c, h, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    ret = cl_context_ref.command_queue_inst_.DispatchCommand(
      kernel_rmsnorm_ptr, work_groups_count, work_group_size);
    if (!ret) {
      break;
    }

    ret = resultbuf.ReadData(cl_context_ref.command_queue_inst_, rdata);
    if (!ret) {
      break;
    }
  } while (false);
}

void RMSNormLayerCl::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  in_step_dim.height(to - from);
  out_step_dim.height(to - from);

  Tensor in_step = in.getSharedDataTensor(in_step_dim, 0, true);
  Tensor out_step = out.getSharedDataTensor(out_step_dim, 0, true);

  auto &epsilon = std::get<props::Epsilon>(rmsnorm_props).get();

  if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    rmsnormProcess(in, out, gamma, epsilon);
  } else {
    rmsnormProcess_fp16(in, out, gamma, epsilon);
  }
}

void RMSNormLayerCl::calcDerivative(RunLayerContext &context) {
  ml_logi("Training not supported");
}

void RMSNormLayerCl::calcGradient(RunLayerContext &context) {
  ml_logi("Training not supported");
}

void RMSNormLayerCl::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rmsnorm_props, method, this);
}

void RMSNormLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, rmsnorm_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace nntrainer
