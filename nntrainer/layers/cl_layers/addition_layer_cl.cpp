// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   addition_layer_cl.cpp
 * @date   17 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yash Singh yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief	 This is Addition Layer Class Class for Neural Network with OpenCl
 * implementation
 */

#include <addition_layer_cl.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

std::string addition_cl_kernel_ =
  R"(__kernel void addition_cl(__global const float* input, __global float* output, const unsigned int size) {
    #pragma printf_support
    size_t idx = get_global_id(0);
    if (idx < size) {
        output[idx] = output[idx] + input[idx];
    }
})";

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayerCL::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayerCL::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      // hidden_.add_i(input_);
      AddProcess(input_, hidden_, context);
    }
  }
}

/**
 * @brief declaring static kerinputnel objects
 *
 */
opencl::Kernel AdditionLayerCL::kernel_addition;

void AdditionLayerCL::AddProcess(Tensor const &input, Tensor &result,
                                 RunLayerContext &context) {

  CREATE_IF_EMPTY_DIMS(result, result.getDim());

  NNTR_THROW_IF(result.getData() == nullptr, std::invalid_argument)
    << result.getName() << " is not allocated";
  NNTR_THROW_IF(input.getData() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";

  if (input.getDim() != result.getDim()) {
    throw std::invalid_argument(
      "Error: Dimensions does not match for addition");
  }

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    unsigned int size = input.size();
    const float *data = input.getData();
    float *rdata = result.getData();

    addition_cl(data, rdata, size, context);

  } else
    throw std::invalid_argument("Error: OpenCL fp16 is not supported yet.");
}

void AdditionLayerCL::addition_cl(const float *input, float *res,
                                  unsigned int size, RunLayerContext &context) {

  bool result = false;
  do {
    result = result =
      context.clCreateKernel(addition_cl_kernel_, context.LayerKernel::ADD,
                             AdditionLayerCL::kernel_addition);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * size;
    opencl::Buffer inputA(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inOutRes(context.context_inst_, dim1_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, input);
    if (!result) {
      break;
    }

    result = inOutRes.WriteData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

    result = AdditionLayerCL::kernel_addition.SetKernelArguments(
      0, &inputA, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = AdditionLayerCL::kernel_addition.SetKernelArguments(
      1, &inOutRes, sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = AdditionLayerCL::kernel_addition.SetKernelArguments(2, &size,
                                                                 sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)size, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value
    result = context.command_queue_inst_.DispatchCommand(
      AdditionLayerCL::kernel_addition, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutRes.ReadData(context.command_queue_inst_, res);
    if (!result) {
      break;
    }

  } while (false);
}

void AdditionLayerCL::incremental_forwarding(RunLayerContext &context,
                                             unsigned int from, unsigned int to,
                                             bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      if (!idx) {
        hidden_step.copy(input_step);
      } else {
        // hidden_step.add_i(input_step);
        AddProcess(input_step, hidden_step, context);
      }
    }
  }
}

void AdditionLayerCL::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayerCL::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
