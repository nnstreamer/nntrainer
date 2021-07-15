// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.cpp
 * @date   27 Oct 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 */

#include <concat_layer.h>
#include <cstring>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ConcatLayer::finalize(InitLayerContext &context) {
  unsigned int channel = 0;

  auto const &input_dims = context.getInputDimensions();
  const TensorDim &input_dim_0 = input_dims[SINGLE_INOUT_IDX];
  channel += input_dim_0.channel();
  for (unsigned int idx = 1; idx < context.getNumInputs(); ++idx) {
    const TensorDim &dim = input_dims[idx];

    for (unsigned int i = 2; i < input_dim_0.rank(); ++i) {
      if (input_dim_0[i] != dim[i])
        throw std::runtime_error("Error: concat layer requires same "
                                 "shape from  all input layers");
    }
    channel += dim.channel();
  }

  TensorDim output_dim = input_dim_0;
  output_dim.channel(channel);

  context.setOutputDimensions({output_dim});
}

void ConcatLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  const TensorDim &output_dim = hidden_.getDim();

#ifdef DEBUG
  const TensorDim &input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  unsigned int channel = 0;
  channel += input_dim.channel();
  for (unsigned int idx = 1; idx < context.getNumInputs(); ++idx) {
    const TensorDim &dim = context.getInput(idx).getDim();

    for (unsigned int i = 2; i < input_dim.rank(); ++i) {
      if (input_dim[i] != dim[i])
        throw std::runtime_error("Error: concat layer requires same "
                                 "shape from  all input layers");
    }
    channel += dim.channel();
  }

  if (channel != output_dim.channel())
    throw std::runtime_error(
      "Error: Sum of channel of input layers is not same with output channel");
#endif

  unsigned int f_size = output_dim.getFeatureLen();

  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  for (unsigned int b = 0; b < output_dim.batch(); ++b) {
    unsigned int position = 0;
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      Tensor &input_ = context.getInput(idx);
      TensorDim in_dim = input_.getDim();
      memcpy(hidden_.getAddress(b * f_size + position),
             input_.getAddress(b * in_dim.getFeatureLen()),
             in_dim.getFeatureLen() * sizeof(float));
      position += in_dim.getFeatureLen();
    }
  }
}

void ConcatLayer::calcDerivative(RunLayerContext &context) {
  Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  TensorDim d = derivative_.getDim();

  unsigned int position = 0;
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    Tensor &ret_ = context.getOutgoingDerivative(idx);
    TensorDim in_dim = ret_.getDim();

    for (unsigned int b = 0; b < in_dim.batch(); ++b) {
      // TODO: replace with tensor::copy/fill
      memcpy(ret_.getAddress(b * in_dim.getFeatureLen()),
             derivative_.getAddress(b * d.getFeatureLen() + position),
             in_dim.getFeatureLen() * sizeof(float));
    }
    position += in_dim.getFeatureLen();
  }
}

void ConcatLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[ConcatLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
