// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   split_layer.cpp
 * @date   21 May 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Split Layer Class for Neural Network
 *
 */

#include <cstring>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <split_layer.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void SplitLayer::finalize(InitLayerContext &context) {
  if (split_dimension < 1) {
    throw std::invalid_argument(
      "Error: cannot split along the batch dimension");
  }

  if (split_dimension >= MAXDIM) {
    throw std::invalid_argument(
      "Error: split dimension exceeding the total number of dimensions");
  }

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument(
      "Error: only a single input is supported with split layer");
  }

  /**
   * The split is only done along the split_dimension dimension.
   * For example, consider input dimension [b,c,h,w],
   * 1. split_dimension = 0, output_dim = [1,c,h,w], num_outputs = b
   * 2. split_dimension = 1, output_dim = [b,1,h,w], num_outputs = c
   * 3. split_dimension = 2, output_dim = [b,c,1,w], num_outputs = h
   * 4. split_dimension = 3, output_dim = [b,c,h,1], num_outputs = w
   */
  const TensorDim &in_dim = context.getInputDimensions()[0];
  if (in_dim.getTensorDim(split_dimension) != context.getNumOutputs())
    throw std::invalid_argument(
      "Split dimension cannot be split into given number of outputs");

  TensorDim d = in_dim;
  d.setTensorDim(split_dimension, 1);

  std::vector<TensorDim> output_dim(context.getNumOutputs());
  for (auto &out_dim : output_dim) {
    out_dim = d;
  }
  context.setOutputDimensions(output_dim);

  /**
   * Setup input_reshape_helper to which input will be reshaped in forwarding
   * to facilitate easier processing.
   *
   * The helper shape consolidates all the dimensions before the split_dimension
   * together and all the dimensions after the split_dimension to faciliate
   * easier splitting of the data.
   */
  leading_helper_dim = 1;
  input_reshape_helper.channel(1);
  input_reshape_helper.height(1);
  input_reshape_helper.width(1);
  for (unsigned int idx = 1; idx < split_dimension; ++idx) {
    leading_helper_dim *= in_dim.getTensorDim(idx);
  }

  input_reshape_helper.height(in_dim.getTensorDim(split_dimension));

  for (unsigned int idx = split_dimension + 1; idx < MAXDIM; ++idx) {
    input_reshape_helper.width(input_reshape_helper.width() *
                               in_dim.getTensorDim(idx));
  }

  /**
   * Setup output_reshape_helper to which input will be reshaped in forwarding
   * to facilitate easier processing.
   */
  output_reshape_helper = input_reshape_helper;
  output_reshape_helper.height(1);

  setBatch(in_dim.batch());
}

void SplitLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  const TensorDim in_dim = input_.getDim();
  input_.reshape(input_reshape_helper);

  for (unsigned int idx = 0; idx < context.getNumOutputs(); idx++) {
    Tensor &output_ = context.getOutput(idx);
    const TensorDim out_dim = output_.getDim();
    output_.reshape(output_reshape_helper);

    for (unsigned int batch = 0; batch < input_.batch(); batch++) {
      const Tensor source_tensor = Tensor::Map(
        input_.getAddress(batch, 0, idx, 0), input_reshape_helper.width(),
        {1, 1, 1, input_reshape_helper.width()});
      Tensor dest_tensor = Tensor::Map(
        output_.getAddress(batch, 0, 0, 0), output_reshape_helper.width(),
        {1, 1, 1, output_reshape_helper.width()});
      dest_tensor.copy(source_tensor);
    }

    output_.reshape(out_dim);
  }

  input_.reshape(in_dim);
}

void SplitLayer::calcDerivative(RunLayerContext &context) {
  Tensor &input_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  const TensorDim in_dim = input_.getDim();
  input_.reshape(input_reshape_helper);

  for (unsigned int idx = 0; idx < context.getNumOutputs(); idx++) {
    Tensor &output_ = context.getIncomingDerivative(idx);
    const TensorDim out_dim = output_.getDim();
    output_.reshape(output_reshape_helper);

    for (unsigned int batch = 0; batch < input_.batch(); batch++) {
      Tensor dest_tensor = Tensor::Map(input_.getAddress(batch, 0, idx, 0),
                                       input_reshape_helper.width(),
                                       {1, 1, 1, input_reshape_helper.width()});
      const Tensor source_tensor = Tensor::Map(
        output_.getAddress(batch, 0, 0, 0), output_reshape_helper.width(),
        {1, 1, 1, output_reshape_helper.width()});
      dest_tensor.copy(source_tensor);
    }

    output_.reshape(out_dim);
  }

  input_.reshape(in_dim);
}

void SplitLayer::setProperty(const std::vector<std::string> &values) {
  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    std::stringstream ss;

    if (getKeyValue(values[i], key, value) != ML_ERROR_NONE) {
      throw std::invalid_argument("Error parsing the property: " + values[i]);
    }

    if (value.empty()) {
      ss << "value is empty: key: " << key << ", value: " << value;
      throw std::invalid_argument(ss.str());
    }

    /// @note this calls derived setProperty if available
    setProperty(key, value);
  }
}

void SplitLayer::setProperty(const std::string &type_str,
                             const std::string &value) {
  using PropertyType = nntrainer::Layer::PropertyType;
  int status = ML_ERROR_NONE;
  nntrainer::Layer::PropertyType type =
    static_cast<nntrainer::Layer::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::split_dimension: {
    status = setUint(split_dimension, value);
    NNTR_THROW_IF(split_dimension == 0, std::invalid_argument)
      << "[Split] Batch dimension cannot be split dimension";
    throw_status(status);
  } break;
  default:
    std::string msg =
      "[SplitLayer] Unknown Layer Property Key for value " + std::string(value);
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
