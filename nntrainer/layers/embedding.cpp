// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.cpp
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <embedding.h>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

void EmbeddingLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("Embedding layer takes only one input");
  }

  const TensorDim &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  if (input_dim.channel() != 1) {
    throw std::invalid_argument(
      "Embedding layer takes only one for channel size");
  }

  TensorDim output_dim = input_dim;

  output_dim.height(in_length);
  output_dim.width(out_dim);
  context.setOutputDimensions({output_dim});

  TensorDim dim = output_dim;
  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  weight_idx =
    context.requestWeight(dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "Embedding", true);
}

void EmbeddingLayer::setProperty(const std::vector<std::string> &values) {
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

void EmbeddingLayer::setProperty(const std::string &type_str,
                                 const std::string &value) {
  using PropertyType = LayerV1::PropertyType;
  int status = ML_ERROR_NONE;
  LayerV1::PropertyType type =
    static_cast<LayerV1::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::in_dim: {
    status = setUint(in_dim, value);
    throw_status(status);
  } break;
  case PropertyType::out_dim: {
    status = setUint(out_dim, value);
    throw_status(status);
  } break;
  case PropertyType::in_length: {
    status = setUint(in_length, value);
    throw_status(status);
  } break;
  default:
    LayerImpl::setProperty(type_str, value);
    break;
  }
}

void EmbeddingLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    float *in_data = input_.getAddress(b * input_.getDim().getFeatureLen());

    for (unsigned int i = 0; i < in_length; ++i) {
      if (in_data[i] > in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      // Assume padding is 0 and index always start from 1.
      // If in_data[i] - 1 < 0, then it skips.
      if (in_data[i] - 1 < 0)
        continue;

      float *weight_data =
        weight.getAddress(static_cast<uint>(in_data[i] - 1) * out_dim);
      float *out_data =
        hidden_.getAddress(b * hidden_.getDim().getFeatureLen() + i * out_dim);

      std::copy(weight_data, weight_data + out_dim, out_data);
    }
  }
}

void EmbeddingLayer::calcDerivative(RunLayerContext &context) {
  // Uncomment this after fixing issues backwarding of first layer. (Issues
  // #1017)
  // throw exception::not_supported(
  //   "calcDerivative for Embedding layer is not supported");
  return; // intended
}

void EmbeddingLayer::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx);
  Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  djdw.setZero();

  // TODO:
  // This is to calculate gradient with current implementation of optimizer.
  // In order to accelerate, we need to better way like using index to weight.

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    float *in_data = input_.getAddress(b * input_.getDim().getFeatureLen());

    for (unsigned int i = 0; i < in_length; ++i) {
      // Assume padding is 0 and index always start from 1.
      // If in_data[i] - 1 < 0, then it skips.
      if (in_data[i] - 1 < 0)
        continue;

      float *djdw_data =
        djdw.getAddress(static_cast<uint>(in_data[i] - 1) * out_dim);
      float *grad_data = derivative_.getAddress(
        b * derivative_.getDim().getFeatureLen() + i * out_dim);

      std::transform(djdw_data, djdw_data + out_dim, grad_data, djdw_data,
                     std::plus<float>());
    }
  }
}

} // namespace nntrainer
