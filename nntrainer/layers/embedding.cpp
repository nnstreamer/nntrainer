// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	embedding.cpp
 * @date	04 March 2021
 * @brief	This is Embedding Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
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

const std::string EmbeddingLayer::type = "embedding";

enum EmbeddingParams { weight };

int EmbeddingLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (num_inputs != 1) {
    throw std::invalid_argument("Embedding layer takes only one input");
  }

  if (input_dim[0].channel() != 1) {
    throw std::invalid_argument(
      "Embedding layer takes only one for channel size");
  }

  output_dim[0] = input_dim[0];

  output_dim[0].height(in_length);
  output_dim[0].width(out_dim);
  input_dim[0].width(in_length);

  TensorDim dim = output_dim[0];

  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  if (weights.empty()) {
    weights.reserve(1);
    weights.emplace_back(dim, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, "Embedding");
    manager.trackWeights(weights);
  } else {
    weights[EmbeddingParams::weight].reset(dim, weight_initializer,
                                           weight_regularizer,
                                           weight_regularizer_constant, true);
  }

  return status;
}

void EmbeddingLayer::setProperty(const PropertyType type,
                                 const std::string &value) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::in_dim: {
    if (!value.empty()) {
      status = setUint(in_dim, value);
      throw_status(status);
      input_dim[0].width(in_dim);
    }
  } break;
  case PropertyType::out_dim: {
    if (!value.empty()) {
      status = setUint(out_dim, value);
      throw_status(status);
      output_dim[0].width(out_dim);
    }
  } break;
  case PropertyType::in_length: {
    if (!value.empty()) {
      status = setUint(in_length, value);
      throw_status(status);
      output_dim[0].height(in_length);
      input_dim[0].height(in_length);
    }
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void EmbeddingLayer::forwarding(bool training) {
  Tensor &weight =
    weightAt(static_cast<int>(EmbeddingParams::weight)).getVariableRef();
  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    float *in_data = input_.getAddress(b * input_.getDim().getFeatureLen());

    for (unsigned int i = 0; i < in_length; ++i) {
      if (in_data[i] < 0)
        continue;

      float *weight_data =
        weight.getAddress(static_cast<uint>(in_data[i]) * out_dim);
      float *out_data =
        hidden_.getAddress(b * hidden_.getDim().getFeatureLen() + i * out_dim);

      for (unsigned int j = 0; j < out_dim; ++j) {
        out_data[j] = weight_data[j];
      }
    }
  }

  loss =
    weightAt(static_cast<int>(EmbeddingParams::weight)).getRegularizationLoss();
}

void EmbeddingLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<EmbeddingLayer> from =
    std::static_pointer_cast<EmbeddingLayer>(l);
  this->in_dim = from->in_dim;
  this->out_dim = from->out_dim;
  this->in_length = from->in_length;
}

void EmbeddingLayer::calcDerivative() {
  // NYI
}
void EmbeddingLayer::calcGradient() {
  // NYI
}

} // namespace nntrainer
