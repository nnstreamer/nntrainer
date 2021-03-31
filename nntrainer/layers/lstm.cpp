// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	lstm.cpp
 * @date	17 March 2021
 * @brief	This is Long Short-Term Memory Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <cmath>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <lstm.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string LSTMLayer::type = "lstm";

enum LSTMParams { weight_xh, weight_hh, bias_h };

#define NUM_GATE 4

// - weight_xh ( input to hidden )
//  : [1, 1, input_sizex4, unit (hidden_size) ] -> f, g, i, o
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) x 4, unit (hidden_size)] -> f, g, i, o
// - bias_h ( hidden bias )
//  : [1, 1, 4, unit (hidden_size)] -> f, g, i, o
int LSTMLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (getNumInputs() != 1) {
    throw std::invalid_argument("LSTM layer takes only one input");
  }

  // input_dim = [ batch, 1, time_iterantion, feature_size ]
  // outut_dim = [ batch, 1, time_iteration, hidden_size ( unit ) ]
  output_dim[0] = input_dim[0];
  output_dim[0].width(unit);

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit);
  bias_dim.height(NUM_GATE);

  TensorDim dim_xh = output_dim[0];
  dim_xh.height(input_dim[0].width() * NUM_GATE);
  dim_xh.batch(1);

  TensorDim dim_hh = output_dim[0];
  dim_hh.height(unit * NUM_GATE);
  dim_hh.batch(1);

  if (weights.empty()) {
    weights.reserve(3);
    // weight_initializer can be set sepeartely. weight_xh initializer,
    // weight_hh initializer kernel initializer & recurrent_initializer in keras
    // for now, it is set same way.
    weights.emplace_back(dim_xh, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, "LSTM:weight_xh");
    weights.emplace_back(dim_hh, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, "LSTM:weight_hh");
    weights.emplace_back(bias_dim, bias_initializer, WeightRegularizer::NONE,
                         1.0f, true, "LSTM:bias_h");
    manager.trackWeights(weights);
  } else {
    weights[LSTMParams::weight_xh].reset(dim_xh, weight_initializer,
                                         weight_regularizer,
                                         weight_regularizer_constant, true);
    weights[LSTMParams::weight_hh].reset(dim_hh, weight_initializer,
                                         weight_regularizer,
                                         weight_regularizer_constant, true);
    weights[LSTMParams::bias_h].reset(bias_dim, bias_initializer,
                                      WeightRegularizer::NONE, 1.0f, true);
  }

  TensorDim cell_dim = TensorDim();
  cell_dim.setTensorDim(3, unit);
  cell_dim.batch(input_dim[0].batch());

  h_prev = Tensor(cell_dim);
  h_prev.setZero();

  c_prev = Tensor(cell_dim);
  c_prev.setZero();

  return status;
}

void LSTMLayer::setProperty(const PropertyType type, const std::string &value) {
  int status = ML_ERROR_NONE;
  // TODO : Add return_state property & api to get the hidden input
  switch (type) {
  case PropertyType::unit: {
    if (!value.empty()) {
      status = setUint(unit, value);
      throw_status(status);
      output_dim[0].width(unit);
    }
    break;
  case PropertyType::activation:
    if (!value.empty()) {
      ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
      Layer::activation_type = acti_type;
      acti_func.setActiFunc(acti_type);
    }
    break;
  case PropertyType::recurrent_activation:
    if (!value.empty()) {
      ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
      recurrent_activation_type = acti_type;
      recurrent_acti_func.setActiFunc(acti_type);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
  }
}

void LSTMLayer::setRecurrentActivation(ActivationType activation) {
  if (activation == ActivationType::ACT_UNKNOWN) {
    throw std::invalid_argument("Error: have to specify activation function");
  }
  recurrent_activation_type = activation;
}

void LSTMLayer::forwarding(bool training) {
  // NYI
}

void LSTMLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<LSTMLayer> from = std::static_pointer_cast<LSTMLayer>(l);
  this->unit = from->unit;
  this->acti_func = from->acti_func;
  this->recurrent_activation_type = from->recurrent_activation_type;
  this->recurrent_acti_func = from->recurrent_acti_func;
}

void LSTMLayer::calcDerivative() {
  // NYI
}

void LSTMLayer::calcGradient() {
  // NYI
}

} // namespace nntrainer
