// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   lstm.cpp
 * @date   17 March 2021
 * @brief  This is Long Short-Term Memory Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
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
//  : [1, 1, input_size, unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size) x NUM_GATE] -> f, g, i, o
int LSTMLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (getNumInputs() != 1) {
    throw std::invalid_argument("LSTM layer takes only one input");
  }

  // input_dim = [ batch, 1, time_iteration, feature_size ]
  // outut_dim = [ batch, 1, time_iteration, hidden_size ( unit ) ]
  output_dim[0] = input_dim[0];
  output_dim[0].width(unit);

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit * NUM_GATE);

  TensorDim dim_xh = output_dim[0];
  dim_xh.height(input_dim[0].width());
  dim_xh.width(unit * NUM_GATE);
  dim_xh.batch(1);

  TensorDim dim_hh = output_dim[0];
  dim_hh.height(unit);
  dim_hh.width(unit * NUM_GATE);
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

  mem_cell =
    std::make_shared<Var_Grad>(output_dim[0], true, true, "LSTM:mem_cell");
  mem_cell->getVariableRef().setZero();
  mem_cell->getGradientRef().setZero();

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
  Tensor &weight_xh =
    weightAt(static_cast<int>(LSTMParams::weight_xh)).getVariableRef();
  Tensor &weight_hh =
    weightAt(static_cast<int>(LSTMParams::weight_hh)).getVariableRef();
  Tensor &bias_h =
    weightAt(static_cast<int>(LSTMParams::bias_h)).getVariableRef();

  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();
  Tensor &m_cell_ = mem_cell->getVariableRef();

  Tensor temp;
  Tensor hs_prev;
  Tensor cs_prev;
  Tensor hs;
  Tensor cs;
  Tensor f;
  Tensor g;
  Tensor i;
  Tensor o;

  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);
    Tensor cell = m_cell_.getBatchSlice(b, 1);

    for (unsigned int t = 0; t < islice.height(); ++t) {
      Tensor xs =
        input_.getSharedDataTensor({islice.width()}, t * islice.width());
      hs = oslice.getSharedDataTensor({oslice.width()}, t * oslice.width());
      cs = cell.getSharedDataTensor({cell.width()}, t * cell.width());

      if (t > 0) {
        hs_prev = oslice.getSharedDataTensor({oslice.width()},
                                             (t - 1) * oslice.width());
        cs_prev =
          cell.getSharedDataTensor({cell.width()}, (t - 1) * cell.width());
      } else {
        hs_prev = h_prev.getBatchSlice(b, 1);
        cs_prev = c_prev.getBatchSlice(b, 1);
      }
      hs_prev.dot(weight_hh, temp);
      temp.add_i(bias_h);
      temp.add_i(xs.dot(weight_xh));

      f = temp.getSharedDataTensor({unit}, 0);
      g = temp.getSharedDataTensor({unit}, temp.width());
      i = temp.getSharedDataTensor({unit}, 2 * temp.width());
      o = temp.getSharedDataTensor({unit}, 3 * temp.width());

      recurrent_acti_func.run_fn(f, f);
      recurrent_acti_func.run_fn(i, i);
      recurrent_acti_func.run_fn(o, o);
      acti_func.run_fn(g, g);

      f.multiply(cs_prev, cs);
      cs.add_i(g.multiply_i(i));

      acti_func.run_fn(cs, hs);
      hs.multiply_i(o);
    }
    // size of h_prev and hs size is same : unit.
    // size of c_prev and cs is same : unit.
    h_prev.getBatchSlice(b, 1).copy(hs);
    c_prev.getBatchSlice(b, 1).copy(cs);
  }
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
