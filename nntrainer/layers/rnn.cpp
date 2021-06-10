// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rnn.cpp
 * @date   17 March 2021
 * @brief  This is Recurrent Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <rnn.h>
#include <util_func.h>

namespace nntrainer {

const std::string RNNLayer::type = "rnn";

// - weight_xh ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) ]
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size)]
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size)]
enum RNNParams { weight_xh, weight_hh, bias_h };

int RNNLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (getNumInputs() != 1) {
    throw std::invalid_argument("RNN layer takes only one input");
  }

  // input_dim = [ batch, 1, time_iterantion, feature_size ]
  // outut_dim = [ batch, 1, time_iteration, hidden_size ( unit ) ]
  output_dim[0] = input_dim[0];
  output_dim[0].width(unit);

  if (!return_sequences) {
    output_dim[0].height(1);
  }

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit);

  TensorDim dim_xh = output_dim[0];
  dim_xh.height(input_dim[0].width());
  dim_xh.batch(1);

  TensorDim dim_hh = output_dim[0];
  dim_hh.height(unit);
  dim_hh.batch(1);

  if (weights.empty()) {
    weights.reserve(3);
    // weight_initializer can be set sepeartely. weight_xh initializer,
    // weight_hh initializer kernel initializer & recurrent_initializer in keras
    // for now, it is set same way.
    weights.emplace_back(dim_xh, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, "RNN:weight_xh");
    weights.emplace_back(dim_hh, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, "RNN:weight_hh");
    weights.emplace_back(bias_dim, bias_initializer, WeightRegularizer::NONE,
                         1.0f, true, "RNN:bias_h");
    manager.trackWeights(weights);
  } else {
    weights[RNNParams::weight_xh].reset(dim_xh, weight_initializer,
                                        weight_regularizer,
                                        weight_regularizer_constant, true);
    weights[RNNParams::weight_hh].reset(dim_hh, weight_initializer,
                                        weight_regularizer,
                                        weight_regularizer_constant, true);
    weights[RNNParams::bias_h].reset(bias_dim, bias_initializer,
                                     WeightRegularizer::NONE, 1.0f, true);
  }

  bias_dim.batch(input_dim[0].batch());
  h_prev = Tensor(bias_dim);

  TensorDim d = input_dim[0];
  d.width(unit);

  // We do not need this if we reuse net_hidden[0]. But if we do, then the unit
  // test will fail. Becuase it modifies the date during gradient calculation
  // TODO : We could control with something like #define test to save memory
  hidden = std::make_shared<Var_Grad>(d, true, true, "RNN:temp_hidden");

  if (hidden_state_activation_type == ActivationType::ACT_NONE) {
    hidden_state_activation_type = ActivationType::ACT_TANH;
    acti_func.setActiFunc(hidden_state_activation_type);
  }

  return status;
}

void RNNLayer::setProperty(const PropertyType type, const std::string &value) {
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
  case PropertyType::hidden_state_activation:
    if (!value.empty()) {
      ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
      hidden_state_activation_type = acti_type;
      acti_func.setActiFunc(acti_type);
    }
    break;
  case PropertyType::return_sequences:
    if (!value.empty()) {
      status = setBoolean(return_sequences, value);
      throw_status(status);
    }
    break;
  default:
    LayerV1::setProperty(type, value);
    break;
  }
  }
}

void RNNLayer::forwarding(bool training) {
  Tensor &weight_xh =
    weightAt(static_cast<int>(RNNParams::weight_xh)).getVariableRef();
  Tensor &weight_hh =
    weightAt(static_cast<int>(RNNParams::weight_hh)).getVariableRef();
  Tensor &bias_h =
    weightAt(static_cast<int>(RNNParams::bias_h)).getVariableRef();

  hidden->getVariableRef().setZero();

  if (training) {
    hidden->getGradientRef().setZero();
  }
  h_prev.setZero();

  Tensor &hidden_ = hidden->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  Tensor temp;
  Tensor hs_prev;
  Tensor hs;

  // TODO : check merge b and t index
  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);

    for (unsigned int t = 0; t < islice.height(); ++t) {
      Tensor xs =
        islice.getSharedDataTensor({islice.width()}, t * islice.width());

      hs = oslice.getSharedDataTensor({oslice.width()}, t * oslice.width());
      if (t > 0) {
        hs_prev = oslice.getSharedDataTensor({oslice.width()},
                                             (t - 1) * oslice.width());
      } else {
        hs_prev = h_prev.getBatchSlice(b, 1);
      }

      hs_prev.dot(weight_hh, temp);

      xs.dot(weight_xh, hs);
      temp.add_i(bias_h);

      hs.add_i(temp);
      // TODO : In-place calculation for activation
      acti_func.run_fn(hs, hs);
    }
    if (!training)
      h_prev.getBatchSlice(b, 1).copy(hs);
  }

  if (!return_sequences) {
    TensorDim d = hidden_.getDim();
    for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
      float *data = hidden_.getAddress(b * d.width() * d.height() +
                                       (d.height() - 1) * d.width());
      float *rdata = net_hidden[0]->getVariableRef().getAddress(b * d.width());
      std::copy(data, data + d.width(), rdata);
    }
  } else {
    net_hidden[0]->getVariableRef().copy(hidden_);
  }
}

void RNNLayer::copy(std::shared_ptr<LayerV1> l) {
  LayerV1::copy(l);

  std::shared_ptr<RNNLayer> from = std::static_pointer_cast<RNNLayer>(l);
  this->unit = from->unit;
  this->hidden_state_activation_type = from->hidden_state_activation_type;
  this->return_sequences = from->return_sequences;
  this->acti_func = from->acti_func;
}

void RNNLayer::calcDerivative() {
  Tensor &derivative_ = hidden->getGradientRef();
  Tensor &weight =
    weightAt(static_cast<int>(RNNParams::weight_xh)).getVariableRef();
  Tensor &ret_ = net_input[0]->getGradientRef();

  derivative_.dot(weight, ret_, false, true);
}

void RNNLayer::calcGradient() {
  Tensor &djdw_x =
    weightAt(static_cast<int>(RNNParams::weight_xh)).getGradientRef();
  Tensor &djdw_h =
    weightAt(static_cast<int>(RNNParams::weight_hh)).getGradientRef();
  Tensor &djdb_h =
    weightAt(static_cast<int>(RNNParams::bias_h)).getGradientRef();
  Tensor &weight_hh =
    weightAt(static_cast<int>(RNNParams::weight_hh)).getVariableRef();

  djdw_x.setZero();
  djdw_h.setZero();
  djdb_h.setZero();

  Tensor &derivative_ = hidden->getGradientRef();

  if (!return_sequences) {
    TensorDim d = derivative_.getDim();
    for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
      float *data = derivative_.getAddress(b * d.width() * d.height() +
                                           (d.height() - 1) * d.width());
      float *rdata = net_hidden[0]->getGradientRef().getAddress(b * d.width());
      std::copy(rdata, rdata + d.width(), data);
    }
  } else {
    derivative_.copy(net_hidden[0]->getGradientRef());
  }

  Tensor &hidden_ = hidden->getVariableRef();

  Tensor &input_ = net_input[0]->getVariableRef();
  Tensor dh_nx = Tensor(TensorDim(1, 1, 1, derivative_.width()));

  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    Tensor deriv_t = derivative_.getBatchSlice(b, 1);
    Tensor xs_t = input_.getBatchSlice(b, 1);
    Tensor hs_t = hidden_.getBatchSlice(b, 1);
    dh_nx.setZero();

    Tensor dh;
    Tensor xs;
    Tensor hs_prev;
    Tensor hs;

    for (unsigned int t = deriv_t.height(); t-- > 0;) {
      dh = deriv_t.getSharedDataTensor(TensorDim(1, 1, 1, deriv_t.width()),
                                       t * deriv_t.width());
      xs = xs_t.getSharedDataTensor(TensorDim(1, 1, 1, xs_t.width()),
                                    t * xs_t.width());
      hs = hs_t.getSharedDataTensor(TensorDim(1, 1, 1, hs_t.width()),
                                    t * hs_t.width());
      if (t == 0) {
        hs_prev = Tensor(TensorDim(1, 1, 1, hs_t.width()));
        hs_prev.setZero();
      } else {
        hs_prev = hs_t.getSharedDataTensor(TensorDim(1, 1, 1, hs_t.width()),
                                           (t - 1) * hs_t.width());
      }

      if (t < deriv_t.height() - 1) {
        dh.add_i(dh_nx);
      }

      acti_func.run_prime_fn(hs, dh, dh);

      djdb_h.add_i(dh);
      xs.dot(dh, djdw_x, true, false, 1.0);
      hs_prev.dot(dh, djdw_h, true, false, 1.0);
      dh.dot(weight_hh, dh_nx, false, true);
    }
  }
}

} // namespace nntrainer
