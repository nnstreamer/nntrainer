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
  // if return_sequences == False :
  //      output_dim = [ batch, 1, 1, hidden_size (unit)]
  // else:
  //      output_dim = [ batch, 1, time_iteration, hidden_size ( unit ) ]
  output_dim[0] = input_dim[0];
  output_dim[0].width(unit);

  if (!return_sequences) {
    output_dim[0].height(1);
  }

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

  TensorDim d = input_dim[0];
  d.width(unit);

  mem_cell = std::make_shared<Var_Grad>(d, true, true, "LSTM:mem_cell");
  hidden = std::make_shared<Var_Grad>(d, true, true, "LSTM:output");
  d.width(unit * NUM_GATE);
  fgio = std::make_shared<Var_Grad>(d, true, true, "LSTM:fgio");

  TensorDim cell_dim = TensorDim();
  cell_dim.setTensorDim(3, unit);
  cell_dim.batch(input_dim[0].batch());

  h_prev = Tensor(cell_dim);

  c_prev = Tensor(cell_dim);

  if (hidden_state_activation_type == ActivationType::ACT_NONE) {
    hidden_state_activation_type = ActivationType::ACT_TANH;
    acti_func.setActiFunc(hidden_state_activation_type);
  }

  if (recurrent_activation_type == ActivationType::ACT_NONE) {
    recurrent_activation_type = ActivationType::ACT_SIGMOID;
    recurrent_acti_func.setActiFunc(recurrent_activation_type);
  }

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
  case PropertyType::hidden_state_activation:
    if (!value.empty()) {
      ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
      hidden_state_activation_type = acti_type;
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
  case PropertyType::return_sequences:
    if (!value.empty()) {
      status = setBoolean(return_sequences, value);
      throw_status(status);
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

  mem_cell->getVariableRef().setZero();
  hidden->getVariableRef().setZero();
  fgio->getVariableRef().setZero();

  h_prev.setZero();
  c_prev.setZero();

  Tensor hidden_;
  hidden_ = hidden->getVariableRef();

  Tensor &input_ = net_input[0]->getVariableRef();
  Tensor &m_cell_ = mem_cell->getVariableRef();

  Tensor hs_prev;
  Tensor cs_prev;
  Tensor hs;
  Tensor cs;

  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);
    Tensor cell = m_cell_.getBatchSlice(b, 1);
    Tensor fgio_ = fgio->getVariableRef().getBatchSlice(b, 1);

    for (unsigned int t = 0; t < islice.height(); ++t) {
      Tensor xs =
        islice.getSharedDataTensor({islice.width()}, t * islice.width());
      hs = oslice.getSharedDataTensor({oslice.width()}, t * oslice.width());
      cs = cell.getSharedDataTensor({cell.width()}, t * cell.width());
      Tensor fgio_t =
        fgio_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      if (t > 0) {
        hs_prev = oslice.getSharedDataTensor({oslice.width()},
                                             (t - 1) * oslice.width());
        cs_prev =
          cell.getSharedDataTensor({cell.width()}, (t - 1) * cell.width());
      } else {
        hs_prev = h_prev.getBatchSlice(b, 1);
        cs_prev = c_prev.getBatchSlice(b, 1);
      }
      hs_prev.dot(weight_hh, fgio_t);
      fgio_t.add_i(xs.dot(weight_xh));
      fgio_t.add_i(bias_h);

      Tensor hi = fgio_t.getSharedDataTensor({unit}, 0);
      Tensor hf = fgio_t.getSharedDataTensor({unit}, unit);
      Tensor hg = fgio_t.getSharedDataTensor({unit}, unit * 2);
      Tensor ho = fgio_t.getSharedDataTensor({unit}, unit * 3);

      recurrent_acti_func.run_fn(hf, hf);
      recurrent_acti_func.run_fn(hi, hi);
      recurrent_acti_func.run_fn(ho, ho);
      acti_func.run_fn(hg, hg);

      hf.multiply(cs_prev, cs);
      cs.add_i(hg.multiply(hi));

      acti_func.run_fn(cs, hs);
      hs.multiply_i(ho);
    }
    // size of h_prev and hs size is same : unit.
    // size of c_prev and cs is same : unit.
    h_prev.getBatchSlice(b, 1).copy(hs);
    c_prev.getBatchSlice(b, 1).copy(cs);
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
    std::copy(hidden_.getData(), hidden_.getData() + hidden_.length(),
              net_hidden[0]->getVariableRef().getData());
  }
}

void LSTMLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<LSTMLayer> from = std::static_pointer_cast<LSTMLayer>(l);
  this->unit = from->unit;
  this->hidden_state_activation_type = from->hidden_state_activation_type;
  this->acti_func = from->acti_func;
  this->recurrent_activation_type = from->recurrent_activation_type;
  this->recurrent_acti_func = from->recurrent_acti_func;
}

void LSTMLayer::calcDerivative() {
  Tensor &derivative_ = fgio->getGradientRef();
  Tensor &weight =
    weightAt(static_cast<int>(LSTMParams::weight_xh)).getVariableRef();
  Tensor &ret_ = net_input[0]->getGradientRef();
  derivative_.dot(weight, ret_, false, true);
}

void LSTMLayer::calcGradient() {
  Tensor &djdw_x =
    weightAt(static_cast<int>(LSTMParams::weight_xh)).getGradientRef();
  Tensor &djdw_h =
    weightAt(static_cast<int>(LSTMParams::weight_hh)).getGradientRef();
  Tensor &djdb_h =
    weightAt(static_cast<int>(LSTMParams::bias_h)).getGradientRef();
  Tensor &weight_hh =
    weightAt(static_cast<int>(LSTMParams::weight_hh)).getVariableRef();

  djdw_x.setZero();
  djdw_h.setZero();
  djdb_h.setZero();

  mem_cell->getGradientRef().setZero();
  hidden->getGradientRef().setZero();
  fgio->getGradientRef().setZero();

  Tensor derivative_ = hidden->getGradientRef();
  Tensor hidden_;

  if (!return_sequences) {
    TensorDim d = derivative_.getDim();
    for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
      float *data = derivative_.getAddress(b * d.width() * d.height() +
                                           (d.height() - 1) * d.width());
      float *rdata = net_hidden[0]->getGradientRef().getAddress(b * d.width());
      std::copy(rdata, rdata + d.width(), data);
    }
  } else {
    std::copy(net_hidden[0]->getGradientRef().getData(),
              net_hidden[0]->getGradientRef().getData() +
                net_hidden[0]->getGradientRef().length(),
              derivative_.getData());
  }

  hidden_ = hidden->getVariableRef();

  Tensor &input_ = net_input[0]->getVariableRef();
  Tensor &m_cell_ = mem_cell->getVariableRef();
  Tensor &dm_cell_ = mem_cell->getGradientRef();

  Tensor dh_nx = Tensor(derivative_.width());
  Tensor dc_nx = Tensor(derivative_.width());

  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    Tensor deriv_t = derivative_.getBatchSlice(b, 1);
    Tensor derivc_t = dm_cell_.getBatchSlice(b, 1);
    Tensor xs_t = input_.getBatchSlice(b, 1);
    Tensor hs_t = hidden_.getBatchSlice(b, 1);
    Tensor cs_t = m_cell_.getBatchSlice(b, 1);

    dc_nx.setZero();
    dh_nx.setZero();

    Tensor dh;
    Tensor xs;
    Tensor hs_prev;
    Tensor cs_prev;
    Tensor hs;
    Tensor cs;
    Tensor dc;
    Tensor dfgio_ = fgio->getGradientRef().getBatchSlice(b, 1);
    Tensor fgio_ = fgio->getVariableRef().getBatchSlice(b, 1);

    for (unsigned int t = deriv_t.height(); t-- > 0;) {
      dh = deriv_t.getSharedDataTensor({deriv_t.width()}, t * deriv_t.width());
      dc =
        derivc_t.getSharedDataTensor({derivc_t.width()}, t * derivc_t.width());
      xs = xs_t.getSharedDataTensor({xs_t.width()}, t * xs_t.width());
      hs = hs_t.getSharedDataTensor({hs_t.width()}, t * hs_t.width());
      cs = cs_t.getSharedDataTensor({cs_t.width()}, t * cs_t.width());

      Tensor dfgio_t =
        dfgio_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);
      Tensor fgio_t =
        fgio_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      if (t == 0) {
        hs_prev = Tensor(hs_t.width());
        hs_prev.setZero();
        cs_prev = Tensor(cs_t.width());
        cs_prev.setZero();
      } else {
        hs_prev =
          hs_t.getSharedDataTensor({hs_t.width()}, (t - 1) * hs_t.width());
        cs_prev =
          cs_t.getSharedDataTensor({cs_t.width()}, (t - 1) * cs_t.width());
      }

      if (t < deriv_t.height() - 1) {
        dh.add_i(dh_nx);
      }

      Tensor dhi = dfgio_t.getSharedDataTensor({unit}, 0);
      Tensor dhf = dfgio_t.getSharedDataTensor({unit}, unit);
      Tensor dhg = dfgio_t.getSharedDataTensor({unit}, unit * 2);
      Tensor dho = dfgio_t.getSharedDataTensor({unit}, unit * 3);

      Tensor hi = fgio_t.getSharedDataTensor({unit}, 0);
      Tensor hf = fgio_t.getSharedDataTensor({unit}, unit);
      Tensor hg = fgio_t.getSharedDataTensor({unit}, unit * 2);
      Tensor ho = fgio_t.getSharedDataTensor({unit}, unit * 3);

      acti_func.run_fn(cs, dho);
      dho.multiply_i(dh);
      acti_func.run_fn(cs, cs);
      acti_func.run_prime_fn(cs, dc, dh);
      dc.multiply_i(ho);
      dc.add_i(dc_nx);

      dc.multiply(cs_prev, dhf);
      dc.multiply(hg, dhi);
      dc.multiply(hi, dhg);
      dc.multiply(hf, dc_nx);

      recurrent_acti_func.run_prime_fn(ho, dho, dho);
      recurrent_acti_func.run_prime_fn(hf, dhf, dhf);
      recurrent_acti_func.run_prime_fn(hi, dhi, dhi);
      acti_func.run_prime_fn(hg, dhg, dhg);

      djdb_h.add_i(dfgio_t);
      djdw_x.add_i(xs.dot(dfgio_t, true, false));
      djdw_h.add_i(hs_prev.dot(dfgio_t, true, false));
      dfgio_t.dot(weight_hh, dh_nx, false, true);
    }
  }
}

} // namespace nntrainer
