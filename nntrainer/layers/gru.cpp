// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gru.cpp
 * @date   17 March 2021
 * @brief  This is Gated Recurrent Unit Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <gru.h>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string GRULayer::type = "gru";

enum GRUParams { weight_xh, weight_hh, bias_h };

#define NUM_GATE 3

// - weight_xh ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size) x NUM_GATE] -> f, g, i, o
int GRULayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (getNumInputs() != 1) {
    throw std::invalid_argument("GRU layer takes only one input");
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
                         weight_regularizer_constant, true, "GRU:weight_xh");
    weights.emplace_back(dim_hh, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, "GRU:weight_hh");
    weights.emplace_back(bias_dim, bias_initializer, WeightRegularizer::NONE,
                         1.0f, true, "GRU:bias_h");
    manager.trackWeights(weights);
  } else {
    weights[GRUParams::weight_xh].reset(dim_xh, weight_initializer,
                                        weight_regularizer,
                                        weight_regularizer_constant, true);
    weights[GRUParams::weight_hh].reset(dim_hh, weight_initializer,
                                        weight_regularizer,
                                        weight_regularizer_constant, true);
    weights[GRUParams::bias_h].reset(bias_dim, bias_initializer,
                                     WeightRegularizer::NONE, 1.0f, true);
  }

  TensorDim d = input_dim[0];
  d.width(unit);

  hidden = std::make_shared<Var_Grad>(d, true, true, "GRU:output");

  d.width(unit * NUM_GATE);
  zrg = std::make_shared<Var_Grad>(d, true, true, "GRU:zrg");

  TensorDim h_dim = TensorDim();
  h_dim.setTensorDim(3, unit);
  h_dim.batch(input_dim[0].batch());

  h_prev = Tensor(h_dim);

  if (LayerV1::activation_type == ActivationType::ACT_NONE) {
    LayerV1::activation_type = ActivationType::ACT_TANH;
    acti_func.setActiFunc(activation_type);
  }

  if (recurrent_activation_type == ActivationType::ACT_NONE) {
    recurrent_activation_type = ActivationType::ACT_SIGMOID;
    recurrent_acti_func.setActiFunc(recurrent_activation_type);
  }

  return status;
}

void GRULayer::setProperty(const PropertyType type, const std::string &value) {
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
    LayerV1::setProperty(type, value);
    break;
  }
  }
}

void GRULayer::setRecurrentActivation(ActivationType activation) {
  if (activation == ActivationType::ACT_UNKNOWN) {
    throw std::invalid_argument("Error: have to specify activation function");
  }
  recurrent_activation_type = activation;
}

void GRULayer::forwarding(bool training) {
  Tensor &weight_xh =
    weightAt(static_cast<int>(GRUParams::weight_xh)).getVariableRef();
  Tensor &weight_hh =
    weightAt(static_cast<int>(GRUParams::weight_hh)).getVariableRef();
  Tensor &bias_h =
    weightAt(static_cast<int>(GRUParams::bias_h)).getVariableRef();

  hidden->getVariableRef().setZero();
  zrg->getVariableRef().setZero();

  h_prev.setZero();

  Tensor &hidden_ = hidden->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  Tensor hs_prev;
  Tensor hs;

  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);
    Tensor zrg_ = zrg->getVariableRef().getBatchSlice(b, 1);

    for (unsigned int t = 0; t < islice.height(); ++t) {
      Tensor xs =
        islice.getSharedDataTensor({islice.width()}, t * islice.width());
      hs = oslice.getSharedDataTensor({oslice.width()}, t * oslice.width());
      Tensor zrg_t =
        zrg_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      if (t > 0) {
        hs_prev = oslice.getSharedDataTensor({oslice.width()},
                                             (t - 1) * oslice.width());
      } else {
        hs_prev = h_prev.getBatchSlice(b, 1);
      }

      xs.dot(weight_xh, zrg_t);

      Tensor ztrt = zrg_t.getSharedDataTensor({unit * 2}, 0);
      Tensor ztrt_b = bias_h.getSharedDataTensor({unit * 2}, 0);

      Tensor w_hh = weight_hh.getSharedDataTensor({unit * unit * 2}, 0);
      Tensor w_g =
        weight_hh.getSharedDataTensor({unit * unit}, unit * unit * 2);
      Tensor gt = zrg_t.getSharedDataTensor({unit}, unit * 2);
      Tensor gt_b = bias_h.getSharedDataTensor({unit}, unit * 2);

      ztrt.add_i(hs_prev.dot(w_hh));
      ztrt.add_i(ztrt_b);

      Tensor zt = ztrt.getSharedDataTensor({unit}, 0);
      Tensor rt = ztrt.getSharedDataTensor({unit}, unit);

      recurrent_acti_func.run_fn(rt, rt);
      recurrent_acti_func.run_fn(zt, zt);

      gt.add_i(rt.multiply(hs_prev).dot(w_g));
      gt.add_i(gt_b);
      acti_func.run_fn(gt, gt);

      zt.multiply(hs_prev, hs);
      Tensor a = zt.multiply(-1.0).add(1.0);
      hs.add_i(gt.multiply(a));
    }
    h_prev.getBatchSlice(b, 1).copy(hs);
  }

  if (!return_sequences) {
    TensorDim d = hidden_.getDim();
    for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
      Tensor dest = net_hidden[0]->getVariableRef().getSharedDataTensor(
        {d.width()}, b * d.width());
      Tensor src = hidden_.getSharedDataTensor(
        {d.width()}, b * d.width() * d.height() + (d.height() - 1) * d.width());
      dest.copy(src);
    }
  } else {
    net_hidden[0]->getVariableRef().copy(hidden_);
  }
}

void GRULayer::copy(std::shared_ptr<LayerV1> l) {
  LayerV1::copy(l);

  std::shared_ptr<GRULayer> from = std::static_pointer_cast<GRULayer>(l);
  this->unit = from->unit;
  this->hidden_state_activation_type = from->hidden_state_activation_type;
  this->acti_func = from->acti_func;
  this->recurrent_activation_type = from->recurrent_activation_type;
  this->recurrent_acti_func = from->recurrent_acti_func;
  this->return_sequences = from->return_sequences;
}

void GRULayer::calcDerivative() {
  Tensor &derivative_ = zrg->getGradientRef();
  Tensor &weight =
    weightAt(static_cast<int>(GRUParams::weight_xh)).getVariableRef();
  Tensor &ret_ = net_input[0]->getGradientRef();
  derivative_.dot(weight, ret_, false, true);
}

void GRULayer::calcGradient() {
  Tensor &djdw_x =
    weightAt(static_cast<int>(GRUParams::weight_xh)).getGradientRef();
  Tensor &djdw_h =
    weightAt(static_cast<int>(GRUParams::weight_hh)).getGradientRef();
  Tensor &djdb_h =
    weightAt(static_cast<int>(GRUParams::bias_h)).getGradientRef();
  Tensor &weight_hh =
    weightAt(static_cast<int>(GRUParams::weight_hh)).getVariableRef();

  djdw_x.setZero();
  djdw_h.setZero();
  djdb_h.setZero();

  hidden->getGradientRef().setZero();
  zrg->getGradientRef().setZero();

  Tensor derivative_ = hidden->getGradientRef();

  if (!return_sequences) {
    TensorDim d = derivative_.getDim();
    for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
      Tensor dest = derivative_.getSharedDataTensor(
        {d.width()}, b * d.width() * d.height() + (d.height() - 1) * d.width());
      Tensor src = net_hidden[0]->getGradientRef().getSharedDataTensor(
        {d.width()}, b * d.width());
      dest.copy(src);
    }
  } else {
    derivative_.copy(net_hidden[0]->getGradientRef());
  }

  Tensor &hidden_ = hidden->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();
  Tensor dh_nx = Tensor({derivative_.width()});

  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    Tensor deriv_t = derivative_.getBatchSlice(b, 1);
    Tensor xs_t = input_.getBatchSlice(b, 1);
    Tensor hs_t = hidden_.getBatchSlice(b, 1);

    dh_nx.setZero();

    Tensor dh;
    Tensor hs_prev;
    Tensor hs;
    Tensor xs;
    Tensor dzrg_ = zrg->getGradientRef().getBatchSlice(b, 1);
    Tensor zrg_ = zrg->getVariableRef().getBatchSlice(b, 1);

    for (unsigned int t = deriv_t.height(); t-- > 0;) {
      dh = deriv_t.getSharedDataTensor({deriv_t.width()}, t * deriv_t.width());
      xs = xs_t.getSharedDataTensor({xs_t.width()}, t * xs_t.width());
      hs = hs_t.getSharedDataTensor({hs_t.width()}, t * hs_t.width());

      Tensor dzrg_t =
        dzrg_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);
      Tensor zrg_t =
        zrg_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      if (t == 0) {
        hs_prev = Tensor({hs_t.width()});
        hs_prev.setZero();
      } else {
        hs_prev =
          hs_t.getSharedDataTensor({hs_t.width()}, (t - 1) * hs_t.width());
      }
      if (t < deriv_t.height() - 1) {
        dh.add_i(dh_nx);
      }

      Tensor dhz = dzrg_t.getSharedDataTensor({unit}, 0);
      Tensor dhr = dzrg_t.getSharedDataTensor({unit}, unit);
      Tensor dhg = dzrg_t.getSharedDataTensor({unit}, unit * 2);

      Tensor zt = zrg_t.getSharedDataTensor({unit}, 0);
      Tensor rt = zrg_t.getSharedDataTensor({unit}, unit);
      Tensor gt = zrg_t.getSharedDataTensor({unit}, unit * 2);

      dh.multiply(hs_prev, dhz);
      dhz.subtract_i(gt.multiply(dh));
      zt.multiply(-1.0, dhg);
      dhg.add_i(1.0);
      dhg.multiply_i(dh);
      recurrent_acti_func.run_prime_fn(zt, dhz, dhz);
      acti_func.run_prime_fn(gt, dhg, dhg);

      Tensor dhzr = dzrg_t.getSharedDataTensor({unit * 2}, 0);
      Tensor djdw_zr_h = djdw_h.getSharedDataTensor({unit * unit * 2}, 0);
      Tensor djdw_g_h =
        djdw_h.getSharedDataTensor({unit * unit}, unit * unit * 2);

      Tensor wg_hh =
        weight_hh.getSharedDataTensor({unit * unit}, unit * unit * 2);
      Tensor wzr_hh = weight_hh.getSharedDataTensor({unit * unit * 2}, 0);

      dhg.multiply(wg_hh, dh_nx);
      hs_prev.multiply(dh_nx, dhr);
      dh_nx.multiply_i(rt);
      recurrent_acti_func.run_prime_fn(rt, dhr, dhr);

      djdb_h.add_i(dzrg_t);

      djdw_x.add_i(xs.dot(dzrg_t, true, false));
      djdw_zr_h.add_i(hs_prev.dot(dhzr, true, false));
      djdw_g_h.add_i(hs_prev.multiply(rt).dot(dhg, true, false));

      dhzr.dot(wzr_hh, dh_nx, false, true);
      dh_nx.add_i(zt.multiply(dh));
    }
  }
}

} // namespace nntrainer
