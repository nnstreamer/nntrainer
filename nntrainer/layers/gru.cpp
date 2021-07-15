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
 * h_prev --------d1------->[*]-------d0----->[+]---d0--> h
 * dh_nx    |  |             |                 | d0      dh
 *          | d14            | d2        d3    |
 *          |  |             +-----[1-]------>[*]
 *          | [*]<---+ d15   |d5               | d6
 *          |  |     |rt     | zt              |gt
 *          |  |    [sig]   [sig]            [tanh]
 *          |  |     |d16    | d7              |d8
 *          |  |    [+]      [+]              [+]
 *          |  |    / \d16   |  \ d7          / \ d8
 *          |  |  Wxhr Whhr Wxhz Whhz       Wxhg Whhg
 *          |  |  |d17  |d13 |d12 |d11       |d10 | d9
 *          +- |--+------|---+    |          |    |
 *             +---------|--------|----------+    |
 *   xs------------------+--------+---------------+
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

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum GRUParams { weight_xh, weight_hh, bias_h, hidden_state, zrg, h_prev };

#define NUM_GATE 3

// - weight_xh ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size) x NUM_GATE] -> f, g, i, o
void GRULayer::finalize(InitLayerContext &context) {
  auto unit = std::get<props::Unit>(props).get();

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("GRU layer takes only one input");
  }

  TensorDim output_dim;
  const TensorDim &input_dim = context.getInputDimensions()[0];

  // input_dim = [ batch, 1, time_iteration, feature_size ]
  // if return_sequences == False :
  //      output_dim = [ batch, 1, 1, hidden_size (unit)]
  // else:
  //      output_dim = [ batch, 1, time_iteration, hidden_size ( unit ) ]
  output_dim = input_dim;
  output_dim.width(unit);

  if (!return_sequences) {
    output_dim.height(1);
  }

  context.setOutputDimensions({output_dim});

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit * NUM_GATE);

  TensorDim dim_xh = output_dim;
  dim_xh.height(input_dim.width());
  dim_xh.width(unit * NUM_GATE);
  dim_xh.batch(1);

  TensorDim dim_hh = output_dim;
  dim_hh.height(unit);
  dim_hh.width(unit * NUM_GATE);
  dim_hh.batch(1);

  // weight_initializer can be set sepeartely. weight_xh initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.
  wt_idx[GRUParams::weight_xh] =
    context.requestWeight(dim_xh, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "GRU:weight_xh", true);
  wt_idx[GRUParams::weight_hh] =
    context.requestWeight(dim_hh, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "GRU:weight_hh", true);
  wt_idx[GRUParams::bias_h] =
    context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                          1.0f, "GRU:bias_h", true);

  TensorDim d = input_dim;
  d.width(unit);

  wt_idx[GRUParams::hidden_state] =
    context.requestTensor(d, "GRU:hidden_state", true, ITERATION_LIFESPAN);

  d.width(unit * NUM_GATE);
  wt_idx[GRUParams::zrg] =
    context.requestTensor(d, "GRU:zrg", true, ITERATION_LIFESPAN);

  TensorDim h_dim = TensorDim();
  h_dim.setTensorDim(3, unit);
  h_dim.batch(input_dim.batch());
  wt_idx[GRUParams::h_prev] =
    context.requestTensor(h_dim, "GRU:h_prev", false, FORWARD_FUNC_LIFESPAN);

  if (hidden_state_activation_type == ActivationType::ACT_NONE) {
    hidden_state_activation_type = ActivationType::ACT_TANH;
    acti_func.setActiFunc(hidden_state_activation_type);
  }

  if (recurrent_activation_type == ActivationType::ACT_NONE) {
    recurrent_activation_type = ActivationType::ACT_SIGMOID;
    recurrent_acti_func.setActiFunc(recurrent_activation_type);
  }
}

void GRULayer::setProperty(const std::vector<std::string> &values) {
  /// @todo: deprecate this in favor of loadProperties
  auto remain_props = loadProperties(values, props);
  for (unsigned int i = 0; i < remain_props.size(); ++i) {
    std::string key;
    std::string value;
    std::stringstream ss;

    if (getKeyValue(remain_props[i], key, value) != ML_ERROR_NONE) {
      throw std::invalid_argument("Error parsing the property: " +
                                  remain_props[i]);
    }

    if (value.empty()) {
      ss << "value is empty: key: " << key << ", value: " << value;
      throw std::invalid_argument(ss.str());
    }

    /// @note this calls derived setProperty if available
    setProperty(key, value);
  }
}

void GRULayer::setProperty(const std::string &type_str,
                           const std::string &value) {
  using PropertyType = LayerV1::PropertyType;
  int status = ML_ERROR_NONE;
  LayerV1::PropertyType type =
    static_cast<LayerV1::PropertyType>(parseLayerProperty(type_str));

  // TODO : Add return_state property & api to get the hidden input
  switch (type) {
  case PropertyType::hidden_state_activation: {
    ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
    hidden_state_activation_type = acti_type;
    acti_func.setActiFunc(acti_type);
  } break;
  case PropertyType::recurrent_activation: {
    ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
    recurrent_activation_type = acti_type;
    recurrent_acti_func.setActiFunc(acti_type);
  } break;
  case PropertyType::return_sequences: {
    status = setBoolean(return_sequences, value);
    throw_status(status);
  } break;
  default:
    LayerImpl::setProperty(type_str, value);
    break;
  }
}

void GRULayer::exportTo(Exporter &exporter, const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(props, method, this);
}

void GRULayer::forwarding(RunLayerContext &context, bool training) {
  auto unit = std::get<props::Unit>(props).get();
  Tensor &weight_xh = context.getWeight(wt_idx[GRUParams::weight_xh]);
  Tensor &weight_hh = context.getWeight(wt_idx[GRUParams::weight_hh]);
  Tensor &bias_h = context.getWeight(wt_idx[GRUParams::bias_h]);

  Tensor &hidden_ = context.getTensor(wt_idx[GRUParams::hidden_state]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &zrg = context.getTensor(wt_idx[GRUParams::zrg]);
  Tensor &h_prev = context.getTensor(wt_idx[GRUParams::h_prev]);
  const TensorDim &input_dim = input_.getDim();

  hidden_.setZero();
  zrg.setZero();
  h_prev.setZero();

  Tensor hs_prev;
  Tensor hs;

  // zt = sigma(W_hz.h_prev + W_xz.xs)
  // rt = sigma(W_hr.h_prev + W_xr.xs)
  // gt = tanh((h_prev*rt).W_hr + W_xg.xs)
  // h_nx = (1-zt)*gt + zt*h_prev

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);
    Tensor zrg_ = zrg.getBatchSlice(b, 1);

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

      xs.dot(weight_xh, zrg_t); // x_z, x_r, x_g

      Tensor ztrt = zrg_t.getSharedDataTensor({unit * 2}, 0);
      Tensor ztrt_b = bias_h.getSharedDataTensor({unit * 2}, 0);

      Tensor w_hh;
      w_hh.copy_with_stride(
        weight_hh.getSharedDataTensor({1, 1, unit, unit * 2}, 0, false));
      Tensor w_g;
      w_g.copy_with_stride(
        weight_hh.getSharedDataTensor({1, 1, unit, unit}, unit * 2, false));

      Tensor gt = zrg_t.getSharedDataTensor({unit}, unit * 2);
      Tensor gt_b = bias_h.getSharedDataTensor({unit}, unit * 2);

      ztrt.add_i(hs_prev.dot(w_hh));
      ztrt.add_i(ztrt_b);

      Tensor zt = ztrt.getSharedDataTensor({unit}, 0);
      Tensor rt = ztrt.getSharedDataTensor({unit}, unit);

      recurrent_acti_func.run_fn(rt, rt);
      recurrent_acti_func.run_fn(zt, zt);

      Tensor temp;
      rt.multiply(hs_prev, temp);
      gt.add_i(temp.dot(w_g));
      gt.add_i(gt_b);
      acti_func.run_fn(gt, gt);

      zt.multiply(hs_prev, hs);
      temp = zt.multiply(-1.0).add(1.0);
      hs.add_i(gt.multiply(temp));
    }
  }

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  if (!return_sequences) {
    TensorDim d = hidden_.getDim();
    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      Tensor dest = output.getSharedDataTensor({d.width()}, b * d.width());
      Tensor src = hidden_.getSharedDataTensor(
        {d.width()}, b * d.width() * d.height() + (d.height() - 1) * d.width());
      dest.copy(src);
    }
  } else {
    output.copy(hidden_);
  }
}

void GRULayer::calcDerivative(RunLayerContext &context) {
  Tensor &derivative_ = context.getTensorGrad(wt_idx[GRUParams::zrg]);
  Tensor &weight = context.getWeight(wt_idx[GRUParams::weight_xh]);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  derivative_.dot(weight, ret_, false, true);
}

void GRULayer::calcGradient(RunLayerContext &context) {
  auto unit = std::get<props::Unit>(props).get();
  Tensor &djdw_x = context.getWeightGrad(wt_idx[GRUParams::weight_xh]);
  Tensor &djdw_h = context.getWeightGrad(wt_idx[GRUParams::weight_hh]);
  Tensor &djdb_h = context.getWeightGrad(wt_idx[GRUParams::bias_h]);
  Tensor &weight_hh = context.getWeight(wt_idx[GRUParams::weight_hh]);

  Tensor djdw_zr_h = Tensor({1, 1, unit, unit * 2}, true);
  Tensor djdw_g_h = Tensor({1, 1, unit, unit}, true);
  Tensor &derivative_ = context.getTensorGrad(wt_idx[GRUParams::hidden_state]);
  Tensor &hidden_ = context.getTensor(wt_idx[GRUParams::hidden_state]);
  Tensor &incoming_deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &zrg = context.getTensor(wt_idx[GRUParams::zrg]);
  Tensor &d_zrg = context.getTensorGrad(wt_idx[GRUParams::zrg]);
  const TensorDim &input_dim = input_.getDim();

  djdw_x.setZero();
  djdw_zr_h.setZero();
  djdw_g_h.setZero();
  djdb_h.setZero();

  derivative_.setZero();
  d_zrg.setZero();

  if (!return_sequences) {
    TensorDim d = derivative_.getDim();
    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      Tensor dest = derivative_.getSharedDataTensor(
        {d.width()}, b * d.width() * d.height() + (d.height() - 1) * d.width());
      Tensor src =
        incoming_deriv.getSharedDataTensor({d.width()}, b * d.width());
      dest.copy(src);
    }
  } else {
    derivative_.copy(incoming_deriv);
  }

  Tensor dh_nx = Tensor({derivative_.width()});

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor deriv_t = derivative_.getBatchSlice(b, 1);
    Tensor xs_t = input_.getBatchSlice(b, 1);
    Tensor hs_t = hidden_.getBatchSlice(b, 1);

    dh_nx.setZero();

    Tensor dh;
    Tensor hs_prev;
    Tensor xs;
    Tensor dzrg_ = d_zrg.getBatchSlice(b, 1);
    Tensor zrg_ = zrg.getBatchSlice(b, 1);

    for (unsigned int t = deriv_t.height(); t-- > 0;) {
      dh = deriv_t.getSharedDataTensor({deriv_t.width()}, t * deriv_t.width());
      xs = xs_t.getSharedDataTensor({xs_t.width()}, t * xs_t.width());

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

      zt.multiply(dh, dh_nx); // dh_nx = d1

      dh.multiply(hs_prev, dhz);       // dhz = d2
      dhz.subtract_i(gt.multiply(dh)); // dhz = d5
      zt.multiply(-1.0, dhg);
      dhg.add_i(1.0);
      dhg.multiply_i(dh); // dhg = d6

      recurrent_acti_func.run_prime_fn(zt, dhz, dhz); // dhz = d7
      acti_func.run_prime_fn(gt, dhg, dhg);           // dhg = d8

      Tensor dhzr = dzrg_t.getSharedDataTensor({unit * 2}, 0); // dhz+dhr

      Tensor wg_hh;
      wg_hh.copy_with_stride(
        weight_hh.getSharedDataTensor({1, 1, unit, unit}, unit * 2, false));
      Tensor wzr_hh;
      wzr_hh.copy_with_stride(
        weight_hh.getSharedDataTensor({1, 1, unit, unit * 2}, 0, false));

      Tensor temp = Tensor({hs_t.width()});
      temp.setZero();
      dhg.dot(wg_hh, temp, false, true); // temp = d10
      hs_prev.multiply(temp, dhr);       // dhr = d15
      temp.multiply_i(rt);               // temp=d14
      dh_nx.add_i(temp);                 //  dh_nx = d1 + d14
      // reset temp : hs_prev * rt for djdw_g_h
      hs_prev.multiply(rt, temp);
      recurrent_acti_func.run_prime_fn(rt, dhr, dhr); // dhr = d16

      djdb_h.add_i(dzrg_t); // dzrg_t = d7+d16+d8

      djdw_x.add_i(xs.dot(dzrg_t, true, false));

      djdw_zr_h.add_i(hs_prev.dot(dhzr, true, false));
      djdw_g_h.add_i(temp.dot(dhg, true, false));
      dhzr.dot(wzr_hh, dh_nx, false, true, 1.0); // dh_nx = d1 + d14 + d12 + d17
    }
  }
  for (unsigned int h = 0; h < unit; ++h) {
    float *data = djdw_zr_h.getAddress(h * unit * 2);
    float *rdata = djdw_h.getAddress(h * unit * NUM_GATE);
    std::copy(data, data + unit * 2, rdata);
  }

  for (unsigned int h = 0; h < unit; ++h) {
    float *data = djdw_g_h.getAddress(h * unit);
    float *rdata = djdw_h.getAddress(h * unit * NUM_GATE + unit * 2);
    std::copy(data, data + unit, rdata);
  }
}

} // namespace nntrainer
