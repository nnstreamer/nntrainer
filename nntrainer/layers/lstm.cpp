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
#include <lazy_tensor.h>
#include <lstm.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LSTMParams {
  weight_xh,
  weight_hh,
  bias_h,
  hidden_state,
  mem_cell,
  fgio,
  dropout_mask
};

#define NUM_GATE 4

// - weight_xh ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size) x NUM_GATE] -> f, g, i, o
void LSTMLayer::finalize(InitLayerContext &context) {
  auto unit = std::get<props::Unit>(props).get();

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("LSTM layer takes only one input");
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

  if (dropout_rate > epsilon) {
    wt_idx[LSTMParams::dropout_mask] = context.requestTensor(
      output_dim, context.getName() + ":dropout_mask",
      Tensor::Initializer::NONE, false, ITERATION_LIFESPAN);
  }

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
  wt_idx[LSTMParams::weight_xh] = context.requestWeight(
    dim_xh, weight_initializer, weight_regularizer, weight_regularizer_constant,
    context.getName() + ":weight_xh", true);
  wt_idx[LSTMParams::weight_hh] = context.requestWeight(
    dim_hh, weight_initializer, weight_regularizer, weight_regularizer_constant,
    context.getName() + ":weight_hh", true);
  wt_idx[LSTMParams::bias_h] =
    context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                          1.0f, context.getName() + ":bias_h", true);

  TensorDim d = input_dim;
  d.width(unit);

  wt_idx[LSTMParams::hidden_state] =
    context.requestTensor(d, context.getName() + ":hidden_state",
                          Tensor::Initializer::NONE, true, ITERATION_LIFESPAN);
  wt_idx[LSTMParams::mem_cell] =
    context.requestTensor(d, context.getName() + ":mem_cell",
                          Tensor::Initializer::NONE, true, ITERATION_LIFESPAN);

  d.width(unit * NUM_GATE);
  wt_idx[LSTMParams::fgio] =
    context.requestTensor(d, context.getName() + ":fgio",
                          Tensor::Initializer::NONE, true, ITERATION_LIFESPAN);

  if (hidden_state_activation_type == ActivationType::ACT_NONE) {
    hidden_state_activation_type = ActivationType::ACT_TANH;
    acti_func.setActiFunc(hidden_state_activation_type);
  }

  if (recurrent_activation_type == ActivationType::ACT_NONE) {
    recurrent_activation_type = ActivationType::ACT_SIGMOID;
    recurrent_acti_func.setActiFunc(recurrent_activation_type);
  }
}

void LSTMLayer::setProperty(const std::vector<std::string> &values) {
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

void LSTMLayer::setProperty(const std::string &type_str,
                            const std::string &value) {
  using PropertyType = nntrainer::Layer::PropertyType;
  int status = ML_ERROR_NONE;
  nntrainer::Layer::PropertyType type =
    static_cast<nntrainer::Layer::PropertyType>(parseLayerProperty(type_str));

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
  case PropertyType::dropout:
    status = setFloat(dropout_rate, value);
    throw_status(status);
    break;
  default:
    LayerImpl::setProperty(type_str, value);
    break;
  }
}

void LSTMLayer::exportTo(Exporter &exporter,
                         const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(props, method, this);
}

void LSTMLayer::forwarding(RunLayerContext &context, bool training) {
  auto unit = std::get<props::Unit>(props).get();
  Tensor &weight_xh = context.getWeight(wt_idx[LSTMParams::weight_xh]);
  Tensor &weight_hh = context.getWeight(wt_idx[LSTMParams::weight_hh]);
  Tensor &bias_h = context.getWeight(wt_idx[LSTMParams::bias_h]);

  Tensor &hidden_ = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &m_cell_ = context.getTensor(wt_idx[LSTMParams::mem_cell]);
  Tensor &fgio = context.getTensor(wt_idx[LSTMParams::fgio]);
  const TensorDim &input_dim = input_.getDim();

  hidden_.setZero();
  m_cell_.setZero();
  fgio.setZero();

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);
    Tensor cell = m_cell_.getBatchSlice(b, 1);
    Tensor fgio_ = fgio.getBatchSlice(b, 1);

    for (unsigned int t = 0; t < islice.height(); ++t) {
      Tensor xs =
        islice.getSharedDataTensor({islice.width()}, t * islice.width());

      Tensor hs =
        oslice.getSharedDataTensor({oslice.width()}, t * oslice.width());
      Tensor cs = cell.getSharedDataTensor({cell.width()}, t * cell.width());
      Tensor fgio_t =
        fgio_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      xs.dot(weight_xh, fgio_t);
      fgio_t.add_i(bias_h);

      if (t > 0) {
        Tensor hs_prev = oslice.getSharedDataTensor({oslice.width()},
                                                    (t - 1) * oslice.width());
        hs_prev.dot(weight_hh, fgio_t, false, false, 1.0);
      }

      Tensor hi = fgio_t.getSharedDataTensor({unit}, 0);
      Tensor hf = fgio_t.getSharedDataTensor({unit}, unit);
      Tensor hg = fgio_t.getSharedDataTensor({unit}, unit * 2);
      Tensor ho = fgio_t.getSharedDataTensor({unit}, unit * 3);

      recurrent_acti_func.run_fn(hf, hf);
      recurrent_acti_func.run_fn(hi, hi);
      recurrent_acti_func.run_fn(ho, ho);
      acti_func.run_fn(hg, hg);

      if (t > 0) {
        Tensor cs_prev =
          cell.getSharedDataTensor({cell.width()}, (t - 1) * cell.width());
        hf.multiply(cs_prev, cs);
      }
      cs.add_i(hg.multiply(hi));

      acti_func.run_fn(cs, hs);
      hs.multiply_i(ho);

      if (dropout_rate > epsilon && training) {
        Tensor mask_ = context.getTensor(wt_idx[LSTMParams::dropout_mask])
                         .getBatchSlice(b, 1);
        Tensor msk =
          mask_.getSharedDataTensor({mask_.width()}, t * mask_.width());
        msk = hs.dropout_mask(dropout_rate);
        hs.multiply_i(msk);
      }
    }
  }

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  if (!return_sequences) {
    TensorDim d = hidden_.getDim();
    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      float *data = hidden_.getAddress(b * d.width() * d.height() +
                                       (d.height() - 1) * d.width());
      float *rdata = output.getAddress(b * d.width());
      std::copy(data, data + d.width(), rdata);
    }
  } else {
    std::copy(hidden_.getData(), hidden_.getData() + hidden_.size(),
              output.getData());
  }
}

void LSTMLayer::calcDerivative(RunLayerContext &context) {
  Tensor &derivative_ = context.getTensorGrad(wt_idx[LSTMParams::fgio]);
  Tensor &weight = context.getWeight(wt_idx[LSTMParams::weight_xh]);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  derivative_.dot(weight, ret_, false, true);
}

void LSTMLayer::calcGradient(RunLayerContext &context) {
  auto unit = std::get<props::Unit>(props).get();
  Tensor &djdw_x = context.getWeightGrad(wt_idx[LSTMParams::weight_xh]);
  Tensor &djdw_h = context.getWeightGrad(wt_idx[LSTMParams::weight_hh]);
  Tensor &djdb_h = context.getWeightGrad(wt_idx[LSTMParams::bias_h]);
  Tensor &weight_hh = context.getWeight(wt_idx[LSTMParams::weight_hh]);

  djdw_x.setZero();
  djdw_h.setZero();
  djdb_h.setZero();

  Tensor &derivative_ = context.getTensorGrad(wt_idx[LSTMParams::hidden_state]);
  Tensor &hidden_ = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &incoming_deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input_.getDim();
  Tensor &m_cell_ = context.getTensor(wt_idx[LSTMParams::mem_cell]);
  Tensor &dm_cell_ = context.getTensorGrad(wt_idx[LSTMParams::mem_cell]);
  Tensor &fgio = context.getTensor(wt_idx[LSTMParams::fgio]);
  Tensor &d_fgio = context.getTensorGrad(wt_idx[LSTMParams::fgio]);

  dm_cell_.setZero();
  derivative_.setZero();
  d_fgio.setZero();

  if (!return_sequences) {
    TensorDim d = derivative_.getDim();
    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      float *data = derivative_.getAddress(b * d.width() * d.height() +
                                           (d.height() - 1) * d.width());
      float *rdata = incoming_deriv.getAddress(b * d.width());
      std::copy(rdata, rdata + d.width(), data);
    }
  } else {
    std::copy(incoming_deriv.getData(),
              incoming_deriv.getData() + incoming_deriv.size(),
              derivative_.getData());
  }

  if (dropout_rate > epsilon) {
    derivative_.multiply_i(context.getTensor(wt_idx[LSTMParams::dropout_mask]));
  }

  Tensor dh_nx = Tensor(derivative_.width());
  Tensor dc_nx = Tensor(derivative_.width());

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
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
    Tensor cs;
    Tensor dc;
    Tensor dfgio_ = d_fgio.getBatchSlice(b, 1);
    Tensor fgio_ = fgio.getBatchSlice(b, 1);

    for (unsigned int t = deriv_t.height(); t-- > 0;) {
      dh = deriv_t.getSharedDataTensor({deriv_t.width()}, t * deriv_t.width());
      dc =
        derivc_t.getSharedDataTensor({derivc_t.width()}, t * derivc_t.width());
      xs = xs_t.getSharedDataTensor({xs_t.width()}, t * xs_t.width());
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
