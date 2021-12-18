// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   zoneout_lstmcell.cpp
 * @date   30 November 2021
 * @brief  This is ZoneoutLSTMCell Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/pdf/1606.01305.pdf
 *         https://github.com/teganmaharaj/zoneout
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <zoneout_lstmcell.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum ZoneoutLSTMParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  hidden_state,
  cell_state,
  ifgo,
  lstm_cell_state,
  hidden_state_zoneout_mask,
  cell_state_zoneout_mask,
};

ZoneoutLSTMCellLayer::ZoneoutLSTMCellLayer() :
  LayerImpl(),
  zoneout_lstmcell_props(
    props::Unit(), props::IntegrateBias(),
    props::HiddenStateActivation() = ActivationType::ACT_TANH,
    props::RecurrentActivation() = ActivationType::ACT_SIGMOID,
    HiddenStateZoneOutRate(), CellStateZoneOutRate(), Test(),
    props::MaxTimestep(), props::Timestep()),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

bool ZoneoutLSTMCellLayer::HiddenStateZoneOutRate::isValid(
  const float &value) const {
  if (value < 0.0f || value > 1.0f) {
    return false;
  } else {
    return true;
  }
}

bool ZoneoutLSTMCellLayer::CellStateZoneOutRate::isValid(
  const float &value) const {
  if (value < 0.0f || value > 1.0f) {
    return false;
  } else {
    return true;
  }
}

void ZoneoutLSTMCellLayer::finalize(InitLayerContext &context) {
  const Tensor::Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props).get();
  const Tensor::Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props).get();
  const WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props).get();
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props).get();
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  NNTR_THROW_IF(std::get<props::Unit>(zoneout_lstmcell_props).empty(),
                std::invalid_argument)
    << "unit property missing for zoneout_lstmcell layer";
  const unsigned int unit = std::get<props::Unit>(zoneout_lstmcell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(zoneout_lstmcell_props).get();
  const ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(zoneout_lstmcell_props).get();
  const ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(zoneout_lstmcell_props).get();
  const bool test = std::get<Test>(zoneout_lstmcell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props).get();

  if (context.getNumInputs() != 1)
    throw std::invalid_argument("ZoneoutLSTMCellLayer takes only one input");
  if (std::get<props::MaxTimestep>(zoneout_lstmcell_props).empty())
    throw std::invalid_argument("Number of unroll steps(max timestep) must be "
                                "provided to zoneout LSTM cells");
  if (std::get<props::Timestep>(zoneout_lstmcell_props).empty())
    throw std::invalid_argument(
      "Current timestep must be provided to zoneout LSTM cell");

  // input_dim = [ batch_size, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[0];
  if (input_dim.channel() != 1 || input_dim.height() != 1)
    throw std::invalid_argument("Input must be single time dimension for "
                                "ZoneoutLSTMCell (shape should be "
                                "[batch_size, 1, 1, feature_size])");
  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();

  // output_dim = [ batch_size, 1, 1, unit ]
  const TensorDim output_dim(batch_size, 1, 1, unit);
  context.setOutputDimensions({output_dim});

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // - weight_ih ( input to hidden )
  //  : [ 1, 1, feature_size, NUM_GATE x unit ] -> i, f, g, o
  TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[ZoneoutLSTMParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // - weight_hh ( hidden to hidden )
  //  : [ 1, 1, unit, NUM_GATE x unit ] -> i, f, g, o
  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[ZoneoutLSTMParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // - bias_h ( input bias, hidden bias are integrate to 1 bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[ZoneoutLSTMParams::bias_h] =
        context.requestWeight(bias_h_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_h", true);
    } else {
      // - bias_ih ( input bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[ZoneoutLSTMParams::bias_ih] =
        context.requestWeight(bias_ih_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_ih", true);
      // - bias_hh ( hidden bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_hh_dim({NUM_GATE * unit});
      wt_idx[ZoneoutLSTMParams::bias_hh] =
        context.requestWeight(bias_hh_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_hh", true);
    }
  }

  /**
   * TODO: hidden_state is only used from the previous timestep. Once it is
   * supported as input, no need to cache the hidden_state itself
   */
  /** hidden_state_dim = [ max_timestep * batch_size, 1, 1, unit ] */
  const TensorDim hidden_state_dim(max_timestep * batch_size, 1, 1, unit);
  wt_idx[ZoneoutLSTMParams::hidden_state] = context.requestTensor(
    hidden_state_dim, "hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);
  /** cell_state_dim = [ max_timestep * batch_size, 1, 1, unit ] */
  const TensorDim cell_state_dim(max_timestep * batch_size, 1, 1, unit);
  wt_idx[ZoneoutLSTMParams::cell_state] = context.requestTensor(
    cell_state_dim, "cell_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);

  /** ifgo_dim = [ batch_size, 1, 1, NUM_GATE * unit ] */
  const TensorDim ifgo_dim(batch_size, 1, 1, NUM_GATE * unit);
  wt_idx[ZoneoutLSTMParams::ifgo] =
    context.requestTensor(ifgo_dim, "ifgo", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  /** lstm_cell_state_dim = [ batch_size, 1, 1, unit ] */
  const TensorDim lstm_cell_state_dim(batch_size, 1, 1, unit);
  wt_idx[ZoneoutLSTMParams::lstm_cell_state] = context.requestTensor(
    lstm_cell_state_dim, "lstm_cell_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  // hidden_state_zoneout_mask_dim = [ max_timestep * batch_size, 1, 1, unit ]
  const TensorDim hidden_state_zoneout_mask_dim(max_timestep * batch_size, 1, 1,
                                                unit);
  if (test) {
    wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask] =
      context.requestWeight(hidden_state_zoneout_mask_dim,
                            Tensor::Initializer::NONE, WeightRegularizer::NONE,
                            1.0f, "hidden_state_zoneout_mask", false);
  } else {
    wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask] =
      context.requestTensor(
        hidden_state_zoneout_mask_dim, "hidden_state_zoneout_mask",
        Tensor::Initializer::NONE, false, TensorLifespan::ITERATION_LIFESPAN);
  }
  // cell_state_zoneout_mask_dim = [ max_timestep * batch_size, 1, 1, unit ]
  const TensorDim cell_state_zoneout_mask_dim(max_timestep * batch_size, 1, 1,
                                              unit);
  if (test) {
    wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask] = context.requestWeight(
      cell_state_zoneout_mask_dim, Tensor::Initializer::NONE,
      WeightRegularizer::NONE, 1.0f, "cell_state_zoneout_mask", false);
  } else {
    wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask] = context.requestTensor(
      cell_state_zoneout_mask_dim, "cell_state_zoneout_mask",
      Tensor::Initializer::NONE, false, TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);
  recurrent_acti_func.setActiFunc(recurrent_activation_type);
}

void ZoneoutLSTMCellLayer::setProperty(const std::vector<std::string> &values) {
  const std::vector<std::string> &remain_props =
    loadProperties(values, zoneout_lstmcell_props);
  LayerImpl::setProperty(remain_props);
}

void ZoneoutLSTMCellLayer::exportTo(Exporter &exporter,
                                    const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(zoneout_lstmcell_props, method, this);
}

void ZoneoutLSTMCellLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(zoneout_lstmcell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(zoneout_lstmcell_props).get();
  const float hidden_state_zoneout_rate =
    std::get<HiddenStateZoneOutRate>(zoneout_lstmcell_props).get();
  const float cell_state_zoneout_rate =
    std::get<CellStateZoneOutRate>(zoneout_lstmcell_props).get();
  const bool test = std::get<Test>(zoneout_lstmcell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props).get();
  const unsigned int timestep =
    std::get<props::Timestep>(zoneout_lstmcell_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const unsigned int batch_size = input.getDim().batch();

  const Tensor &weight_ih =
    context.getWeight(wt_idx[ZoneoutLSTMParams::weight_ih]);
  const Tensor &weight_hh =
    context.getWeight(wt_idx[ZoneoutLSTMParams::weight_hh]);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[ZoneoutLSTMParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[ZoneoutLSTMParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[ZoneoutLSTMParams::bias_hh])
                      : empty;

  Tensor &hs = context.getTensor(wt_idx[ZoneoutLSTMParams::hidden_state]);
  hs.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, 1, 1, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hs.getBatchSlice(timestep - 1, 1);
    prev_hidden_state.reshape({batch_size, 1, 1, unit});
  }
  Tensor hidden_state = hs.getBatchSlice(timestep, 1);
  hidden_state.reshape({batch_size, 1, 1, unit});

  Tensor &cs = context.getTensor(wt_idx[ZoneoutLSTMParams::cell_state]);
  cs.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_cell_state;
  if (!timestep) {
    prev_cell_state = Tensor(batch_size, 1, 1, unit);
    prev_cell_state.setZero();
  } else {
    prev_cell_state = cs.getBatchSlice(timestep - 1, 1);
    prev_cell_state.reshape({batch_size, 1, 1, unit});
  }
  Tensor cell_state = cs.getBatchSlice(timestep, 1);
  cell_state.reshape({batch_size, 1, 1, unit});

  Tensor &ifgo = context.getTensor(wt_idx[ZoneoutLSTMParams::ifgo]);

  Tensor &lstm_cell_state =
    context.getTensor(wt_idx[ZoneoutLSTMParams::lstm_cell_state]);

  lstmcell_forwarding(unit, batch_size, disable_bias, integrate_bias, acti_func,
                      recurrent_acti_func, input, prev_hidden_state,
                      prev_cell_state, hidden_state, lstm_cell_state, weight_ih,
                      weight_hh, bias_h, bias_ih, bias_hh, ifgo);

  if (training) {
    Tensor &hs_zoneout_mask =
      test ? context.getWeight(
               wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask])
           : context.getTensor(
               wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask]);
    hs_zoneout_mask.reshape({max_timestep, 1, batch_size, unit});
    Tensor hidden_state_zoneout_mask =
      hs_zoneout_mask.getBatchSlice(timestep, 1);
    hidden_state_zoneout_mask.reshape({batch_size, 1, 1, unit});
    Tensor prev_hidden_state_zoneout_mask;
    if (!test) {
      prev_hidden_state_zoneout_mask =
        hidden_state_zoneout_mask.zoneout_mask(hidden_state_zoneout_rate);
    } else {
      hidden_state_zoneout_mask.multiply(-1.0f, prev_hidden_state_zoneout_mask);
      prev_hidden_state_zoneout_mask.add_i(1.0f);
    }

    hidden_state.multiply_i(hidden_state_zoneout_mask);
    prev_hidden_state.multiply(prev_hidden_state_zoneout_mask, hidden_state,
                               1.0f);

    Tensor &cs_zoneout_mask =
      test
        ? context.getWeight(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask])
        : context.getTensor(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask]);
    cs_zoneout_mask.reshape({max_timestep, 1, batch_size, unit});
    Tensor cell_state_zoneout_mask = cs_zoneout_mask.getBatchSlice(timestep, 1);
    cell_state_zoneout_mask.reshape({batch_size, 1, 1, unit});
    Tensor prev_cell_state_zoneout_mask;
    if (!test) {
      prev_cell_state_zoneout_mask =
        cell_state_zoneout_mask.zoneout_mask(cell_state_zoneout_rate);
    } else {
      cell_state_zoneout_mask.multiply(-1.0f, prev_cell_state_zoneout_mask);
      prev_cell_state_zoneout_mask.add_i(1.0f);
    }

    lstm_cell_state.multiply(cell_state_zoneout_mask, cell_state);
    prev_cell_state.multiply(prev_cell_state_zoneout_mask, cell_state, 1.0f);
  }
  // Todo: zoneout at inference

  output.copyData(hidden_state);
}

void ZoneoutLSTMCellLayer::calcDerivative(RunLayerContext &context) {
  Tensor &d_ifgo = context.getTensorGrad(wt_idx[ZoneoutLSTMParams::ifgo]);
  const Tensor &weight_ih =
    context.getWeight(wt_idx[ZoneoutLSTMParams::weight_ih]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  lstmcell_calcDerivative(d_ifgo, weight_ih, outgoing_derivative);
}

void ZoneoutLSTMCellLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(zoneout_lstmcell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(zoneout_lstmcell_props).get();
  const float hidden_state_zoneout_rate =
    std::get<HiddenStateZoneOutRate>(zoneout_lstmcell_props).get();
  const float cell_state_zoneout_rate =
    std::get<CellStateZoneOutRate>(zoneout_lstmcell_props).get();
  const bool test = std::get<Test>(zoneout_lstmcell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props).get();
  const unsigned int timestep =
    std::get<props::Timestep>(zoneout_lstmcell_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  unsigned int batch_size = input.getDim().batch();

  Tensor &d_weight_ih =
    context.getWeightGrad(wt_idx[ZoneoutLSTMParams::weight_ih]);
  const Tensor &weight_hh =
    context.getWeight(wt_idx[ZoneoutLSTMParams::weight_hh]);
  Tensor &d_weight_hh =
    context.getWeightGrad(wt_idx[ZoneoutLSTMParams::weight_hh]);
  Tensor empty;
  Tensor &d_bias_h =
    !disable_bias && integrate_bias
      ? context.getWeightGrad(wt_idx[ZoneoutLSTMParams::bias_h])
      : empty;
  Tensor &d_bias_ih =
    !disable_bias && !integrate_bias
      ? context.getWeightGrad(wt_idx[ZoneoutLSTMParams::bias_ih])
      : empty;
  Tensor &d_bias_hh =
    !disable_bias && !integrate_bias
      ? context.getWeightGrad(wt_idx[ZoneoutLSTMParams::bias_hh])
      : empty;

  Tensor &hs = context.getTensor(wt_idx[ZoneoutLSTMParams::hidden_state]);
  hs.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, 1, 1, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hs.getBatchSlice(timestep - 1, 1);
    prev_hidden_state.reshape({batch_size, 1, 1, unit});
  }

  Tensor &d_hs = context.getTensorGrad(wt_idx[ZoneoutLSTMParams::hidden_state]);
  d_hs.reshape({max_timestep, 1, batch_size, unit});
  Tensor d_prev_hidden_state;
  if (!timestep) {
    d_prev_hidden_state = Tensor(batch_size, 1, 1, unit);
    d_prev_hidden_state.setZero();
  } else {
    d_prev_hidden_state = d_hs.getBatchSlice(timestep - 1, 1);
    d_prev_hidden_state.reshape({batch_size, 1, 1, unit});
  }
  Tensor d_hidden_state = d_hs.getBatchSlice(timestep, 1);
  d_hidden_state.reshape({batch_size, 1, 1, unit});

  Tensor &cs = context.getTensor(wt_idx[ZoneoutLSTMParams::cell_state]);
  cs.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_cell_state;
  if (!timestep) {
    prev_cell_state = Tensor(batch_size, 1, 1, unit);
    prev_cell_state.setZero();
  } else {
    prev_cell_state = cs.getBatchSlice(timestep - 1, 1);
    prev_cell_state.reshape({batch_size, 1, 1, unit});
  }
  Tensor cell_state = cs.getBatchSlice(timestep, 1);
  cell_state.reshape({batch_size, 1, 1, unit});

  Tensor &d_cs = context.getTensorGrad(wt_idx[ZoneoutLSTMParams::cell_state]);
  d_cs.reshape({max_timestep, 1, batch_size, unit});
  Tensor d_prev_cell_state;
  if (!timestep) {
    d_prev_cell_state = Tensor(batch_size, 1, 1, unit);
    d_prev_cell_state.setZero();
  } else {
    d_prev_cell_state = d_cs.getBatchSlice(timestep - 1, 1);
    d_prev_cell_state.reshape({batch_size, 1, 1, unit});
  }
  Tensor d_cell_state = d_cs.getBatchSlice(timestep, 1);
  d_cell_state.reshape({batch_size, 1, 1, unit});

  Tensor &ifgo = context.getTensor(wt_idx[ZoneoutLSTMParams::ifgo]);
  Tensor &d_ifgo = context.getTensorGrad(wt_idx[ZoneoutLSTMParams::ifgo]);

  const Tensor &lstm_cell_state =
    context.getTensor(wt_idx[ZoneoutLSTMParams::lstm_cell_state]);
  Tensor &d_lstm_cell_state =
    context.getTensorGrad(wt_idx[ZoneoutLSTMParams::lstm_cell_state]);

  if (timestep + 1 == max_timestep) {
    d_weight_ih.setZero();
    d_weight_hh.setZero();
    if (!disable_bias) {
      if (integrate_bias) {
        d_bias_h.setZero();
      } else {
        d_bias_ih.setZero();
        d_bias_hh.setZero();
      }
    }
    d_hidden_state.setZero();
    d_cell_state.setZero();
  }

  d_hidden_state.add_i(incoming_derivative);

  Tensor d_prev_hidden_state_residual;

  Tensor &hs_zoneout_mask =
    test
      ? context.getWeight(wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask])
      : context.getTensor(wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask]);
  hs_zoneout_mask.reshape({max_timestep, 1, batch_size, unit});
  Tensor hidden_state_zoneout_mask = hs_zoneout_mask.getBatchSlice(timestep, 1);
  hidden_state_zoneout_mask.reshape({batch_size, 1, 1, unit});
  Tensor prev_hidden_state_zoneout_mask;
  if (!test) {
    prev_hidden_state_zoneout_mask =
      hidden_state_zoneout_mask.zoneout_mask(hidden_state_zoneout_rate);
  } else {
    hidden_state_zoneout_mask.multiply(-1.0f, prev_hidden_state_zoneout_mask);
    prev_hidden_state_zoneout_mask.add_i(1.0f);
  }

  d_hidden_state.multiply(prev_hidden_state_zoneout_mask,
                          d_prev_hidden_state_residual);
  d_hidden_state.multiply_i(hidden_state_zoneout_mask);

  Tensor d_prev_cell_state_residual;

  Tensor &cs_zoneout_mask =
    test
      ? context.getWeight(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask])
      : context.getTensor(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask]);
  cs_zoneout_mask.reshape({max_timestep, 1, batch_size, unit});
  Tensor cell_state_zoneout_mask = cs_zoneout_mask.getBatchSlice(timestep, 1);
  cell_state_zoneout_mask.reshape({batch_size, 1, 1, unit});
  Tensor prev_cell_state_zoneout_mask;
  if (!test) {
    prev_cell_state_zoneout_mask =
      cell_state_zoneout_mask.zoneout_mask(cell_state_zoneout_rate);
  } else {
    cell_state_zoneout_mask.multiply(-1.0f, prev_cell_state_zoneout_mask);
    prev_cell_state_zoneout_mask.add_i(1.0f);
  }

  d_cell_state.multiply(prev_cell_state_zoneout_mask,
                        d_prev_cell_state_residual);
  d_cell_state.multiply(cell_state_zoneout_mask, d_lstm_cell_state);

  lstmcell_calcGradient(unit, batch_size, disable_bias, integrate_bias,
                        acti_func, recurrent_acti_func, input,
                        prev_hidden_state, d_prev_hidden_state, prev_cell_state,
                        d_prev_cell_state, d_hidden_state, lstm_cell_state,
                        d_lstm_cell_state, d_weight_ih, weight_hh, d_weight_hh,
                        d_bias_h, d_bias_ih, d_bias_hh, ifgo, d_ifgo);

  d_prev_hidden_state.add_i(d_prev_hidden_state_residual);
  d_prev_cell_state.add_i(d_prev_cell_state_residual);
}

void ZoneoutLSTMCellLayer::setBatch(RunLayerContext &context,
                                    unsigned int batch) {
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props);

  context.updateTensor(wt_idx[ZoneoutLSTMParams::hidden_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[ZoneoutLSTMParams::cell_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[ZoneoutLSTMParams::ifgo], batch);
  context.updateTensor(wt_idx[ZoneoutLSTMParams::lstm_cell_state], batch);

  context.updateTensor(wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask],
                       max_timestep * batch);
  context.updateTensor(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask],
                       max_timestep * batch);
}

} // namespace nntrainer
