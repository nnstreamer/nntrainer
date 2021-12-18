// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lstmcell.cpp
 * @date   17 March 2021
 * @brief  This is LSTMCell Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <lstmcell.h>
#include <lstmcell_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LSTMCellParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  hidden_state,
  cell_state,
  ifgo,
  dropout_mask
};

LSTMCellLayer::LSTMCellLayer() :
  LayerImpl(),
  lstmcell_props(props::Unit(), props::IntegrateBias(),
                 props::HiddenStateActivation() = ActivationType::ACT_TANH,
                 props::RecurrentActivation() = ActivationType::ACT_SIGMOID,
                 props::DropOutRate(), props::MaxTimestep(), props::Timestep()),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void LSTMCellLayer::finalize(InitLayerContext &context) {
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

  NNTR_THROW_IF(std::get<props::Unit>(lstmcell_props).empty(),
                std::invalid_argument)
    << "unit property missing for lstmcell layer";
  const unsigned int unit = std::get<props::Unit>(lstmcell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcell_props).get();
  const ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(lstmcell_props).get();
  const ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(lstmcell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props).get();

  if (context.getNumInputs() != 1)
    throw std::invalid_argument("LSTMCell layer takes only one input");
  if (std::get<props::MaxTimestep>(lstmcell_props).empty())
    throw std::invalid_argument(
      "Number of unroll steps(max timestep) must be provided to LSTM cell");
  if (std::get<props::Timestep>(lstmcell_props).empty())
    throw std::invalid_argument(
      "Current Timestep must be provided to LSTM cell");

  // input_dim = [ batch_size, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[0];
  if (input_dim.channel() != 1 || input_dim.height() != 1)
    throw std::invalid_argument(
      "Input must be single time dimension for LSTMCell (shape should be "
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
  wt_idx[LSTMCellParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // - weight_hh ( hidden to hidden )
  //  : [ 1, 1, unit, NUM_GATE x unit ] -> i, f, g, o
  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[LSTMCellParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // - bias_h ( input bias, hidden bias are integrate to 1 bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[LSTMCellParams::bias_h] =
        context.requestWeight(bias_h_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_h", true);
    } else {
      // - bias_ih ( input bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[LSTMCellParams::bias_ih] =
        context.requestWeight(bias_ih_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_ih", true);
      // - bias_hh ( hidden bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_hh_dim({NUM_GATE * unit});
      wt_idx[LSTMCellParams::bias_hh] =
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
  wt_idx[LSTMCellParams::hidden_state] = context.requestTensor(
    hidden_state_dim, "hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);
  /** cell_state_dim = [ max_timestep * batch_size, 1, 1, unit ] */
  const TensorDim cell_state_dim(max_timestep * batch_size, 1, 1, unit);
  wt_idx[LSTMCellParams::cell_state] = context.requestTensor(
    cell_state_dim, "cell_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);

  /** ifgo_dim = [ batch_size, 1, 1, NUM_GATE * unit ] */
  const TensorDim ifgo_dim(batch_size, 1, 1, NUM_GATE * unit);
  wt_idx[LSTMCellParams::ifgo] =
    context.requestTensor(ifgo_dim, "ifgo", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  if (dropout_rate > epsilon) {
    // dropout_mask_dim = [ batch_size, 1, 1, unit ]
    const TensorDim dropout_mask_dim(batch_size, 1, 1, unit);
    wt_idx[LSTMCellParams::dropout_mask] = context.requestTensor(
      dropout_mask_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);
  recurrent_acti_func.setActiFunc(recurrent_activation_type);
}

void LSTMCellLayer::setProperty(const std::vector<std::string> &values) {
  const std::vector<std::string> &remain_props =
    loadProperties(values, lstmcell_props);
  LayerImpl::setProperty(remain_props);
}

void LSTMCellLayer::exportTo(Exporter &exporter,
                             const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(lstmcell_props, method, this);
}

void LSTMCellLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstmcell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props).get();
  const unsigned int timestep = std::get<props::Timestep>(lstmcell_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  const unsigned int batch_size = input.getDim().batch();

  const Tensor &weight_ih =
    context.getWeight(wt_idx[LSTMCellParams::weight_ih]);
  const Tensor &weight_hh =
    context.getWeight(wt_idx[LSTMCellParams::weight_hh]);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[LSTMCellParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[LSTMCellParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[LSTMCellParams::bias_hh])
                      : empty;

  Tensor &hs = context.getTensor(wt_idx[LSTMCellParams::hidden_state]);
  hs.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hs.getBatchSlice(timestep - 1, 1);
  }
  prev_hidden_state.reshape({batch_size, 1, 1, unit});
  Tensor hidden_state = hs.getBatchSlice(timestep, 1);
  hidden_state.reshape({batch_size, 1, 1, unit});

  Tensor &cs = context.getTensor(wt_idx[LSTMCellParams::cell_state]);
  cs.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_cell_state;
  if (!timestep) {
    prev_cell_state = Tensor(batch_size, unit);
    prev_cell_state.setZero();
  } else {
    prev_cell_state = cs.getBatchSlice(timestep - 1, 1);
  }
  prev_cell_state.reshape({batch_size, 1, 1, unit});
  Tensor cell_state = cs.getBatchSlice(timestep, 1);
  cell_state.reshape({batch_size, 1, 1, unit});

  Tensor &ifgo = context.getTensor(wt_idx[LSTMCellParams::ifgo]);

  lstmcell_forwarding(unit, batch_size, disable_bias, integrate_bias, acti_func,
                      recurrent_acti_func, input, prev_hidden_state,
                      prev_cell_state, hidden_state, cell_state, weight_ih,
                      weight_hh, bias_h, bias_ih, bias_hh, ifgo);

  if (dropout_rate > epsilon && training) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[LSTMCellParams::dropout_mask]);
    dropout_mask.dropout_mask(dropout_rate);
    hidden_state.multiply_i(dropout_mask);
  }

  output.copyData(hidden_state);
}

void LSTMCellLayer::calcDerivative(RunLayerContext &context) {
  Tensor &d_ifgo = context.getTensorGrad(wt_idx[LSTMCellParams::ifgo]);
  const Tensor &weight_ih =
    context.getWeight(wt_idx[LSTMCellParams::weight_ih]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  lstmcell_calcDerivative(d_ifgo, weight_ih, outgoing_derivative);
}

void LSTMCellLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstmcell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props);
  const unsigned int timestep = std::get<props::Timestep>(lstmcell_props);

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  unsigned int batch_size = input.getDim().batch();

  Tensor &d_weight_ih =
    context.getWeightGrad(wt_idx[LSTMCellParams::weight_ih]);
  const Tensor &weight_hh =
    context.getWeight(wt_idx[LSTMCellParams::weight_hh]);
  Tensor &d_weight_hh =
    context.getWeightGrad(wt_idx[LSTMCellParams::weight_hh]);
  Tensor empty;
  Tensor &d_bias_h = !disable_bias && integrate_bias
                       ? context.getWeightGrad(wt_idx[LSTMCellParams::bias_h])
                       : empty;
  Tensor &d_bias_ih = !disable_bias && !integrate_bias
                        ? context.getWeightGrad(wt_idx[LSTMCellParams::bias_ih])
                        : empty;
  Tensor &d_bias_hh = !disable_bias && !integrate_bias
                        ? context.getWeightGrad(wt_idx[LSTMCellParams::bias_hh])
                        : empty;

  Tensor &hs = context.getTensor(wt_idx[LSTMCellParams::hidden_state]);
  hs.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hs.getBatchSlice(timestep - 1, 1);
  }
  prev_hidden_state.reshape({batch_size, 1, 1, unit});

  Tensor &d_hs = context.getTensorGrad(wt_idx[LSTMCellParams::hidden_state]);
  d_hs.reshape({max_timestep, 1, batch_size, unit});
  Tensor d_prev_hidden_state;
  if (!timestep) {
    d_prev_hidden_state = Tensor(batch_size, unit);
    d_prev_hidden_state.setZero();
  } else {
    d_prev_hidden_state = d_hs.getBatchSlice(timestep - 1, 1);
  }
  d_prev_hidden_state.reshape({batch_size, 1, 1, unit});
  Tensor d_hidden_state = d_hs.getBatchSlice(timestep, 1);
  d_hidden_state.reshape({batch_size, 1, 1, unit});

  Tensor &cs = context.getTensor(wt_idx[LSTMCellParams::cell_state]);
  cs.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_cell_state;
  if (!timestep) {
    prev_cell_state = Tensor(batch_size, unit);
    prev_cell_state.setZero();
  } else {
    prev_cell_state = cs.getBatchSlice(timestep - 1, 1);
  }
  prev_cell_state.reshape({batch_size, 1, 1, unit});
  Tensor cell_state = cs.getBatchSlice(timestep, 1);
  cell_state.reshape({batch_size, 1, 1, unit});

  Tensor &d_cs = context.getTensorGrad(wt_idx[LSTMCellParams::cell_state]);
  d_cs.reshape({max_timestep, 1, batch_size, unit});
  Tensor d_prev_cell_state;
  if (!timestep) {
    d_prev_cell_state = Tensor(batch_size, unit);
    d_prev_cell_state.setZero();
  } else {
    d_prev_cell_state = d_cs.getBatchSlice(timestep - 1, 1);
  }
  d_prev_cell_state.reshape({batch_size, 1, 1, unit});
  Tensor d_cell_state = d_cs.getBatchSlice(timestep, 1);
  d_cell_state.reshape({batch_size, 1, 1, unit});

  const Tensor &ifgo = context.getTensor(wt_idx[LSTMCellParams::ifgo]);
  Tensor &d_ifgo = context.getTensorGrad(wt_idx[LSTMCellParams::ifgo]);

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

  if (dropout_rate > epsilon) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[LSTMCellParams::dropout_mask]);
    d_hidden_state.multiply_i(dropout_mask);
  }

  d_hidden_state.add_i(incoming_derivative);

  lstmcell_calcGradient(unit, batch_size, disable_bias, integrate_bias,
                        acti_func, recurrent_acti_func, input,
                        prev_hidden_state, d_prev_hidden_state, prev_cell_state,
                        d_prev_cell_state, d_hidden_state, cell_state,
                        d_cell_state, d_weight_ih, weight_hh, d_weight_hh,
                        d_bias_h, d_bias_ih, d_bias_hh, ifgo, d_ifgo);
}

void LSTMCellLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props);
  context.updateTensor(wt_idx[LSTMCellParams::hidden_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[LSTMCellParams::cell_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[LSTMCellParams::ifgo], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(wt_idx[LSTMCellParams::dropout_mask], batch);
  }
}

} // namespace nntrainer
