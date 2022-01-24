// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   rnncell.cpp
 * @date   29 Oct 2021
 * @brief  This is Recurrent Cell Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <common_properties.h>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <rnncell.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// - weight_ih ( weights of input to hidden )
// - weight_hh ( weights of hidden to hidden )
// - bias_h ( input bias, hidden_bias )
// - bias_ih ( input bias )
// - bias_hh ( hidden bias )
enum RNNCellParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  hidden_state,
  dropout_mask
};

RNNCellLayer::RNNCellLayer() :
  LayerImpl(),
  rnncell_props(props::Unit(),
                props::HiddenStateActivation() = ActivationType::ACT_TANH,
                props::DropOutRate(), props::IntegrateBias(),
                props::MaxTimestep(), props::Timestep()),
  acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void RNNCellLayer::finalize(InitLayerContext &context) {
  const nntrainer::WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  const Tensor::Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  const Tensor::Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  const nntrainer::ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(rnncell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(rnncell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(rnncell_props).get();

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("RNNCell layer takes only one input");
  }

  // input_dim = [ batch, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  if (input_dim.channel() != 1 || input_dim.height() != 1) {
    throw std::invalid_argument(
      "Input must be single time dimension for RNNCell");
  }
  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();

  // output_dim = [ batch, 1, 1, unit ]
  TensorDim output_dim(batch_size, 1, 1, unit);

  context.setOutputDimensions({output_dim});

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // weight_ih_dim : [ 1, 1, feature_size, unit ]
  const TensorDim weight_ih_dim({feature_size, unit});
  wt_idx[RNNCellParams::weight_ih] = context.requestWeight(
    weight_ih_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_ih", true);
  // weight_hh_dim : [ 1, 1, unit, unit ]
  const TensorDim weight_hh_dim({unit, unit});
  wt_idx[RNNCellParams::weight_hh] = context.requestWeight(
    weight_hh_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // bias_h_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_h_dim({unit});
      wt_idx[RNNCellParams::bias_h] = context.requestWeight(
        bias_h_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
        "bias_h", true);
    } else {
      // bias_ih_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_ih_dim({unit});
      wt_idx[RNNCellParams::bias_ih] = context.requestWeight(
        bias_ih_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_ih", true);
      // bias_hh_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_hh_dim({unit});
      wt_idx[RNNCellParams::bias_hh] = context.requestWeight(
        bias_hh_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_hh", true);
    }
  }

  // We do not need this if we reuse net_hidden[0]. But if we do, then the unit
  // test will fail. Becuase it modifies the data during gradient calculation
  // TODO : We could control with something like #define test to save memory

  // hidden_state_dim = [ max_timestep * batch, 1, 1, unit ]
  const TensorDim hidden_state_dim(max_timestep * batch_size, 1, 1, unit);
  wt_idx[RNNCellParams::hidden_state] = context.requestTensor(
    hidden_state_dim, "hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);

  if (dropout_rate > epsilon) {
    // dropout_mask_dim = [ max_timestep * batch, 1, 1, unit ]
    const TensorDim dropout_mask_dim(max_timestep * batch_size, 1, 1, unit);
    wt_idx[RNNCellParams::dropout_mask] = context.requestTensor(
      dropout_mask_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);

  if (!acti_func.supportInPlace()) {
    throw exception::not_supported(
      "Out of place activation functions not supported");
  }
}

void RNNCellLayer::setProperty(const std::vector<std::string> &values) {
  const std::vector<std::string> &remain_props =
    loadProperties(values, rnncell_props);
  LayerImpl::setProperty(remain_props);
}

void RNNCellLayer::exportTo(Exporter &exporter,
                            const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rnncell_props, method, this);
}

void RNNCellLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(rnncell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(rnncell_props).get();
  const unsigned int timestep = std::get<props::Timestep>(rnncell_props).get();

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const unsigned int batch_size = input.getDim().batch();
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &weight_ih = context.getWeight(wt_idx[RNNCellParams::weight_ih]);
  Tensor &weight_hh = context.getWeight(wt_idx[RNNCellParams::weight_hh]);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[RNNCellParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[RNNCellParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[RNNCellParams::bias_hh])
                      : empty;

  Tensor &hidden_states =
    context.getTensor(wt_idx[RNNCellParams::hidden_state]);
  hidden_states.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hidden_states.getBatchSlice(timestep - 1, 1);
  }
  Tensor hidden_state = hidden_states.getBatchSlice(timestep, 1);

  input.dot(weight_ih, hidden_state);
  prev_hidden_state.dot(weight_hh, hidden_state, false, false, 1.0f);
  if (!disable_bias) {
    if (integrate_bias) {
      hidden_state.add_i(bias_h);
    } else {
      hidden_state.add_i(bias_ih);
      hidden_state.add_i(bias_hh);
    }
  }

  acti_func.run_fn(hidden_state, hidden_state);

  if (dropout_rate > epsilon && training) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[RNNCellParams::dropout_mask]);
    dropout_mask.reshape({max_timestep, 1, batch_size, unit});
    Tensor dropout_mask_t = dropout_mask.getBatchSlice(timestep, 1);
    dropout_mask_t.dropout_mask(dropout_rate);
    hidden_state.multiply_i(dropout_mask_t);
  }

  output.copy(hidden_state);
}

void RNNCellLayer::calcDerivative(RunLayerContext &context) {
  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(rnncell_props).get();
  const unsigned int timestep = std::get<props::Timestep>(rnncell_props).get();

  const unsigned int batch_size =
    context.getInput(SINGLE_INOUT_IDX).getDim().batch();

  Tensor &hidden_states_derivatives =
    context.getTensorGrad(wt_idx[RNNCellParams::hidden_state]);
  hidden_states_derivatives.reshape({max_timestep, 1, batch_size, unit});
  Tensor hidden_state_derivative =
    hidden_states_derivatives.getBatchSlice(timestep, 1);
  Tensor &weight_ih = context.getWeight(wt_idx[RNNCellParams::weight_ih]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  hidden_state_derivative.dot(weight_ih, outgoing_derivative, false, true);
}

void RNNCellLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(rnncell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(rnncell_props).get();
  const unsigned int timestep = std::get<props::Timestep>(rnncell_props).get();

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  const unsigned int batch_size = input.getDim().batch();

  Tensor &djdweight_ih =
    context.getWeightGrad(wt_idx[RNNCellParams::weight_ih]);
  Tensor &weight_hh = context.getWeight(wt_idx[RNNCellParams::weight_hh]);
  Tensor &djdweight_hh =
    context.getWeightGrad(wt_idx[RNNCellParams::weight_hh]);
  Tensor empty;
  Tensor &djdbias_h = !disable_bias && integrate_bias
                        ? context.getWeightGrad(wt_idx[RNNCellParams::bias_h])
                        : empty;
  Tensor &djdbias_ih = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[RNNCellParams::bias_ih])
                         : empty;
  Tensor &djdbias_hh = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[RNNCellParams::bias_hh])
                         : empty;

  Tensor &hidden_states =
    context.getTensor(wt_idx[RNNCellParams::hidden_state]);
  hidden_states.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hidden_states.getBatchSlice(timestep - 1, 1);
  }
  Tensor hidden_state = hidden_states.getBatchSlice(timestep, 1);
  Tensor &hidden_states_derivatives =
    context.getTensorGrad(wt_idx[RNNCellParams::hidden_state]);
  hidden_states_derivatives.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state_derivative;
  if (!timestep) {
    prev_hidden_state_derivative = Tensor(batch_size, unit);
  } else {
    prev_hidden_state_derivative =
      hidden_states_derivatives.getBatchSlice(timestep - 1, 1);
  }
  Tensor hidden_state_derivative =
    hidden_states_derivatives.getBatchSlice(timestep, 1);

  if (timestep + 1 == max_timestep) {
    djdweight_ih.setZero();
    djdweight_hh.setZero();
    if (!disable_bias) {
      if (integrate_bias) {
        djdbias_h.setZero();
      } else {
        djdbias_ih.setZero();
        djdbias_hh.setZero();
      }
    }
    hidden_state_derivative.setZero();
  }

  hidden_state_derivative.reshape(
    {batch_size, 1, 1, unit}); // reshape to incoming_derivative dim
  hidden_state_derivative.add_i(incoming_derivative);
  hidden_state_derivative.reshape({batch_size, unit}); // restore dimension

  if (dropout_rate > epsilon) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[RNNCellParams::dropout_mask]);
    dropout_mask.reshape({max_timestep, 1, batch_size, unit});
    Tensor dropout_mask_t = dropout_mask.getBatchSlice(timestep, 1);
    hidden_state_derivative.multiply_i(dropout_mask_t);
  }

  acti_func.run_prime_fn(hidden_state, hidden_state_derivative,
                         hidden_state_derivative);

  input.dot(hidden_state_derivative, djdweight_ih, true, false, 1.0);

  hidden_state_derivative.dot(weight_hh, prev_hidden_state_derivative, false,
                              true);
  prev_hidden_state.dot(hidden_state_derivative, djdweight_hh, true, false,
                        1.0);
  if (!disable_bias) {
    if (integrate_bias) {
      hidden_state_derivative.sum(2, djdbias_h, 1.0, 1.0);
    } else {
      hidden_state_derivative.sum(2, djdbias_ih, 1.0, 1.0);
      hidden_state_derivative.sum(2, djdbias_hh, 1.0, 1.0);
    }
  }
}

void RNNCellLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(rnncell_props).get();

  context.updateTensor(wt_idx[RNNCellParams::hidden_state],
                       max_timestep * batch);

  if (dropout_rate > epsilon) {
    /// @note default value of wt_idx[dropout_mask] is 0
    context.updateTensor(wt_idx[RNNCellParams::dropout_mask],
                         max_timestep * batch);
  }
}

} // namespace nntrainer
