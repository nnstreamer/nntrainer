// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   rnncell.cpp
 * @date   29 Oct 2021
 * @brief  This is Recurrent Layer Cell Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>

#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <rnncell.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// - weight_xh ( weights of input to hidden )
// - weight_hh ( weights of hidden to hidden )
// - bias_h ( hidden bias )
enum RNNCellParams { weight_xh, weight_hh, bias_h, hidden_state, dropout_mask };

RNNCellLayer::RNNCellLayer() :
  LayerImpl(),
  rnncell_props(props::Unit(), props::HiddenStateActivation(),
                props::DropOutRate(), props::MaxTimestep(), props::Timestep()),
  wt_idx({0}),
  acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {}

void RNNCellLayer::finalize(InitLayerContext &context) {
  const nntrainer::WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  const Tensor::Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  const Tensor::Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props);

  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  nntrainer::props::HiddenStateActivation hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(rnncell_props);
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props);
  const unsigned int max_timestep = std::get<props::MaxTimestep>(rnncell_props);

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("RNNCell layer takes only one input");
  }

  // input_dim = [ batch, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[0];
  if (input_dim.channel() != 1 && input_dim.height() != 1) {
    throw std::invalid_argument(
      "Input must be single time dimension for RNNCell");
  }
  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();

  // outut_dim = [ batch, 1, 1, hidden_size ( unit ) ]
  TensorDim output_dim(batch_size, 1, 1, unit);

  if (dropout_rate > epsilon) {
    wt_idx[RNNCellParams::dropout_mask] = context.requestTensor(
      output_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  context.setOutputDimensions({output_dim});

  // weight_xh_dim : [1, 1, input_size, unit]
  const TensorDim weight_xh_dim({feature_size, unit});
  // weight_hh_dim : [1, 1, unit, unit]
  const TensorDim weight_hh_dim({unit, unit});
  // bias_h_dim : [1, 1, 1, unit]
  const TensorDim bias_h_dim({unit});

  // weight_initializer can be set seperately. weight_xh initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.
  wt_idx[RNNCellParams::weight_xh] =
    context.requestWeight(weight_xh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_xh", true);
  wt_idx[RNNCellParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  wt_idx[RNNCellParams::bias_h] =
    context.requestWeight(bias_h_dim, bias_initializer, WeightRegularizer::NONE,
                          1.0f, "bias_h", true);

  // We do not need this if we reuse net_hidden[0]. But if we do, then the unit
  // test will fail. Becuase it modifies the data during gradient calculation
  // TODO : We could control with something like #define test to save memory
  const TensorDim dim(batch_size * max_timestep, 1, 1, unit);
  wt_idx[RNNCellParams::hidden_state] =
    context.requestTensor(dim, "hidden_state", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  if (hidden_state_activation_type.get() == ActivationType::ACT_NONE) {
    hidden_state_activation_type.set(ActivationType::ACT_TANH);
  }
  acti_func.setActiFunc(hidden_state_activation_type.get());

  if (!acti_func.supportInPlace()) {
    throw exception::not_supported(
      "Out of place activation functions not supported");
  }
}

void RNNCellLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, rnncell_props);
  LayerImpl::setProperty(remain_props);
}

void RNNCellLayer::exportTo(Exporter &exporter,
                            const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rnncell_props, method, this);
}

void RNNCellLayer::forwarding(RunLayerContext &context, bool training) {
  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props);
  const unsigned int max_timestep = std::get<props::MaxTimestep>(rnncell_props);
  const unsigned int timestep = std::get<props::Timestep>(rnncell_props);

  Tensor &weight_xh = context.getWeight(wt_idx[RNNCellParams::weight_xh]);
  Tensor &weight_hh = context.getWeight(wt_idx[RNNCellParams::weight_hh]);
  Tensor &bias_h = context.getWeight(wt_idx[RNNCellParams::bias_h]);

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim[0];
  Tensor &hidden_states =
    context.getTensor(wt_idx[RNNCellParams::hidden_state]);
  hidden_states.reshape({max_timestep, 1, batch_size, unit});
  Tensor hidden_state = hidden_states.getBatchSlice(timestep, 1);

  input.dot(weight_xh, hidden_state);
  if (timestep) {
    Tensor prev_hidden_state = hidden_states.getBatchSlice(timestep - 1, 1);
    prev_hidden_state.dot(weight_hh, hidden_state, false, false, 1.0f);
  }
  hidden_state.add_i(bias_h);
  acti_func.run_fn(hidden_state, hidden_state);
  if (dropout_rate > epsilon && training) {
    Tensor &mask = context.getTensor(wt_idx[RNNCellParams::dropout_mask]);
    mask.dropout_mask(dropout_rate);
    hidden_state.multiply_i(mask);
  }

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  output.copy(hidden_state);
}

void RNNCellLayer::calcDerivative(RunLayerContext &context) {
  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  const unsigned int max_timestep = std::get<props::MaxTimestep>(rnncell_props);
  const unsigned int timestep = std::get<props::Timestep>(rnncell_props);
  const TensorDim &input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  const unsigned int batch_size = input_dim.batch();
  Tensor &hidden_states_derivatives =
    context.getTensorGrad(wt_idx[RNNCellParams::hidden_state]);
  hidden_states_derivatives.reshape({max_timestep, 1, batch_size, unit});
  Tensor hidden_state_derivative =
    hidden_states_derivatives.getBatchSlice(timestep, 1);

  Tensor &weight_xh = context.getWeight(wt_idx[RNNCellParams::weight_xh]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  hidden_state_derivative.dot(weight_xh, outgoing_derivative, false, true);
}

void RNNCellLayer::calcGradient(RunLayerContext &context) {
  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props);
  const unsigned int max_timestep = std::get<props::MaxTimestep>(rnncell_props);
  const unsigned int timestep = std::get<props::Timestep>(rnncell_props);

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &incoming_derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();

  Tensor &djdweight_xh =
    context.getWeightGrad(wt_idx[RNNCellParams::weight_xh]);
  Tensor &djdweight_hh =
    context.getWeightGrad(wt_idx[RNNCellParams::weight_hh]);
  Tensor &djdbias_h = context.getWeightGrad(wt_idx[RNNCellParams::bias_h]);
  Tensor &weight_hh = context.getWeight(wt_idx[RNNCellParams::weight_hh]);

  Tensor &hidden_states =
    context.getTensor(wt_idx[RNNCellParams::hidden_state]);
  hidden_states.reshape({max_timestep, 1, batch_size, unit});
  Tensor hidden_state = hidden_states.getBatchSlice(timestep, 1);
  Tensor &hidden_states_derivatives =
    context.getTensorGrad(wt_idx[RNNCellParams::hidden_state]);
  hidden_states_derivatives.reshape({max_timestep, 1, batch_size, unit});
  Tensor hidden_state_derivative =
    hidden_states_derivatives.getBatchSlice(timestep, 1);

  if (timestep + 1 == max_timestep) {
    djdweight_xh.setZero();
    djdweight_hh.setZero();
    djdbias_h.setZero();
  }

  hidden_state_derivative.reshape(incoming_derivative.getDim());
  if (timestep + 1 == max_timestep) {
    hidden_state_derivative.copyData(incoming_derivative);
  } else {
    hidden_state_derivative.add_i(incoming_derivative);
  }
  // restore the dimension
  hidden_state_derivative.reshape({1, 1, batch_size, unit});

  if (dropout_rate > epsilon) {
    hidden_state_derivative.multiply_i(
      context.getTensor(wt_idx[RNNCellParams::dropout_mask]));
  }

  acti_func.run_prime_fn(hidden_state, hidden_state_derivative,
                         hidden_state_derivative);

  input.dot(hidden_state_derivative, djdweight_xh, true, false, 1.0);
  hidden_state_derivative.sum(2, djdbias_h, 1.0, 1.0);

  if (timestep) {
    Tensor prev_hidden_state = hidden_states.getBatchSlice(timestep - 1, 1);
    prev_hidden_state.dot(hidden_state_derivative, djdweight_hh, true, false,
                          1.0);
    Tensor prev_hidden_state_derivative =
      hidden_states_derivatives.getBatchSlice(timestep - 1, 1);
    hidden_state_derivative.dot(weight_hh, prev_hidden_state_derivative, false,
                                true);
  }
}

void RNNCellLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  const unsigned int max_timestep = std::get<props::MaxTimestep>(rnncell_props);
  context.updateTensor(wt_idx[RNNCellParams::hidden_state],
                       batch * max_timestep);
  context.updateTensor(wt_idx[RNNCellParams::dropout_mask], batch);
}

} // namespace nntrainer
