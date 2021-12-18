// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   lstmcell_core.cpp
 * @date   25 November 2021
 * @brief  This is LSTMCellCore Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <lstmcell_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

// ENABLE_SHARING_WT_IDX implies does the wt_idx of lstm_core can be shared
// with lstm_cell variant layer.
// Todo: remove this if sharing wt_idx with other lstm variant is enabled
#define ENABLE_SHARING_WT_IDX 0

namespace nntrainer {

namespace init_lstm_context {
void fillLayerInitContext(InitLayerContext &context,
                          const InitLayerContext &core_context) {
  /** real set the input flags */
  auto const &input_dims = context.getInputDimensions();
  for (unsigned int idx = 0; idx < core_context.getNumInputs(); idx++) {
    context.setDynDimFlagInputDimension(idx, input_dims[idx].getDynDimFlag());
    context.setEffDimFlagInputDimension(idx, input_dims[idx].getEffDimFlag());
  }

  /** real request of tensors */
  for (auto const &ts : core_context.getTensorsSpec())
    context.requestTensor(ts);

  /** real request of weights */
  for (auto const &ws : core_context.getWeightsSpec())
    context.requestWeight(ws);
}

void fillWeights(std::vector<Weight> &weights, const RunLayerContext &context,
                 bool training, const std::vector<unsigned int> &wt_idx,
                 const unsigned int max_timestep, const unsigned int timestep,
                 bool test) {
  weights.resize(context.getNumWeights());
  for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
    if (training && (!test || i < context.getNumWeights() - 2)) {
      weights[i] =
        Weight(context.getWeight(wt_idx[i]), context.getWeightGrad(wt_idx[i]),
               context.getWeightName(wt_idx[i]));
    } else {
      weights[i] = Weight(context.getWeight(wt_idx[i]), Tensor(),
                          context.getWeightName(wt_idx[i]));
    }
  }
}

const std::vector<Weight *> getWeights(std::vector<Weight> &weights) {
  std::vector<Weight *> ret(weights.size());
  for (unsigned int i = 0; i < weights.size(); ++i) {
    ret[i] = &weights[i];
  }
  return ret;
}

void fillInputs(std::vector<Var_Grad> &inputs, RunLayerContext &context,
                bool training, const std::vector<unsigned int> &wt_idx,
                const unsigned int max_timestep, const unsigned int timestep) {
  inputs.resize(3);
  Tensor empty;
  const TensorDim &output_dim = context.getOutput(wt_idx[0]).getDim();
  const unsigned int batch_size = output_dim.batch();
  const unsigned int unit = output_dim.width();

  const Tensor &input = context.getInput(wt_idx[0]);
  const Tensor &outgoing_derivative =
    training ? context.getOutgoingDerivative(0) : empty;

  Tensor &hidden_state = context.getTensor(wt_idx[1]);
  hidden_state.reshape({max_timestep, 1, batch_size, unit});
  Tensor &hidden_state_derivative =
    training ? context.getTensorGrad(wt_idx[1]) : empty;
  if (training) {
    hidden_state_derivative.reshape({max_timestep, 1, batch_size, unit});
  }

  Tensor &cell_state = context.getTensor(wt_idx[2]);
  cell_state.reshape({max_timestep, 1, batch_size, unit});
  Tensor &cell_state_derivative =
    training ? context.getTensorGrad(wt_idx[2]) : empty;
  if (training) {
    cell_state_derivative.reshape({max_timestep, 1, batch_size, unit});
  }

  Tensor prev_hidden_state;
  Tensor prev_hidden_state_derivative;
  Tensor prev_cell_state;
  Tensor prev_cell_state_derivative;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, 1, 1, unit);
    prev_hidden_state.setZero();
    prev_hidden_state_derivative = Tensor(batch_size, 1, 1, unit);
    prev_hidden_state_derivative.setZero();
    prev_cell_state = Tensor(batch_size, 1, 1, unit);
    prev_cell_state.setZero();
    prev_cell_state_derivative = Tensor(batch_size, 1, 1, unit);
    prev_cell_state_derivative.setZero();
  } else {
    prev_hidden_state = hidden_state.getBatchSlice(timestep - 1, 1);
    prev_hidden_state.reshape({batch_size, 1, 1, unit});
    if (training) {
      prev_hidden_state_derivative =
        hidden_state_derivative.getBatchSlice(timestep - 1, 1);
      prev_hidden_state_derivative.reshape({batch_size, 1, 1, unit});
    }
    prev_cell_state = cell_state.getBatchSlice(timestep - 1, 1);
    prev_cell_state.reshape({batch_size, 1, 1, unit});
    if (training) {
      prev_cell_state_derivative =
        cell_state_derivative.getBatchSlice(timestep - 1, 1);
      prev_cell_state_derivative.reshape({batch_size, 1, 1, unit});
    }
  }

  inputs[0] = Var_Grad(input, outgoing_derivative, "lstmcell_core input");
  inputs[1] = Var_Grad(prev_hidden_state, prev_hidden_state_derivative,
                       context.getTensorName(wt_idx[1]));
  inputs[2] = Var_Grad(prev_cell_state, prev_cell_state_derivative,
                       context.getTensorName(wt_idx[2]));
}

const std::vector<Var_Grad *> getInputs(std::vector<Var_Grad> &inputs) {
  std::vector<Var_Grad *> ret(inputs.size());
  for (unsigned int i = 0; i < inputs.size(); ++i) {
    ret[i] = &inputs[i];
  }
  return ret;
}

void fillOutputs(std::vector<Var_Grad> &outputs, RunLayerContext &context,
                 bool training, const std::vector<unsigned int> &wt_idx,
                 const unsigned int max_timestep, const unsigned int timestep) {
  outputs.resize(2);
  Tensor empty;
  const TensorDim &output_dim = context.getOutput(wt_idx[0]).getDim();
  const unsigned int batch_size = output_dim.batch();
  const unsigned int unit = output_dim.width();

  Tensor &hidden_state = context.getTensor(wt_idx[1]);
  hidden_state.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_hidden_state = hidden_state.getBatchSlice(timestep, 1);
  next_hidden_state.reshape({batch_size, 1, 1, unit});
  Tensor next_hidden_state_derivative;
  if (training) {
    Tensor &hidden_state_derivative = context.getTensorGrad(wt_idx[1]);
    hidden_state_derivative.reshape({max_timestep, 1, batch_size, unit});
    next_hidden_state_derivative =
      hidden_state_derivative.getBatchSlice(timestep, 1);
    next_hidden_state_derivative.reshape({batch_size, 1, 1, unit});
  }

  Tensor &cell_state = context.getTensor(wt_idx[2]);
  cell_state.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_cell_state = cell_state.getBatchSlice(timestep, 1);
  next_cell_state.reshape({batch_size, 1, 1, unit});
  Tensor next_cell_state_derivative;
  if (training) {
    Tensor &cell_state_derivative = context.getTensorGrad(wt_idx[2]);
    cell_state_derivative.reshape({max_timestep, 1, batch_size, unit});
    next_cell_state_derivative =
      cell_state_derivative.getBatchSlice(timestep, 1);
    next_cell_state_derivative.reshape({batch_size, 1, 1, unit});
  }

  outputs[0] = Var_Grad(next_hidden_state, next_hidden_state_derivative,
                        context.getTensorName(wt_idx[1]));
  outputs[1] = Var_Grad(next_cell_state, next_cell_state_derivative,
                        context.getTensorName(wt_idx[2]));
}

const std::vector<Var_Grad *> getOutputs(std::vector<Var_Grad> &outputs) {
  std::vector<Var_Grad *> ret(outputs.size());
  for (unsigned int i = 0; i < outputs.size(); ++i) {
    ret[i] = &outputs[i];
  }
  return ret;
}

void fillTensors(std::vector<Var_Grad> &tensors, RunLayerContext &context,
                 bool training, const std::vector<unsigned int> &wt_idx,
                 const unsigned int max_timestep, const unsigned int timestep) {
  tensors.resize(1);

#if ENABLE_SHARING_WT_IDX
  const Tensor &ifgo = context.getTensor(wt_idx[0]);
  Tensor empty;
  const Tensor &ifgo_derivative =
    training ? context.getTensorGrad(wt_idx[0]) : empty;
  tensors[0] =
    Var_Grad(ifgo, ifgo_derivative, context.getTensorName(wt_idx[0]));
#else
  const TensorDim &output_dim = context.getOutput(0).getDim();
  const unsigned int batch_size = output_dim.batch();
  const unsigned int unit = output_dim.width();

  Tensor &ifgo = context.getTensor(wt_idx[0]);
  const unsigned int NUM_GATE = ifgo.width() / unit;
  ifgo.reshape({max_timestep, 1, batch_size, NUM_GATE * unit});
  Tensor ifgo_t = ifgo.getBatchSlice(timestep, 1);
  ifgo_t.reshape({batch_size, 1, 1, NUM_GATE * unit});
  Tensor ifgo_derivative_t;
  if (training) {
    Tensor &ifgo_derivative = context.getTensorGrad(wt_idx[0]);
    ifgo_derivative.reshape({max_timestep, 1, batch_size, NUM_GATE * unit});
    ifgo_derivative_t = ifgo_derivative.getBatchSlice(timestep, 1);
    ifgo_derivative_t.reshape({batch_size, 1, 1, NUM_GATE * unit});
  }
  tensors[0] =
    Var_Grad(ifgo_t, ifgo_derivative_t, context.getTensorName(wt_idx[0]));
#endif
}

const std::vector<Var_Grad *> getTensors(std::vector<Var_Grad> &tensors) {
  std::vector<Var_Grad *> ret(tensors.size());
  for (unsigned int i = 0; i < tensors.size(); ++i) {
    ret[i] = &tensors[i];
  }
  return ret;
}

} // namespace init_lstm_context

enum INDEX {
  INPUT = 0,
  HIDDEN_STATE_IN = 1,
  CELL_STATE_IN = 2,
  HIDDEN_STATE_OUT = 0,
  CELL_STATE_OUT = 1
};

enum LSTMCellCoreParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  ifgo,
};

LSTMCellCoreLayer::LSTMCellCoreLayer() :
  LayerImpl(),
  lstmcell_core_props(
    props::Unit(), props::HiddenStateActivation() = ActivationType::ACT_TANH,
    props::RecurrentActivation() = ActivationType::ACT_SIGMOID,
    props::IntegrateBias()),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void LSTMCellCoreLayer::finalize(InitLayerContext &context) {
#if ENBABLE_SHARING_WEIGHT
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
#endif

  NNTR_THROW_IF(std::get<props::Unit>(lstmcell_core_props).empty(),
                std::invalid_argument)
    << "unit property missing for lstmcell_core layer";
  const unsigned int unit = std::get<props::Unit>(lstmcell_core_props).get();
  const ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(lstmcell_core_props).get();
  const ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(lstmcell_core_props).get();
#if ENBABLE_SHARING_WEIGHT
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcell_core_props).get();
#endif

  if (context.getNumInputs() != 3)
    throw std::invalid_argument("LSTMCellCore layer should takes 3 input");

  // input_dim = [ batch, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[0];
  if (input_dim.height() != 1 || input_dim.channel() != 1)
    throw std::invalid_argument(
      "Input must be single time dimension for LSTMCellCore");
  // input_hidden_state_dim = [ batch, 1, 1, unit ]
  const TensorDim &input_hidden_state_dim = context.getInputDimensions()[1];
  if (input_hidden_state_dim.channel() != 1 ||
      input_hidden_state_dim.height() != 1)
    throw std::invalid_argument("Input hidden state's dimension should be "
                                "[batch, 1, 1, unit] for LSTMCellCore");
  // input_cell_state_dim = [ batch, 1, 1, unit ]
  const TensorDim &input_cell_state_dim = context.getInputDimensions()[2];
  if (input_cell_state_dim.channel() != 1 || input_cell_state_dim.height() != 1)
    throw std::invalid_argument("Input cell state's dimension should be "
                                "[batch, 1, 1, unit] for LSTMCellCore");

  const unsigned int batch_size = input_dim.batch();
#if ENABLE_SHARING_WT_IDX
  const unsigned int feature_size = input_dim.width();
#endif

  const TensorDim output_dim(batch_size, 1, 1, unit);
  const TensorDim output_hidden_state_dim = input_hidden_state_dim;
  const TensorDim output_cell_state_dim = input_cell_state_dim;

  context.setOutputDimensions(
    {output_dim, output_hidden_state_dim, output_cell_state_dim});

#if ENABLE_SHARING_WT_IDX
  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // - weight_ih ( input to hidden )
  //  : [1, 1, feature_size, NUM_GATE x unit] -> i, f, g, o
  TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[LSTMCellCoreParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // - weight_hh ( hidden to hidden )
  //  : [1, 1, unit, NUM_GATE x unit] -> i, f, g, o
  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[LSTMCellCoreParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // - bias_h ( input bias, hidden bias are integrate to 1 bias )
      //  : [1, 1, 1, NUM_GATE x unit] -> i, f, g, o
      TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[LSTMCellCoreParams::bias_h] =
        context.requestWeight(bias_h_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_h", true);
    } else {
      // - bias_ih ( input bias )
      //  : [1, 1, 1, NUM_GATE x unit] -> i, f, g, o
      TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[LSTMCellCoreParams::bias_ih] =
        context.requestWeight(bias_ih_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_ih", true);
      // - bias_hh ( hidden bias )
      //  : [1, 1, 1, NUM_GATE x unit] -> i, f, g, o
      TensorDim bias_hh_dim({NUM_GATE * unit});
      wt_idx[LSTMCellCoreParams::bias_hh] =
        context.requestWeight(bias_hh_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_hh", true);
    }
  }
#endif

#if ENABLE_SHARING_WT_IDX
  TensorDim ifgo_dim(batch_size, 1, 1, NUM_GATE * unit);
  wt_idx[LSTMCellCoreParams::ifgo] =
    context.requestTensor(ifgo_dim, "ifgo", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);
#endif
  acti_func.setActiFunc(hidden_state_activation_type);
  recurrent_acti_func.setActiFunc(recurrent_activation_type);
}

void LSTMCellCoreLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> remain_props =
    loadProperties(values, lstmcell_core_props);
  LayerImpl::setProperty(remain_props);
}

void LSTMCellCoreLayer::exportTo(Exporter &exporter,
                                 const ExportMethods &method) const {
#if ENABLE_SHARING_WT_IDX
  LayerImpl::exportTo(exporter, method);
#endif
  exporter.saveResult(lstmcell_core_props, method, this);
}

void LSTMCellCoreLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstmcell_core_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcell_core_props).get();

  const Tensor &input = context.getInput(INDEX::INPUT);
  const unsigned int batch_size = input.getDim().batch();

  const Tensor &prev_hidden_state = context.getInput(INDEX::HIDDEN_STATE_IN);
  const Tensor &prev_cell_state = context.getInput(INDEX::CELL_STATE_IN);
  Tensor &next_hidden_state = context.getOutput(INDEX::HIDDEN_STATE_OUT);
  Tensor &next_cell_state = context.getOutput(INDEX::CELL_STATE_OUT);

#if ENABLE_SHARING_WT_IDX
  const Tensor &weight_ih =
    context.getWeight(wt_idx[LSTMCellCoreParams::weight_ih]);
  const Tensor &weight_hh =
    context.getWeight(wt_idx[LSTMCellCoreParams::weight_hh]);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[LSTMCellCoreParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[LSTMCellCoreParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[LSTMCellCoreParams::bias_hh])
                      : empty;
#else
  const Tensor &weight_ih = context.getWeight(LSTMCellCoreParams::weight_ih);
  const Tensor &weight_hh = context.getWeight(LSTMCellCoreParams::weight_hh);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(LSTMCellCoreParams::bias_h)
                     : empty;
  // subtract index by 1 cause there is no bias_h
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(LSTMCellCoreParams::bias_ih - 1)
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(LSTMCellCoreParams::bias_hh - 1)
                      : empty;
#endif

#if ENABLE_SHARING_WT_IDX
  Tensor &ifgo = context.getTensor(wt_idx[LSTMCellCoreParams::ifgo]);
#else
  Tensor &ifgo = context.getTensor(0);
#endif

  input.dot(weight_ih, ifgo);
  prev_hidden_state.dot(weight_hh, ifgo, false, false, 1.0);
  if (!disable_bias) {
    if (integrate_bias) {
      ifgo.add_i(bias_h);
    } else {
      ifgo.add_i(bias_ih);
      ifgo.add_i(bias_hh);
    }
  }

  Tensor input_forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor input_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor memory_cell =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 2, false);
  Tensor output_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 3, false);

  recurrent_acti_func.run_fn(input_forget_gate, input_forget_gate);
  recurrent_acti_func.run_fn(output_gate, output_gate);
  acti_func.run_fn(memory_cell, memory_cell);

  forget_gate.multiply_strided(prev_cell_state, next_cell_state);
  memory_cell.multiply_strided(input_gate, next_cell_state, 1.0f);

  acti_func.run_fn(next_cell_state, next_hidden_state);
  next_hidden_state.multiply_i_strided(output_gate);
}

void LSTMCellCoreLayer::calcDerivative(RunLayerContext &context) {
#if ENABLE_SHARING_WT_IDX
  Tensor &ifgo_derivative =
    context.getTensorGrad(wt_idx[LSTMCellCoreParams::ifgo]);
  const Tensor &weight_ih =
    context.getWeight(wt_idx[LSTMCellCoreParams::weight_ih]);
#else
  const Tensor &weight_ih = context.getWeight(LSTMCellCoreParams::weight_ih);
  Tensor &ifgo_derivative = context.getTensorGrad(0);
#endif
  Tensor &outgoing_derivative = context.getOutgoingDerivative(INDEX::INPUT);

  ifgo_derivative.dot(weight_ih, outgoing_derivative, false, true);
}

void LSTMCellCoreLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstmcell_core_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcell_core_props).get();

  const Tensor &input = context.getInput(INDEX::INPUT);
  const unsigned int batch_size = input.getDim().batch();

#if ENABLE_SHARING_WT_IDX
  Tensor &djdweight_ih =
    context.getWeightGrad(wt_idx[LSTMCellCoreParams::weight_ih]);
  const Tensor &weight_hh =
    context.getWeight(wt_idx[LSTMCellCoreParams::weight_hh]);
  Tensor &djdweight_hh =
    context.getWeightGrad(wt_idx[LSTMCellCoreParams::weight_hh]);
  Tensor empty;
  Tensor &djdbias_h =
    !disable_bias && integrate_bias
      ? context.getWeightGrad(wt_idx[LSTMCellCoreParams::bias_h])
      : empty;
  Tensor &djdbias_ih =
    !disable_bias && !integrate_bias
      ? context.getWeightGrad(wt_idx[LSTMCellCoreParams::bias_ih])
      : empty;
  Tensor &djdbias_hh =
    !disable_bias && !integrate_bias
      ? context.getWeightGrad(wt_idx[LSTMCellCoreParams::bias_hh])
      : empty;
#else
  Tensor &djdweight_ih = context.getWeightGrad(LSTMCellCoreParams::weight_ih);
  const Tensor &weight_hh = context.getWeight(LSTMCellCoreParams::weight_hh);
  Tensor &djdweight_hh = context.getWeightGrad(LSTMCellCoreParams::weight_hh);
  Tensor empty;
  Tensor &djdbias_h = !disable_bias && integrate_bias
                        ? context.getWeightGrad(LSTMCellCoreParams::bias_h)
                        : empty;
  // subtract index by 1 cause there is no bias_h(and also djdbias_h)
  Tensor &djdbias_ih =
    !disable_bias && !integrate_bias
      ? context.getWeightGrad(LSTMCellCoreParams::bias_ih - 1)
      : empty;
  Tensor &djdbias_hh =
    !disable_bias && !integrate_bias
      ? context.getWeightGrad(LSTMCellCoreParams::bias_hh - 1)
      : empty;
#endif

  const Tensor &prev_hidden_state = context.getInput(INDEX::HIDDEN_STATE_IN);
  Tensor &prev_hidden_state_derivative =
    context.getOutgoingDerivative(INDEX::HIDDEN_STATE_IN);
  Tensor &next_hidden_state_derivative =
    context.getIncomingDerivative(INDEX::HIDDEN_STATE_OUT);

  Tensor &prev_cell_state = context.getInput(INDEX::CELL_STATE_IN);
  Tensor &prev_cell_state_derivative =
    context.getOutgoingDerivative(INDEX::CELL_STATE_IN);
  Tensor &next_cell_state = context.getOutput(INDEX::CELL_STATE_OUT);
  Tensor &next_cell_state_derivative =
    context.getIncomingDerivative(INDEX::CELL_STATE_OUT);

#if ENABLE_SHARING_WT_IDX
  Tensor &ifgo = context.getTensor(wt_idx[LSTMCellCoreParams::ifgo]);
  Tensor &ifgo_derivative =
    context.getTensorGrad(wt_idx[LSTMCellCoreParams::ifgo]);
#else
  Tensor &ifgo = context.getTensor(0);
  Tensor &ifgo_derivative = context.getTensorGrad(0);
#endif

  Tensor input_forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor input_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor memory_cell =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 2, false);
  Tensor output_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 3, false);

  Tensor input_forget_gate_derivative =
    ifgo_derivative.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor input_gate_derivative =
    ifgo_derivative.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor forget_gate_derivative =
    ifgo_derivative.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor memory_cell_derivative = ifgo_derivative.getSharedDataTensor(
    {batch_size, 1, 1, unit}, unit * 2, false);
  Tensor output_gate_derivative = ifgo_derivative.getSharedDataTensor(
    {batch_size, 1, 1, unit}, unit * 3, false);

  acti_func.run_fn(next_cell_state, next_cell_state);
  next_hidden_state_derivative.multiply_strided(next_cell_state,
                                                output_gate_derivative);

  acti_func.run_prime_fn(next_cell_state, prev_cell_state_derivative,
                         next_hidden_state_derivative);
  prev_cell_state_derivative.multiply_i_strided(output_gate);
  prev_cell_state_derivative.add_i(next_cell_state_derivative);

  prev_cell_state_derivative.multiply_strided(input_gate,
                                              memory_cell_derivative);
  prev_cell_state_derivative.multiply_strided(memory_cell,
                                              input_gate_derivative);

  prev_cell_state_derivative.multiply_strided(prev_cell_state,
                                              forget_gate_derivative);
  prev_cell_state_derivative.multiply_i_strided(forget_gate);

  recurrent_acti_func.run_prime_fn(output_gate, output_gate_derivative,
                                   output_gate_derivative);
  recurrent_acti_func.run_prime_fn(input_forget_gate,
                                   input_forget_gate_derivative,
                                   input_forget_gate_derivative);
  acti_func.run_prime_fn(memory_cell, memory_cell_derivative,
                         memory_cell_derivative);

  if (!disable_bias) {
    if (integrate_bias) {
      ifgo_derivative.sum(0, djdbias_h, 1.0f, 1.0f);
    } else {
      ifgo_derivative.sum(0, djdbias_ih, 1.0f, 1.0f);
      ifgo_derivative.sum(0, djdbias_hh, 1.0f, 1.0f);
    }
  }
  input.dot(ifgo_derivative, djdweight_ih, true, false, 1.0f);
  prev_hidden_state.dot(ifgo_derivative, djdweight_hh, true, false, 1.0f);
  ifgo_derivative.dot(weight_hh, prev_hidden_state_derivative, false, true);
}

void LSTMCellCoreLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[LSTMCellCoreParams::ifgo], batch);
}

void lstmcell_forwarding(const unsigned int unit, const unsigned int batch_size,
                         const bool disable_bias, const bool integrate_bias,
                         ActiFunc &acti_func, ActiFunc &recurrent_acti_func,
                         const Tensor &input, const Tensor &prev_hidden_state,
                         const Tensor &prev_cell_state, Tensor &hidden_state,
                         Tensor &cell_state, const Tensor &weight_ih,
                         const Tensor &weight_hh, const Tensor &bias_h,
                         const Tensor &bias_ih, const Tensor &bias_hh,
                         Tensor &ifgo) {
  input.dot(weight_ih, ifgo);
  prev_hidden_state.dot(weight_hh, ifgo, false, false, 1.0);
  if (!disable_bias) {
    if (integrate_bias) {
      ifgo.add_i(bias_h);
    } else {
      ifgo.add_i(bias_ih);
      ifgo.add_i(bias_hh);
    }
  }

  Tensor input_forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor input_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor memory_cell =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 2, false);
  Tensor output_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 3, false);

  recurrent_acti_func.run_fn(input_forget_gate, input_forget_gate);
  recurrent_acti_func.run_fn(output_gate, output_gate);
  acti_func.run_fn(memory_cell, memory_cell);

  prev_cell_state.multiply_strided(forget_gate, cell_state);
  memory_cell.multiply_strided(input_gate, cell_state, 1.0f);

  acti_func.run_fn(cell_state, hidden_state);
  hidden_state.multiply_i_strided(output_gate);
}

void lstmcell_calcDerivative(const Tensor &d_ifgo, const Tensor &weight_ih,
                             Tensor &outgoing_derivative) {
  d_ifgo.dot(weight_ih, outgoing_derivative, false, true);
}

void lstmcell_calcGradient(
  const unsigned int unit, const unsigned int batch_size,
  const bool disable_bias, const bool integrate_bias, ActiFunc &acti_func,
  ActiFunc &recurrent_acti_func, const Tensor &input,
  const Tensor &prev_hidden_state, Tensor &d_prev_hidden_state,
  const Tensor &prev_cell_state, Tensor &d_prev_cell_state,
  Tensor &d_hidden_state, const Tensor &cell_state, Tensor &d_cell_state,
  Tensor &d_weight_ih, const Tensor &weight_hh, Tensor &d_weight_hh,
  Tensor &d_bias_h, Tensor &d_bias_ih, Tensor &d_bias_hh, const Tensor &ifgo,
  Tensor &d_ifgo) {
  Tensor input_forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor input_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor memory_cell =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 2, false);
  Tensor output_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 3, false);

  Tensor d_input_forget_gate =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor d_input_gate =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor d_forget_gate =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor d_memory_cell =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 2, false);
  Tensor d_output_gate =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 3, false);

  Tensor activated_cell_state;
  acti_func.run_fn(cell_state, activated_cell_state);
  d_hidden_state.multiply_strided(activated_cell_state, d_output_gate);
  acti_func.run_prime_fn(activated_cell_state, d_prev_cell_state,
                         d_hidden_state);
  d_prev_cell_state.multiply_i_strided(output_gate);
  d_prev_cell_state.add_i(d_cell_state);

  d_prev_cell_state.multiply_strided(input_gate, d_memory_cell);
  d_prev_cell_state.multiply_strided(memory_cell, d_input_gate);

  d_prev_cell_state.multiply_strided(prev_cell_state, d_forget_gate);
  d_prev_cell_state.multiply_i_strided(forget_gate);

  recurrent_acti_func.run_prime_fn(output_gate, d_output_gate, d_output_gate);
  recurrent_acti_func.run_prime_fn(input_forget_gate, d_input_forget_gate,
                                   d_input_forget_gate);
  acti_func.run_prime_fn(memory_cell, d_memory_cell, d_memory_cell);

  if (!disable_bias) {
    if (integrate_bias) {
      d_ifgo.sum(0, d_bias_h, 1.0f, 1.0f);
    } else {
      d_ifgo.sum(0, d_bias_ih, 1.0f, 1.0f);
      d_ifgo.sum(0, d_bias_hh, 1.0f, 1.0f);
    }
  }

  input.dot(d_ifgo, d_weight_ih, true, false, 1.0f);
  prev_hidden_state.dot(d_ifgo, d_weight_hh, true, false, 1.0f);
  d_ifgo.dot(weight_hh, d_prev_hidden_state, false, true);
}

} // namespace nntrainer
