// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lstmcell.cpp
 * @date   17 March 2021
 * @brief  This is LSTMCell Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/pdf/1606.01305.pdf
 *         https://github.com/teganmaharaj/zoneout
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <lstmcell.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LSTMCellParams {
  weight_ih,
  weight_hh,
  bias_ih,
  hidden_state,
  cell_state,
  ifgo,
  dropout_mask
};

const std::vector<unsigned int>
getInOutIdx(std::array<unsigned int, 7> &wt_idx) {
  std::vector<unsigned int> ret(3);
  ret[0] = SINGLE_INOUT_IDX;
  ret[1] = wt_idx[LSTMCellParams::hidden_state];
  ret[2] = wt_idx[LSTMCellParams::cell_state];
  return ret;
}

const std::vector<unsigned int>
getTensorIdx(std::array<unsigned int, 7> &wt_idx) {
  std::vector<unsigned int> ret(1);
  ret[0] = wt_idx[LSTMCellParams::ifgo];
  return ret;
}

LSTMCellLayer::LSTMCellLayer() :
  LayerImpl(),
  lstmcell_props(props::Unit(), props::DropOutRate(), props::MaxTimestep(),
                 props::Timestep()),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void LSTMCellLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(std::get<props::Unit>(lstmcell_props).empty(),
                std::invalid_argument)
    << "unit property missing for lstmcell layer";
  const unsigned int unit = std::get<props::Unit>(lstmcell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props);

#if !ENABLE_SHARING_WT_IDX
  const Tensor::Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  const Tensor::Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props);
  const nntrainer::WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
#endif

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
  if (input_dim.height() != 1 || input_dim.channel() != 1)
    throw std::invalid_argument(
      "Input must be single time dimension for LSTMCell (shape should be "
      "[batch_size, 1, 1, feature_size])");
  const unsigned int batch_size = input_dim.batch();
#if !ENABLE_SHARING_WT_IDX
  const unsigned int feature_size = input_dim.width();
#endif

  // output_dim = [ batch_size, 1, 1, unit ]
  const TensorDim output_dim(batch_size, 1, 1, unit);
  context.setOutputDimensions({output_dim});

#if !ENABLE_SHARING_WT_IDX
  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // - weight_ih ( input to hidden )
  //  : [1, 1, feature_size, NUM_GATE x unit] -> i, f, g, o
  TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[LSTMCellParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // - weight_hh ( hidden to hidden )
  //  : [1, 1, unit, NUM_GATE x unit] -> i, f, g, o
  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[LSTMCellParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  // - bias_ih ( input bias )
  //  : [1, 1, 1, NUM_GATE x unit] -> i, f, g, o
  TensorDim bias_ih_dim({NUM_GATE * unit});
  wt_idx[LSTMCellParams::bias_ih] =
    context.requestWeight(bias_ih_dim, bias_initializer,
                          WeightRegularizer::NONE, 1.0f, "bias_ih", true);
#endif

  // dropout_mask_dim = [ max_timestep * batch_size, 1, 1, unit ]
  const TensorDim dropout_mask_dim(max_timestep * batch_size, 1, 1, unit);
  if (dropout_rate > epsilon) {
    wt_idx[LSTMCellParams::dropout_mask] = context.requestTensor(
      dropout_mask_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
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
#if !ENABLE_SHARING_WT_IDX
  /**
   * TODO: make this independent of time dimension once recurrent realizer
   * supports requesting tensors which are not always shared
   */
  /** ifgo_dim = [ max_timestep * batch_size, 1, 1, NUM_GATE * unit ] */
  const TensorDim ifgo_dim(max_timestep * batch_size, 1, 1, NUM_GATE * unit);
  wt_idx[LSTMCellParams::ifgo] =
    context.requestTensor(ifgo_dim, "ifgo", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);
#endif

  TensorDim hidden_state_t_dim({batch_size, 1, 1, unit});
  TensorDim cell_state_t_dim({batch_size, 1, 1, unit});
  InitLayerContext core_context(
    {input_dim, hidden_state_t_dim, cell_state_t_dim}, 3,
    context.executeInPlace(), context.getName());
  lstmcellcorelayer.finalize(core_context);
  init_lstm_context::fillLayerInitContext(context, core_context);
}

void LSTMCellLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> remain_props =
    loadProperties(values, lstmcell_props);

  // Note: In current implementation the lstmcellcorelayer also has
  // a properties related to weight. But it is not exported or used anywhere.
  lstmcellcorelayer.setProperty(remain_props);
  if (!std::get<props::Unit>(lstmcell_props).empty()) {
    lstmcellcorelayer.setProperty(
      {"unit=" + to_string(std::get<props::Unit>(lstmcell_props))});
  }

#if !ENABLE_SHARING_WT_IDX
  // To remove lstmcell core layer's properties
  std::tuple<props::HiddenStateActivation, props::RecurrentActivation>
    lstmcell_core_props;
  std::vector<std::string> impl_props =
    loadProperties(remain_props, lstmcell_core_props);

  LayerImpl::setProperty(impl_props);
#endif
}

void LSTMCellLayer::exportTo(Exporter &exporter,
                             const ExportMethods &method) const {
#if !ENABLE_SHARING_WT_IDX
  LayerImpl::exportTo(exporter, method);
#endif
  exporter.saveResult(
    std::forward_as_tuple(std::get<props::DropOutRate>(lstmcell_props),
                          std::get<props::MaxTimestep>(lstmcell_props),
                          std::get<props::Timestep>(lstmcell_props)),
    method, this);
  lstmcellcorelayer.exportTo(exporter, method);
}

void LSTMCellLayer::forwarding(RunLayerContext &context, bool training) {
  const unsigned int unit = std::get<props::Unit>(lstmcell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props);
  const unsigned int timestep = std::get<props::Timestep>(lstmcell_props);

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();

  Tensor &hidden_state =
    context.getTensor(wt_idx[LSTMCellParams::hidden_state]);
  hidden_state.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_hidden_state = hidden_state.getBatchSlice(timestep, 1);
  next_hidden_state.reshape({batch_size, 1, 1, unit});

  Tensor &cell_state = context.getTensor(wt_idx[LSTMCellParams::cell_state]);

  if (!timestep) {
    hidden_state.setZero();
    cell_state.setZero();
  }

  init_lstm_context::fillWeights(weights, context, training, max_timestep,
                                 timestep);
  init_lstm_context::fillInputs(inputs, context, training, getInOutIdx(wt_idx),
                                max_timestep, timestep);
  init_lstm_context::fillOutputs(outputs, context, training,
                                 getInOutIdx(wt_idx), max_timestep, timestep);
  init_lstm_context::fillTensors(tensors, context, training,
                                 getTensorIdx(wt_idx), max_timestep, timestep);
  RunLayerContext core_context(context.getName(), context.getTrainable(),
                               context.getLoss(), context.executeInPlace(),
                               init_lstm_context::getWeights(weights),
                               init_lstm_context::getInputs(inputs),
                               init_lstm_context::getOutputs(outputs),
                               init_lstm_context::getTensors(tensors));
  lstmcellcorelayer.forwarding(core_context, training);

  if (dropout_rate > epsilon && training) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[LSTMCellParams::dropout_mask]);
    dropout_mask.reshape({max_timestep, 1, batch_size, unit});
    Tensor dropout_mask_t = dropout_mask.getBatchSlice(timestep, 1);
    dropout_mask_t.dropout_mask(dropout_rate);
    next_hidden_state.multiply_i(dropout_mask_t);
  }

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  output.copyData(next_hidden_state);
}

void LSTMCellLayer::calcDerivative(RunLayerContext &context) {
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props);
  const unsigned int timestep = std::get<props::Timestep>(lstmcell_props);

  init_lstm_context::fillWeights(weights, context, true, max_timestep,
                                 timestep);
  init_lstm_context::fillInputs(inputs, context, true, getInOutIdx(wt_idx),
                                max_timestep, timestep);
  init_lstm_context::fillOutputs(outputs, context, true, getInOutIdx(wt_idx),
                                 max_timestep, timestep);
  init_lstm_context::fillTensors(tensors, context, true, getTensorIdx(wt_idx),
                                 max_timestep, timestep);
  RunLayerContext core_context(context.getName(), context.getTrainable(),
                               context.getLoss(), context.executeInPlace(),
                               init_lstm_context::getWeights(weights),
                               init_lstm_context::getInputs(inputs),
                               init_lstm_context::getOutputs(outputs),
                               init_lstm_context::getTensors(tensors));
  lstmcellcorelayer.calcDerivative(core_context);
}

void LSTMCellLayer::calcGradient(RunLayerContext &context) {
  const unsigned int unit = std::get<props::Unit>(lstmcell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props);
  const unsigned int timestep = std::get<props::Timestep>(lstmcell_props);

  unsigned int batch_size = context.getInput(SINGLE_INOUT_IDX).getDim().batch();

  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &hidden_state_derivative =
    context.getTensorGrad(wt_idx[LSTMCellParams::hidden_state]);
  hidden_state_derivative.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_hidden_state_derivative =
    hidden_state_derivative.getBatchSlice(timestep, 1);
  next_hidden_state_derivative.reshape({batch_size, 1, 1, unit});

  Tensor &cell_state_derivative =
    context.getTensorGrad(wt_idx[LSTMCellParams::cell_state]);
  cell_state_derivative.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_cell_state_derivative =
    cell_state_derivative.getBatchSlice(timestep, 1);
  next_cell_state_derivative.reshape({batch_size, 1, 1, unit});

  if (timestep + 1 == max_timestep) {
    Tensor &djdweight_ih =
      context.getWeightGrad(wt_idx[LSTMCellParams::weight_ih]);
    Tensor &djdweight_hh =
      context.getWeightGrad(wt_idx[LSTMCellParams::weight_hh]);
    Tensor &djdbias_ih = context.getWeightGrad(wt_idx[LSTMCellParams::bias_ih]);
    djdweight_ih.setZero();
    djdweight_hh.setZero();
    djdbias_ih.setZero();

    next_hidden_state_derivative.setZero();
    next_cell_state_derivative.setZero();
  }

  if (dropout_rate > epsilon) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[LSTMCellParams::dropout_mask]);
    dropout_mask.reshape({max_timestep, 1, batch_size, unit});
    Tensor dropout_mask_t = dropout_mask.getBatchSlice(timestep, 1);
    next_hidden_state_derivative.multiply_i(dropout_mask_t);
  }

  next_hidden_state_derivative.add_i(incoming_derivative);

  init_lstm_context::fillWeights(weights, context, true, max_timestep,
                                 timestep);
  init_lstm_context::fillInputs(inputs, context, true, getInOutIdx(wt_idx),
                                max_timestep, timestep);
  init_lstm_context::fillOutputs(outputs, context, true, getInOutIdx(wt_idx),
                                 max_timestep, timestep);
  init_lstm_context::fillTensors(tensors, context, true, getTensorIdx(wt_idx),
                                 max_timestep, timestep);
  RunLayerContext core_context(context.getName(), context.getTrainable(),
                               context.getLoss(), context.executeInPlace(),
                               init_lstm_context::getWeights(weights),
                               init_lstm_context::getInputs(inputs),
                               init_lstm_context::getOutputs(outputs),
                               init_lstm_context::getTensors(tensors));
  lstmcellcorelayer.calcGradient(core_context);
}

void LSTMCellLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstmcell_props);
  context.updateTensor(wt_idx[LSTMCellParams::hidden_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[LSTMCellParams::cell_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[LSTMCellParams::ifgo], max_timestep * batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(wt_idx[LSTMCellParams::dropout_mask],
                         max_timestep * batch);
  }
}

} // namespace nntrainer
