// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   zoneout_lstmcell.cpp
 * @date   30 November 2021
 * @brief  This is ZoneoutLSTMCell Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
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
  bias_ih,
  hidden_state,
  cell_state,
  ifgo,
  hidden_state_zoneout_mask,
  cell_state_zoneout_mask,
};

unsigned int hidden_state_origin_idx = 0, cell_state_origin_idx = 0;

const std::vector<unsigned int>
getInputIdx(std::array<unsigned int, 8> &wt_idx) {
  std::vector<unsigned int> ret(3);
  ret[0] = SINGLE_INOUT_IDX;
  ret[1] = wt_idx[ZoneoutLSTMParams::hidden_state];
  ret[2] = wt_idx[ZoneoutLSTMParams::cell_state];
  return ret;
}

const std::vector<unsigned int>
getOutputIdx(std::array<unsigned int, 8> &wt_idx) {
  std::vector<unsigned int> ret(3);
  ret[0] = SINGLE_INOUT_IDX;
  ret[1] = hidden_state_origin_idx;
  ret[2] = cell_state_origin_idx;
  return ret;
}

const std::vector<unsigned int>
getTensorIdx(std::array<unsigned int, 8> &wt_idx) {
  std::vector<unsigned int> ret(1);
  ret[0] = wt_idx[ZoneoutLSTMParams::ifgo];
  return ret;
}

ZoneoutLSTMCellLayer::ZoneoutLSTMCellLayer() :
  LayerImpl(),
  zoneout_lstmcell_props(props::Unit(), HiddenStateZoneOutRate(),
                         CellStateZoneOutRate(), Test(), props::MaxTimestep(),
                         props::Timestep()),
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
  NNTR_THROW_IF(std::get<props::Unit>(zoneout_lstmcell_props).empty(),
                std::invalid_argument)
    << "unit property missing for zoneout_lstmcell layer";
  const unsigned int unit = std::get<props::Unit>(zoneout_lstmcell_props).get();
  const bool test = std::get<Test>(zoneout_lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props);

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
    throw std::invalid_argument("ZoneoutLSTMCellLayer takes only one input");
  if (std::get<props::MaxTimestep>(zoneout_lstmcell_props).empty())
    throw std::invalid_argument("Number of unroll steps(max timestep) must be "
                                "provided to zoneout LSTM cells");
  if (std::get<props::Timestep>(zoneout_lstmcell_props).empty())
    throw std::invalid_argument(
      "Current timestep must be provided to zoneout LSTM cell");

  // input_dim = [ batch_size, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[0];
  if (input_dim.height() != 1 || input_dim.channel() != 1)
    throw std::invalid_argument("Input must be single time dimension for "
                                "ZoneoutLSTMCell (shape should be "
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
  wt_idx[ZoneoutLSTMParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // - weight_hh ( hidden to hidden )
  //  : [1, 1, unit, NUM_GATE x unit] -> i, f, g, o
  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[ZoneoutLSTMParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  // - bias_ih ( input bias )
  //  : [1, 1, 1, NUM_GATE x unit] -> i, f, g, o
  TensorDim bias_ih_dim({NUM_GATE * unit});
  wt_idx[ZoneoutLSTMParams::bias_ih] =
    context.requestWeight(bias_ih_dim, bias_initializer,
                          WeightRegularizer::NONE, 1.0f, "bias_ih", true);
#endif

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

  hidden_state_origin_idx = context.requestTensor(
    hidden_state_dim, "hidden_state_origin", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);
  cell_state_origin_idx = context.requestTensor(
    cell_state_dim, "cell_state_origin", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);

#if !ENABLE_SHARING_WT_IDX
  /**
   * TODO: make this independent of time dimension once recurrent realizer
   * supports requesting tensors which are not always shared
   */
  /** ifgo_dim = [ max_timestep * batch_size, 1, 1, NUM_GATE * unit ] */
  const TensorDim ifgo_dim(max_timestep * batch_size, 1, 1, NUM_GATE * unit);
  wt_idx[ZoneoutLSTMParams::ifgo] =
    context.requestTensor(ifgo_dim, "ifgo", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN, false);
#endif

  TensorDim hidden_state_t_dim({batch_size, 1, 1, unit});
  TensorDim cell_state_t_dim({batch_size, 1, 1, unit});
  InitLayerContext core_context(
    {input_dim, hidden_state_t_dim, cell_state_t_dim}, 3,
    context.executeInPlace(), context.getName());
  lstmcellcorelayer.finalize(core_context);
  init_lstm_context::fillLayerInitContext(context, core_context);
}

void ZoneoutLSTMCellLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> remain_props =
    loadProperties(values, zoneout_lstmcell_props);

  // Note: In current implementation the lstmcellcorelayer also has
  // a properties related to weight. But it is not exported or used anywhere.
  lstmcellcorelayer.setProperty(remain_props);
  if (!std::get<props::Unit>(zoneout_lstmcell_props).empty()) {
    lstmcellcorelayer.setProperty(
      {"unit=" + to_string(std::get<props::Unit>(zoneout_lstmcell_props))});
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

void ZoneoutLSTMCellLayer::exportTo(Exporter &exporter,
                                    const ExportMethods &method) const {
#if !ENABLE_SHARING_WT_IDX
  LayerImpl::exportTo(exporter, method);
#endif
  exporter.saveResult(
    std::forward_as_tuple(
      std::get<HiddenStateZoneOutRate>(zoneout_lstmcell_props),
      std::get<CellStateZoneOutRate>(zoneout_lstmcell_props),
      std::get<Test>(zoneout_lstmcell_props),
      std::get<props::MaxTimestep>(zoneout_lstmcell_props),
      std::get<props::Timestep>(zoneout_lstmcell_props)),
    method, this);
  lstmcellcorelayer.exportTo(exporter, method);
}

void ZoneoutLSTMCellLayer::forwarding(RunLayerContext &context, bool training) {
  const unsigned int unit = std::get<props::Unit>(zoneout_lstmcell_props).get();
  const float hidden_state_zoneout_rate =
    std::get<HiddenStateZoneOutRate>(zoneout_lstmcell_props);
  const float cell_state_zoneout_rate =
    std::get<CellStateZoneOutRate>(zoneout_lstmcell_props);
  const bool test = std::get<Test>(zoneout_lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props);
  const unsigned int timestep =
    std::get<props::Timestep>(zoneout_lstmcell_props);

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();

  Tensor &hidden_state =
    context.getTensor(wt_idx[ZoneoutLSTMParams::hidden_state]);
  hidden_state.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, 1, 1, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hidden_state.getBatchSlice(timestep - 1, 1);
    prev_hidden_state.reshape({batch_size, 1, 1, unit});
  }
  Tensor next_hidden_state = hidden_state.getBatchSlice(timestep, 1);
  next_hidden_state.reshape({batch_size, 1, 1, unit});

  Tensor &cell_state = context.getTensor(wt_idx[ZoneoutLSTMParams::cell_state]);
  cell_state.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_cell_state;
  if (!timestep) {
    prev_cell_state = Tensor(batch_size, 1, 1, unit);
    prev_cell_state.setZero();
  } else {
    prev_cell_state = cell_state.getBatchSlice(timestep - 1, 1);
    prev_cell_state.reshape({batch_size, 1, 1, unit});
  }
  Tensor next_cell_state = cell_state.getBatchSlice(timestep, 1);
  next_cell_state.reshape({batch_size, 1, 1, unit});

  if (!timestep) {
    hidden_state.setZero();
    cell_state.setZero();
  }

  init_lstm_context::fillWeights(weights, context, training, max_timestep,
                                 timestep, test);
  init_lstm_context::fillInputs(inputs, context, training, getInputIdx(wt_idx),
                                max_timestep, timestep);
  init_lstm_context::fillOutputs(outputs, context, training,
                                 getOutputIdx(wt_idx), max_timestep, timestep);
  init_lstm_context::fillTensors(tensors, context, training,
                                 getTensorIdx(wt_idx), max_timestep, timestep);
  RunLayerContext core_context(context.getName(), context.getTrainable(),
                               context.getLoss(), context.executeInPlace(),
                               init_lstm_context::getWeights(weights),
                               init_lstm_context::getInputs(inputs),
                               init_lstm_context::getOutputs(outputs),
                               init_lstm_context::getTensors(tensors));
  lstmcellcorelayer.forwarding(core_context, training);

  if (training) {
    Tensor &hidden_state_zoneout_mask =
      test ? context.getWeight(
               wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask])
           : context.getTensor(
               wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask]);
    hidden_state_zoneout_mask.reshape({max_timestep, 1, batch_size, unit});
    Tensor next_hidden_state_zoneout_mask =
      hidden_state_zoneout_mask.getBatchSlice(timestep, 1);
    next_hidden_state_zoneout_mask.reshape({batch_size, 1, 1, unit});
    Tensor prev_hidden_state_zoneout_mask;
    if (!test) {
      prev_hidden_state_zoneout_mask =
        next_hidden_state_zoneout_mask.zoneout_mask(hidden_state_zoneout_rate);
    } else {
      next_hidden_state_zoneout_mask.multiply(-1.0f,
                                              prev_hidden_state_zoneout_mask);
      prev_hidden_state_zoneout_mask.add_i(1.0f);
    }

    Tensor &hidden_state_origin = context.getTensor(hidden_state_origin_idx);
    hidden_state_origin.reshape({max_timestep, 1, batch_size, unit});
    Tensor next_hidden_state_origin =
      hidden_state_origin.getBatchSlice(timestep, 1);
    next_hidden_state_origin.reshape({batch_size, 1, 1, unit});

    next_hidden_state_origin.multiply(next_hidden_state_zoneout_mask,
                                      next_hidden_state);
    prev_hidden_state.multiply(prev_hidden_state_zoneout_mask,
                               next_hidden_state, 1.0f);
  }

  if (training) {
    Tensor &cell_state_zoneout_mask =
      test
        ? context.getWeight(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask])
        : context.getTensor(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask]);
    cell_state_zoneout_mask.reshape({max_timestep, 1, batch_size, unit});
    Tensor next_cell_state_zoneout_mask =
      cell_state_zoneout_mask.getBatchSlice(timestep, 1);
    next_cell_state_zoneout_mask.reshape({batch_size, 1, 1, unit});
    Tensor prev_cell_state_zoneout_mask;
    if (!test) {
      prev_cell_state_zoneout_mask =
        next_cell_state_zoneout_mask.zoneout_mask(cell_state_zoneout_rate);
    } else {
      next_cell_state_zoneout_mask.multiply(-1.0f,
                                            prev_cell_state_zoneout_mask);
      prev_cell_state_zoneout_mask.add_i(1.0f);
    }

    Tensor &cell_state_origin = context.getTensor(cell_state_origin_idx);
    cell_state_origin.reshape({max_timestep, 1, batch_size, unit});
    Tensor next_cell_state_origin =
      cell_state_origin.getBatchSlice(timestep, 1);
    next_cell_state_origin.reshape({batch_size, 1, 1, unit});

    next_cell_state_origin.multiply(next_cell_state_zoneout_mask,
                                    next_cell_state);
    prev_cell_state.multiply(prev_cell_state_zoneout_mask, next_cell_state,
                             1.0f);
  }
  // Todo: zoneout at inference

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  output.copyData(next_hidden_state);
}

void ZoneoutLSTMCellLayer::calcDerivative(RunLayerContext &context) {
  const bool test = std::get<Test>(zoneout_lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props);
  const unsigned int timestep =
    std::get<props::Timestep>(zoneout_lstmcell_props);

  init_lstm_context::fillWeights(weights, context, true, max_timestep, timestep,
                                 test);
  init_lstm_context::fillInputs(inputs, context, true, getInputIdx(wt_idx),
                                max_timestep, timestep);
  init_lstm_context::fillOutputs(outputs, context, true, getOutputIdx(wt_idx),
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

void ZoneoutLSTMCellLayer::calcGradient(RunLayerContext &context) {
  const unsigned int unit = std::get<props::Unit>(zoneout_lstmcell_props).get();
  const float hidden_state_zoneout_rate =
    std::get<HiddenStateZoneOutRate>(zoneout_lstmcell_props);
  const float cell_state_zoneout_rate =
    std::get<CellStateZoneOutRate>(zoneout_lstmcell_props);
  const bool test = std::get<Test>(zoneout_lstmcell_props);
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props);
  const unsigned int timestep =
    std::get<props::Timestep>(zoneout_lstmcell_props);

  unsigned int batch_size = context.getInput(SINGLE_INOUT_IDX).getDim().batch();

  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &hidden_state_derivative =
    context.getTensorGrad(wt_idx[ZoneoutLSTMParams::hidden_state]);
  hidden_state_derivative.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_hidden_state_derivative =
    hidden_state_derivative.getBatchSlice(timestep, 1);
  next_hidden_state_derivative.reshape({batch_size, 1, 1, unit});

  Tensor &cell_state_derivative =
    context.getTensorGrad(wt_idx[ZoneoutLSTMParams::cell_state]);
  cell_state_derivative.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_cell_state_derivative =
    cell_state_derivative.getBatchSlice(timestep, 1);
  next_cell_state_derivative.reshape({batch_size, 1, 1, unit});

  if (timestep + 1 == max_timestep) {
    Tensor &djdweight_ih =
      context.getWeightGrad(wt_idx[ZoneoutLSTMParams::weight_ih]);
    Tensor &djdweight_hh =
      context.getWeightGrad(wt_idx[ZoneoutLSTMParams::weight_hh]);
    Tensor &djdbias_ih =
      context.getWeightGrad(wt_idx[ZoneoutLSTMParams::bias_ih]);
    djdweight_ih.setZero();
    djdweight_hh.setZero();
    djdbias_ih.setZero();

    hidden_state_derivative.setZero();
    cell_state_derivative.setZero();
  }

  next_hidden_state_derivative.add_i(incoming_derivative);

  Tensor prev_hidden_state_derivative;
  Tensor prev_cell_state_derivative;
  Tensor prev_hidden_state_derivative_residual;
  Tensor prev_cell_state_derivative_residual;

  Tensor &hidden_state_zoneout_mask =
    test
      ? context.getWeight(wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask])
      : context.getTensor(wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask]);
  hidden_state_zoneout_mask.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_hidden_state_zoneout_mask =
    hidden_state_zoneout_mask.getBatchSlice(timestep, 1);
  next_hidden_state_zoneout_mask.reshape({batch_size, 1, 1, unit});
  Tensor prev_hidden_state_zoneout_mask;
  if (!test) {
    prev_hidden_state_zoneout_mask =
      next_hidden_state_zoneout_mask.zoneout_mask(hidden_state_zoneout_rate);
  } else {
    next_hidden_state_zoneout_mask.multiply(-1.0f,
                                            prev_hidden_state_zoneout_mask);
    prev_hidden_state_zoneout_mask.add_i(1.0f);
  }

  if (timestep) {
    prev_hidden_state_derivative =
      hidden_state_derivative.getBatchSlice(timestep - 1, 1);
    prev_hidden_state_derivative.reshape({batch_size, 1, 1, unit});
    next_hidden_state_derivative.multiply(
      prev_hidden_state_zoneout_mask, prev_hidden_state_derivative_residual);
  }

  Tensor &hidden_state_origin_derivative =
    context.getTensorGrad(hidden_state_origin_idx);
  hidden_state_origin_derivative.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_hidden_state_origin_derivative =
    hidden_state_origin_derivative.getBatchSlice(timestep, 1);
  next_hidden_state_origin_derivative.reshape({batch_size, 1, 1, unit});

  next_hidden_state_derivative.multiply(next_hidden_state_zoneout_mask,
                                        next_hidden_state_origin_derivative);

  Tensor &cell_state_zoneout_mask =
    test
      ? context.getWeight(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask])
      : context.getTensor(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask]);
  cell_state_zoneout_mask.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_cell_state_zoneout_mask =
    cell_state_zoneout_mask.getBatchSlice(timestep, 1);
  next_cell_state_zoneout_mask.reshape({batch_size, 1, 1, unit});
  Tensor prev_cell_state_zoneout_mask;
  if (!test) {
    prev_cell_state_zoneout_mask =
      next_cell_state_zoneout_mask.zoneout_mask(cell_state_zoneout_rate);
  } else {
    next_cell_state_zoneout_mask.multiply(-1.0f, prev_cell_state_zoneout_mask);
    prev_cell_state_zoneout_mask.add_i(1.0f);
  }

  if (timestep) {
    prev_cell_state_derivative =
      cell_state_derivative.getBatchSlice(timestep - 1, 1);
    prev_cell_state_derivative.reshape({batch_size, 1, 1, unit});
    next_cell_state_derivative.multiply(prev_cell_state_zoneout_mask,
                                        prev_cell_state_derivative_residual);
  }

  Tensor &cell_state_origin_derivative =
    context.getTensorGrad(cell_state_origin_idx);
  cell_state_origin_derivative.reshape({max_timestep, 1, batch_size, unit});
  Tensor next_cell_state_origin_derivative =
    cell_state_origin_derivative.getBatchSlice(timestep, 1);
  next_cell_state_origin_derivative.reshape({batch_size, 1, 1, unit});

  next_cell_state_derivative.multiply(next_cell_state_zoneout_mask,
                                      next_cell_state_origin_derivative);

  init_lstm_context::fillWeights(weights, context, true, max_timestep, timestep,
                                 test);
  init_lstm_context::fillInputs(inputs, context, true, getInputIdx(wt_idx),
                                max_timestep, timestep);
  init_lstm_context::fillOutputs(outputs, context, true, getOutputIdx(wt_idx),
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

  if (timestep) {
    prev_hidden_state_derivative.add_i(prev_hidden_state_derivative_residual);
    prev_cell_state_derivative.add_i(prev_cell_state_derivative_residual);
  }
}

void ZoneoutLSTMCellLayer::setBatch(RunLayerContext &context,
                                    unsigned int batch) {
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(zoneout_lstmcell_props);
  context.updateTensor(wt_idx[ZoneoutLSTMParams::hidden_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[ZoneoutLSTMParams::cell_state],
                       max_timestep * batch);
  context.updateTensor(hidden_state_origin_idx, max_timestep * batch);
  context.updateTensor(cell_state_origin_idx, max_timestep * batch);
  context.updateTensor(wt_idx[ZoneoutLSTMParams::ifgo], max_timestep * batch);

  context.updateTensor(wt_idx[ZoneoutLSTMParams::hidden_state_zoneout_mask],
                       max_timestep * batch);
  context.updateTensor(wt_idx[ZoneoutLSTMParams::cell_state_zoneout_mask],
                       max_timestep * batch);
}

} // namespace nntrainer
