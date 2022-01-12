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

#include <layer_context.h>
#include <lstm.h>
#include <lstmcell_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LSTMParams {
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

LSTMLayer::LSTMLayer() :
  LayerImpl(),
  lstm_props(props::Unit(), props::IntegrateBias(),
             props::HiddenStateActivation() = ActivationType::ACT_TANH,
             props::RecurrentActivation() = ActivationType::ACT_SIGMOID,
             props::ReturnSequences(), props::DropOutRate(),
             props::MaxTimestep()),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void LSTMLayer::finalize(InitLayerContext &context) {
  const Tensor::Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props).get();
  const Tensor::Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props).get();
  const nntrainer::WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props).get();
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props).get();
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  NNTR_THROW_IF(std::get<props::Unit>(lstm_props).empty(),
                std::invalid_argument)
    << "unit property missing for lstm layer";
  const unsigned int unit = std::get<props::Unit>(lstm_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(lstm_props).get();
  const ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(lstm_props).get();
  const ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(lstm_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("LSTM layer takes only one input");
  }

  // input_dim = [ batch_size, 1, time_iteration, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  if (input_dim.channel() != 1) {
    throw std::invalid_argument(
      "Input must be single channel dimension for LSTM (shape should be "
      "[batch_size, 1, time_iteration, feature_size])");
  }
  const unsigned int batch_size = input_dim.batch();
  unsigned int max_timestep = input_dim.height();
  if (!std::get<props::MaxTimestep>(lstm_props).empty())
    max_timestep =
      std::max(max_timestep, std::get<props::MaxTimestep>(lstm_props).get());
  std::get<props::MaxTimestep>(lstm_props).set(max_timestep);
  const unsigned int feature_size = input_dim.width();

  // if return_sequences == false :
  //      output_dim = [ batch_size, 1, 1, unit ]
  // else:
  //      output_dim = [ batch_size, 1, time_iteration, unit ]
  const TensorDim output_dim(batch_size, 1, return_sequences ? max_timestep : 1,
                             unit);
  context.setOutputDimensions({output_dim});

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // weight_ih ( input to hidden ) : [ 1, 1, feature_size, NUM_GATE * unit ] ->
  // i, f, g, o
  const TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[LSTMParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // weight_hh ( hidden to hidden ) : [ 1, 1, unit, NUM_GATE * unit ] -> i, f,
  // g, o
  const TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[LSTMParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // bias_h ( input bias, hidden bias are integrate to 1 bias ) : [ 1, 1, 1,
      // NUM_GATE * unit ] -> i, f, g, o
      const TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[LSTMParams::bias_h] =
        context.requestWeight(bias_h_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_h", true);
    } else {
      // bias_ih ( input bias ) : [ 1, 1, 1, NUM_GATE * unit ] -> i, f, g, o
      const TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[LSTMParams::bias_ih] =
        context.requestWeight(bias_ih_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_ih", true);
      // bias_hh ( hidden bias ) : [ 1, 1, 1, NUM_GATE * unit ] -> i, f, g, o
      const TensorDim bias_hh_dim({NUM_GATE * unit});
      wt_idx[LSTMParams::bias_hh] =
        context.requestWeight(bias_hh_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_hh", true);
    }
  }

  // hidden_state_dim : [ batch_size, 1, max_timestep, unit ]
  const TensorDim hidden_state_dim(batch_size, 1, max_timestep, unit);
  wt_idx[LSTMParams::hidden_state] = context.requestTensor(
    hidden_state_dim, "hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);
  // cell_state_dim : [ batch_size, 1, max_timestep, unit ]
  const TensorDim cell_state_dim(batch_size, 1, max_timestep, unit);
  wt_idx[LSTMParams::cell_state] = context.requestTensor(
    cell_state_dim, "cell_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  // ifgo_dim : [ batch_size, 1, max_timestep, NUM_GATE * unit ]
  const TensorDim ifgo_dim(batch_size, 1, max_timestep, NUM_GATE * unit);
  wt_idx[LSTMParams::ifgo] =
    context.requestTensor(ifgo_dim, "ifgo", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  if (dropout_rate > epsilon) {
    // dropout_mask_dim = [ batch, 1, time_iteration, unit ]
    const TensorDim dropout_mask_dim(batch_size, 1, max_timestep, unit);
    wt_idx[LSTMParams::dropout_mask] = context.requestTensor(
      dropout_mask_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);
  recurrent_acti_func.setActiFunc(recurrent_activation_type);
}

void LSTMLayer::setProperty(const std::vector<std::string> &values) {
  const std::vector<std::string> &remain_props =
    loadProperties(values, lstm_props);
  LayerImpl::setProperty(remain_props);
}

void LSTMLayer::exportTo(Exporter &exporter,
                         const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(lstm_props, method, this);
}

void LSTMLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstm_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(lstm_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstm_props).get();

  const Tensor &inputs = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim input_dim = inputs.getDim();
  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const Tensor &weight_ih = context.getWeight(wt_idx[LSTMParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[LSTMParams::weight_hh]);
  Tensor empty;
  const Tensor &bias_h = !disable_bias && integrate_bias
                           ? context.getWeight(wt_idx[LSTMParams::bias_h])
                           : empty;
  const Tensor &bias_ih = !disable_bias && !integrate_bias
                            ? context.getWeight(wt_idx[LSTMParams::bias_ih])
                            : empty;
  const Tensor &bias_hh = !disable_bias && !integrate_bias
                            ? context.getWeight(wt_idx[LSTMParams::bias_hh])
                            : empty;

  Tensor &hs = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &cs = context.getTensor(wt_idx[LSTMParams::cell_state]);
  Tensor &ifgos = context.getTensor(wt_idx[LSTMParams::ifgo]);

  hs.setZero();
  cs.setZero();

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    const Tensor input_batch = inputs.getBatchSlice(batch, 1);
    Tensor hs_batch = hs.getBatchSlice(batch, 1);
    Tensor cs_batch = cs.getBatchSlice(batch, 1);
    Tensor ifgo_batch = ifgos.getBatchSlice(batch, 1);

    for (unsigned int t = 0; t < max_timestep; ++t) {
      Tensor input;
      if (input_batch.height() != 1)
        input =
          input_batch.getSharedDataTensor({feature_size}, t * feature_size);
      else
        input = input_batch;

      Tensor prev_hidden_state;
      if (!t) {
        prev_hidden_state = Tensor(unit);
        prev_hidden_state.setZero();
      } else {
        prev_hidden_state =
          hs_batch.getSharedDataTensor({unit}, (t - 1) * unit);
      }
      Tensor hidden_state = hs_batch.getSharedDataTensor({unit}, t * unit);
      Tensor prev_cell_state;
      if (!t) {
        prev_cell_state = Tensor(unit);
        prev_cell_state.setZero();
      } else {
        prev_cell_state = cs_batch.getSharedDataTensor({unit}, (t - 1) * unit);
      }
      Tensor cell_state = cs_batch.getSharedDataTensor({unit}, t * unit);
      Tensor ifgo =
        ifgo_batch.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      lstmcell_forwarding(unit, 1, disable_bias, integrate_bias, acti_func,
                          recurrent_acti_func, input, prev_hidden_state,
                          prev_cell_state, hidden_state, cell_state, weight_ih,
                          weight_hh, bias_h, bias_ih, bias_hh, ifgo);

      if (dropout_rate > epsilon && training) {
        Tensor masks = context.getTensor(wt_idx[LSTMParams::dropout_mask])
                         .getBatchSlice(batch, 1);
        Tensor mask = masks.getSharedDataTensor({unit}, t * unit);
        mask.dropout_mask(dropout_rate);
        hidden_state.multiply_i(mask);
      }
    }
  }

  if (return_sequences) {
    std::copy(hs.getData(), hs.getData() + hs.size(), output.getData());
  } else {
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      float *hidden_state_data =
        hs.getAddress(batch * max_timestep * unit + (max_timestep - 1) * unit);
      float *output_data = output.getAddress(batch * unit);
      std::copy(hidden_state_data, hidden_state_data + unit, output_data);
    }
  }
}

void LSTMLayer::calcDerivative(RunLayerContext &context) {
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &weight_ih = context.getWeight(wt_idx[LSTMParams::weight_ih]);
  const Tensor &d_ifgos = context.getTensorGrad(wt_idx[LSTMParams::ifgo]);

  lstmcell_calcDerivative(d_ifgos, weight_ih, outgoing_derivative);
}

void LSTMLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstm_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(lstm_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstm_props).get();

  unsigned int start_timestep = max_timestep - 1;
  int end_timestep = -1;

  const Tensor &inputs = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  const TensorDim input_dim = inputs.getDim();
  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();

  Tensor &d_weight_ih = context.getWeightGrad(wt_idx[LSTMParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[LSTMParams::weight_hh]);
  Tensor &d_weight_hh = context.getWeightGrad(wt_idx[LSTMParams::weight_hh]);
  Tensor empty;
  Tensor &d_bias_h = !disable_bias && integrate_bias
                       ? context.getWeightGrad(wt_idx[LSTMParams::bias_h])
                       : empty;
  Tensor &d_bias_ih = !disable_bias && !integrate_bias
                        ? context.getWeightGrad(wt_idx[LSTMParams::bias_ih])
                        : empty;
  Tensor &d_bias_hh = !disable_bias && !integrate_bias
                        ? context.getWeightGrad(wt_idx[LSTMParams::bias_hh])
                        : empty;

  Tensor &hs = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &d_hs = context.getTensorGrad(wt_idx[LSTMParams::hidden_state]);
  Tensor &cs = context.getTensor(wt_idx[LSTMParams::cell_state]);
  Tensor &d_cs = context.getTensorGrad(wt_idx[LSTMParams::cell_state]);

  Tensor &ifgos = context.getTensor(wt_idx[LSTMParams::ifgo]);
  Tensor &d_ifgos = context.getTensorGrad(wt_idx[LSTMParams::ifgo]);

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

  d_cs.setZero();
  d_hs.setZero();

  if (return_sequences) {
    std::copy(incoming_derivative.getData(),
              incoming_derivative.getData() + incoming_derivative.size(),
              d_hs.getData());
  } else {
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      Tensor data = d_hs.getSharedDataTensor(
        {unit}, batch * max_timestep * unit + start_timestep * unit);

      Tensor rdata =
        incoming_derivative.getSharedDataTensor({unit}, batch * unit);
      /// @note this is not copying from start ~ end but only start time step
      // This is copying for self rolling as well as last recurrent unrolled.
      if ((unsigned)start_timestep + 1 == max_timestep) {
        data.fill(rdata);
      } else {
        data.add_i(rdata);
      }
    }
  }

  if (dropout_rate > epsilon) {
    d_hs.multiply_i(context.getTensor(wt_idx[LSTMParams::dropout_mask]));
  }

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    const Tensor input_batch = inputs.getBatchSlice(batch, 1);

    Tensor hs_batch = hs.getBatchSlice(batch, 1);
    Tensor d_hidden_state_batch = d_hs.getBatchSlice(batch, 1);
    Tensor cs_batch = cs.getBatchSlice(batch, 1);
    Tensor d_cell_state_batch = d_cs.getBatchSlice(batch, 1);

    Tensor ifgo_batch = ifgos.getBatchSlice(batch, 1);
    Tensor d_ifgo_batch = d_ifgos.getBatchSlice(batch, 1);

    Tensor input;
    Tensor prev_hidden_state;
    Tensor d_prev_hidden_state;
    Tensor prev_cell_state;
    Tensor d_prev_cell_state;
    Tensor d_hidden_state;
    Tensor cell_state;
    Tensor d_cell_state;

    for (int t = start_timestep; t > end_timestep; t--) {
      if (input_batch.height() != 1)
        input =
          input_batch.getSharedDataTensor({feature_size}, t * feature_size);
      else
        input = input_batch;

      if (!t) {
        prev_hidden_state = Tensor(unit);
        prev_hidden_state.setZero();
        d_prev_hidden_state = Tensor(unit);
        d_prev_hidden_state.setZero();
      } else {
        prev_hidden_state =
          hs_batch.getSharedDataTensor({unit}, (t - 1) * unit);
        d_prev_hidden_state =
          d_hidden_state_batch.getSharedDataTensor({unit}, (t - 1) * unit);
      }
      d_hidden_state =
        d_hidden_state_batch.getSharedDataTensor({unit}, t * unit);

      if (!t) {
        prev_cell_state = Tensor(unit);
        prev_cell_state.setZero();
        d_prev_cell_state = Tensor(unit);
        d_prev_cell_state.setZero();
      } else {
        prev_cell_state = cs_batch.getSharedDataTensor({unit}, (t - 1) * unit);
        d_prev_cell_state =
          d_cell_state_batch.getSharedDataTensor({unit}, (t - 1) * unit);
      }
      cell_state = cs_batch.getSharedDataTensor({unit}, t * unit);
      d_cell_state = d_cell_state_batch.getSharedDataTensor({unit}, t * unit);

      Tensor ifgo =
        ifgo_batch.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);
      Tensor d_ifgo = d_ifgo_batch.getSharedDataTensor({unit * NUM_GATE},
                                                       unit * t * NUM_GATE);

      // Temporary variable for d_prev_hidden_state. d_prev_hidden_state already
      // have precalculated values from incomming derivatives
      Tensor d_prev_hidden_state_temp;

      lstmcell_calcGradient(unit, 1, disable_bias, integrate_bias, acti_func,
                            recurrent_acti_func, input, prev_hidden_state,
                            d_prev_hidden_state_temp, prev_cell_state,
                            d_prev_cell_state, d_hidden_state, cell_state,
                            d_cell_state, d_weight_ih, weight_hh, d_weight_hh,
                            d_bias_h, d_bias_ih, d_bias_hh, ifgo, d_ifgo);
      d_prev_hidden_state.add_i(d_prev_hidden_state_temp);
    }
  }
}

void LSTMLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[LSTMParams::hidden_state], batch);
  context.updateTensor(wt_idx[LSTMParams::cell_state], batch);
  context.updateTensor(wt_idx[LSTMParams::ifgo], batch);

  if (std::get<props::DropOutRate>(lstm_props).get() > epsilon) {
    context.updateTensor(wt_idx[LSTMParams::dropout_mask], batch);
  }
}

} // namespace nntrainer
