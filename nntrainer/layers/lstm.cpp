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
#include <nntr_threads.h>
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
  reverse_weight_ih,
  reverse_weight_hh,
  reverse_bias_h,
  reverse_bias_ih,
  reverse_bias_hh,
  reverse_hidden_state,
  reverse_cell_state,
  reverse_ifgo,
  dropout_mask
};

void LSTMLayer::forwardingBatchFirstLSTM(
  unsigned int NUM_GATE, const unsigned int batch_size,
  const unsigned int feature_size, const bool disable_bias,
  const unsigned int unit, const bool integrate_bias, ActiFunc &acti_func,
  ActiFunc &recurrent_acti_func, const bool enable_dropout,
  const float dropout_rate, const unsigned int max_timestep, const bool reverse,
  const Tensor &input_, const Tensor &weight_ih, const Tensor &weight_hh,
  const Tensor &bias_h, const Tensor &bias_ih, const Tensor &bias_hh,
  Tensor &hidden_state_, Tensor &cell_state_, Tensor &ifgo_,
  const Tensor &mask_) {
  hidden_state_.setZero();
  cell_state_.setZero();

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    const Tensor input_sample = input_.getBatchSlice(batch, 1);
    Tensor hidden_state_sample = hidden_state_.getBatchSlice(batch, 1);
    Tensor cell_state_sample = cell_state_.getBatchSlice(batch, 1);
    Tensor ifgo_sample = ifgo_.getBatchSlice(batch, 1);

    for (unsigned int t = 0; t < max_timestep; ++t) {
      Tensor input = input_sample.getSharedDataTensor(
        {feature_size}, (reverse ? max_timestep - 1 - t : t) * feature_size);
      Tensor prev_hidden_state;

      if (!t) {
        prev_hidden_state = Tensor(unit);
        prev_hidden_state.setZero();
      } else {
        prev_hidden_state = hidden_state_sample.getSharedDataTensor(
          {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
      }
      Tensor hidden_state = hidden_state_sample.getSharedDataTensor(
        {unit}, (reverse ? max_timestep - 1 - t : t) * unit);
      Tensor prev_cell_state;
      if (!t) {
        prev_cell_state = Tensor(unit);
        prev_cell_state.setZero();
      } else {
        prev_cell_state = cell_state_sample.getSharedDataTensor(
          {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
      }
      Tensor cell_state = cell_state_sample.getSharedDataTensor(
        {unit}, (reverse ? max_timestep - 1 - t : t) * unit);
      Tensor ifgo = ifgo_sample.getSharedDataTensor(
        {NUM_GATE * unit},
        (reverse ? max_timestep - 1 - t : t) * NUM_GATE * unit);

      forwardLSTM(1, unit, disable_bias, integrate_bias, acti_func,
                  recurrent_acti_func, input, prev_hidden_state,
                  prev_cell_state, hidden_state, cell_state, weight_ih,
                  weight_hh, bias_h, bias_ih, bias_hh, ifgo);

      if (enable_dropout) {
        Tensor mask_sample = mask_.getBatchSlice(batch, 1);
        Tensor mask = mask_sample.getSharedDataTensor({unit}, t * unit);
        mask.dropout_mask(dropout_rate);
        hidden_state.multiply_i(mask);
      }
    }
  }
}

void LSTMLayer::calcGradientBatchFirstLSTM(
  unsigned int NUM_GATE, const unsigned int batch_size,
  const unsigned int feature_size, const bool disable_bias,
  const unsigned int unit, const bool integrate_bias, ActiFunc &acti_func,
  ActiFunc &recurrent_acti_func, const bool return_sequences,
  const bool bidirectional, const bool enable_dropout, const float dropout_rate,
  const unsigned int max_timestep, const bool reverse, const Tensor &input_,
  const Tensor &incoming_derivative, Tensor &d_weight_ih,
  const Tensor &weight_hh, Tensor &d_weight_hh, Tensor &d_bias_h,
  Tensor &d_bias_ih, Tensor &d_bias_hh, const Tensor &hidden_state_,
  Tensor &d_hidden_state_, const Tensor &cell_state_, Tensor &d_cell_state_,
  const Tensor &ifgo_, Tensor &d_ifgo_, const Tensor &mask_) {
  const unsigned int bidirectional_constant = bidirectional ? 2 : 1;

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

  d_cell_state_.setZero();
  d_hidden_state_.setZero();

  if (return_sequences && !bidirectional && !reverse) {
    std::copy(incoming_derivative.getData(),
              incoming_derivative.getData() + incoming_derivative.size(),
              d_hidden_state_.getData());
  } else {
    unsigned int end_timestep = return_sequences ? max_timestep : 1;
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      for (unsigned int timestep = 0; timestep < end_timestep; ++timestep) {
        Tensor d_hidden_state_sample = d_hidden_state_.getSharedDataTensor(
          {unit}, batch * max_timestep * unit +
                    (return_sequences ? 0 : max_timestep - 1) * unit +
                    timestep * unit);
        Tensor incoming_derivative_sample =
          incoming_derivative.getSharedDataTensor(
            {unit}, batch * (return_sequences ? max_timestep : 1) *
                        bidirectional_constant * unit +
                      timestep * bidirectional_constant * unit +
                      (reverse ? unit : 0));
        d_hidden_state_sample.add_i(incoming_derivative_sample);
      }
    }
  }

  if (enable_dropout) {
    d_hidden_state_.multiply_i(mask_);
  }

  auto workers = ParallelBatch(batch_size);

  if (workers.getNumWorkers() > 1) {

    TensorDim weight_ih_d = d_weight_ih.getDim();
    TensorDim weight_hh_d = d_weight_hh.getDim();

    TensorDim bias_ih_d = d_bias_ih.getDim();
    TensorDim bias_hh_d = d_bias_hh.getDim();
    TensorDim bias_h_d = d_bias_h.getDim();

    weight_ih_d.batch(workers.getNumWorkers());
    weight_hh_d.batch(workers.getNumWorkers());
    bias_ih_d.batch(workers.getNumWorkers());
    bias_hh_d.batch(workers.getNumWorkers());
    bias_h_d.batch(workers.getNumWorkers());

    Tensor sub_d_weight_ih = Tensor(weight_ih_d);
    Tensor sub_d_weight_hh = Tensor(weight_hh_d);
    Tensor sub_d_bias_ih = Tensor(bias_ih_d);
    Tensor sub_d_bias_hh = Tensor(bias_hh_d);
    Tensor sub_d_bias_h = Tensor(bias_h_d);

    sub_d_weight_ih.setZero();
    sub_d_weight_hh.setZero();
    sub_d_bias_ih.setZero();
    sub_d_bias_hh.setZero();
    sub_d_bias_h.setZero();

    auto batch_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                         void *user_data) {
      for (unsigned int batch = s; batch < e; ++batch) {
        const Tensor input_sample = input_.getBatchSlice(batch, 1);

        const Tensor hidden_state_sample =
          hidden_state_.getBatchSlice(batch, 1);
        Tensor d_hidden_state_sample = d_hidden_state_.getBatchSlice(batch, 1);
        const Tensor cell_state_sample = cell_state_.getBatchSlice(batch, 1);
        Tensor d_cell_state_sample = d_cell_state_.getBatchSlice(batch, 1);

        const Tensor ifgo_sample = ifgo_.getBatchSlice(batch, 1);
        Tensor d_ifgo_sample = d_ifgo_.getBatchSlice(batch, 1);

        Tensor input;
        Tensor prev_hidden_state;
        Tensor d_prev_hidden_state;
        Tensor prev_cell_state;
        Tensor d_prev_cell_state;
        Tensor d_hidden_state;
        Tensor cell_state;
        Tensor d_cell_state;

        Tensor p_d_weight_ih = sub_d_weight_ih.getBatchSlice(pid, 1);
        Tensor p_d_weight_hh = sub_d_weight_hh.getBatchSlice(pid, 1);
        Tensor p_d_bias_ih = sub_d_bias_ih.getBatchSlice(pid, 1);
        Tensor p_d_bias_hh = sub_d_bias_hh.getBatchSlice(pid, 1);
        Tensor p_d_bias_h = sub_d_bias_h.getBatchSlice(pid, 1);

        for (int t = max_timestep - 1; t > -1; t--) {
          input = input_sample.getSharedDataTensor(
            {feature_size},
            (reverse ? max_timestep - 1 - t : t) * feature_size);

          if (!t) {
            prev_hidden_state = Tensor(unit);
            prev_hidden_state.setZero();
            d_prev_hidden_state = Tensor(unit);
            d_prev_hidden_state.setZero();
          } else {
            prev_hidden_state = hidden_state_sample.getSharedDataTensor(
              {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
            d_prev_hidden_state = d_hidden_state_sample.getSharedDataTensor(
              {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
          }
          d_hidden_state = d_hidden_state_sample.getSharedDataTensor(
            {unit}, (reverse ? max_timestep - 1 - t : t) * unit);

          if (!t) {
            prev_cell_state = Tensor(unit);
            prev_cell_state.setZero();
            d_prev_cell_state = Tensor(unit);
            d_prev_cell_state.setZero();
          } else {
            prev_cell_state = cell_state_sample.getSharedDataTensor(
              {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
            d_prev_cell_state = d_cell_state_sample.getSharedDataTensor(
              {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
          }
          cell_state = cell_state_sample.getSharedDataTensor(
            {unit}, (reverse ? max_timestep - 1 - t : t) * unit);
          d_cell_state = d_cell_state_sample.getSharedDataTensor(
            {unit}, (reverse ? max_timestep - 1 - t : t) * unit);

          Tensor ifgo = ifgo_sample.getSharedDataTensor(
            {NUM_GATE * unit},
            (reverse ? max_timestep - 1 - t : t) * NUM_GATE * unit);
          Tensor d_ifgo = d_ifgo_sample.getSharedDataTensor(
            {NUM_GATE * unit},
            (reverse ? max_timestep - 1 - t : t) * NUM_GATE * unit);

          // Temporary variable for d_prev_hidden_state. d_prev_hidden_state
          // already have precalculated values from incomming derivatives
          Tensor d_prev_hidden_state_temp;

          calcGradientLSTM(
            1, unit, disable_bias, integrate_bias, acti_func,
            recurrent_acti_func, input, prev_hidden_state,
            d_prev_hidden_state_temp, prev_cell_state, d_prev_cell_state,
            d_hidden_state, cell_state, d_cell_state, p_d_weight_ih, weight_hh,
            p_d_weight_hh, p_d_bias_h, p_d_bias_ih, p_d_bias_hh, ifgo, d_ifgo);

          d_prev_hidden_state.add_i(d_prev_hidden_state_temp);
        }
      }
    };

    workers.setCallback(batch_job, nullptr);
    workers.run();

    for (unsigned int b = 0; b < workers.getNumWorkers(); ++b) {

      Tensor p_d_weight_ih = sub_d_weight_ih.getBatchSlice(b, 1);
      Tensor p_d_weight_hh = sub_d_weight_hh.getBatchSlice(b, 1);
      Tensor p_d_bias_ih = sub_d_bias_ih.getBatchSlice(b, 1);
      Tensor p_d_bias_hh = sub_d_bias_hh.getBatchSlice(b, 1);
      Tensor p_d_bias_h = sub_d_bias_h.getBatchSlice(b, 1);

      d_weight_ih.add_i(p_d_weight_ih);
      d_weight_hh.add_i(p_d_weight_hh);
      d_bias_ih.add_i(p_d_bias_ih);
      d_bias_hh.add_i(p_d_bias_hh);
      d_bias_h.add_i(p_d_bias_h);
    }

  } else {
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      const Tensor input_sample = input_.getBatchSlice(batch, 1);

      const Tensor hidden_state_sample = hidden_state_.getBatchSlice(batch, 1);
      Tensor d_hidden_state_sample = d_hidden_state_.getBatchSlice(batch, 1);
      const Tensor cell_state_sample = cell_state_.getBatchSlice(batch, 1);
      Tensor d_cell_state_sample = d_cell_state_.getBatchSlice(batch, 1);

      const Tensor ifgo_sample = ifgo_.getBatchSlice(batch, 1);
      Tensor d_ifgo_sample = d_ifgo_.getBatchSlice(batch, 1);

      Tensor input;
      Tensor prev_hidden_state;
      Tensor d_prev_hidden_state;
      Tensor prev_cell_state;
      Tensor d_prev_cell_state;
      Tensor d_hidden_state;
      Tensor cell_state;
      Tensor d_cell_state;

      for (int t = max_timestep - 1; t > -1; t--) {
        input = input_sample.getSharedDataTensor(
          {feature_size}, (reverse ? max_timestep - 1 - t : t) * feature_size);

        if (!t) {
          prev_hidden_state = Tensor(unit);
          prev_hidden_state.setZero();
          d_prev_hidden_state = Tensor(unit);
          d_prev_hidden_state.setZero();
        } else {
          prev_hidden_state = hidden_state_sample.getSharedDataTensor(
            {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
          d_prev_hidden_state = d_hidden_state_sample.getSharedDataTensor(
            {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
        }
        d_hidden_state = d_hidden_state_sample.getSharedDataTensor(
          {unit}, (reverse ? max_timestep - 1 - t : t) * unit);

        if (!t) {
          prev_cell_state = Tensor(unit);
          prev_cell_state.setZero();
          d_prev_cell_state = Tensor(unit);
          d_prev_cell_state.setZero();
        } else {
          prev_cell_state = cell_state_sample.getSharedDataTensor(
            {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
          d_prev_cell_state = d_cell_state_sample.getSharedDataTensor(
            {unit}, (reverse ? (max_timestep - t) : (t - 1)) * unit);
        }
        cell_state = cell_state_sample.getSharedDataTensor(
          {unit}, (reverse ? max_timestep - 1 - t : t) * unit);
        d_cell_state = d_cell_state_sample.getSharedDataTensor(
          {unit}, (reverse ? max_timestep - 1 - t : t) * unit);

        Tensor ifgo = ifgo_sample.getSharedDataTensor(
          {NUM_GATE * unit},
          (reverse ? max_timestep - 1 - t : t) * NUM_GATE * unit);
        Tensor d_ifgo = d_ifgo_sample.getSharedDataTensor(
          {NUM_GATE * unit},
          (reverse ? max_timestep - 1 - t : t) * NUM_GATE * unit);

        // Temporary variable for d_prev_hidden_state. d_prev_hidden_state
        // already have precalculated values from incomming derivatives
        Tensor d_prev_hidden_state_temp;

        calcGradientLSTM(1, unit, disable_bias, integrate_bias, acti_func,
                         recurrent_acti_func, input, prev_hidden_state,
                         d_prev_hidden_state_temp, prev_cell_state,
                         d_prev_cell_state, d_hidden_state, cell_state,
                         d_cell_state, d_weight_ih, weight_hh, d_weight_hh,
                         d_bias_h, d_bias_ih, d_bias_hh, ifgo, d_ifgo);
        d_prev_hidden_state.add_i(d_prev_hidden_state_temp);
      }
    }
  }
}

LSTMLayer::LSTMLayer() :
  LSTMCore(),
  lstm_props(props::ReturnSequences(), props::Bidirectional(),
             props::DropOutRate(), props::MaxTimestep()) {
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
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  NNTR_THROW_IF(std::get<props::Unit>(lstmcore_props).empty(),
                std::invalid_argument)
    << "unit property missing for lstm layer";
  const unsigned int unit = std::get<props::Unit>(lstmcore_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcore_props).get();
  const ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(lstmcore_props).get();
  const ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(lstmcore_props).get();

  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const bool bidirectional = std::get<props::Bidirectional>(lstm_props).get();
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

  // output_dim = [ batch_size, 1, return_sequences ? time_iteration : 1,
  // bidirectional ? 2 * unit : unit ]
  const TensorDim output_dim(batch_size, 1, return_sequences ? max_timestep : 1,
                             bidirectional ? 2 * unit : unit);
  context.setOutputDimensions({output_dim});

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in
  // keras for now, it is set same way.

  // weight_ih ( input to hidden ) : [ 1, 1, feature_size, NUM_GATE * unit ]
  // -> i, f, g, o
  const TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[LSTMParams::weight_ih] = context.requestWeight(
    weight_ih_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_ih", true);
  // weight_hh ( hidden to hidden ) : [ 1, 1, unit, NUM_GATE * unit ] -> i,
  // f, g, o
  const TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[LSTMParams::weight_hh] = context.requestWeight(
    weight_hh_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // bias_h ( input bias, hidden bias are integrate to 1 bias ) : [ 1,
      // 1, 1, NUM_GATE * unit ] -> i, f, g, o
      const TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[LSTMParams::bias_h] = context.requestWeight(
        bias_h_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
        "bias_h", true);
    } else {
      // bias_ih ( input bias ) : [ 1, 1, 1, NUM_GATE * unit ] -> i, f, g, o
      const TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[LSTMParams::bias_ih] = context.requestWeight(
        bias_ih_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_ih", true);
      // bias_hh ( hidden bias ) : [ 1, 1, 1, NUM_GATE * unit ] -> i, f, g, o
      wt_idx[LSTMParams::bias_hh] = context.requestWeight(
        bias_ih_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_hh", true);
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

  if (bidirectional) {
    // weight_initializer can be set seperately. weight_ih initializer,
    // weight_hh initializer kernel initializer & recurrent_initializer in
    // keras for now, it is set same way.

    // reverse_weight_ih ( input to hidden ) : [ 1, 1, feature_size,
    // NUM_GATE * unit ] -> i, f, g, o
    const TensorDim reverse_weight_ih_dim({feature_size, NUM_GATE * unit});
    wt_idx[LSTMParams::reverse_weight_ih] = context.requestWeight(
      reverse_weight_ih_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "reverse_weight_ih", true);
    // reverse_weight_hh ( hidden to hidden ) : [ 1, 1, unit, NUM_GATE *
    // unit ]
    // -> i, f, g, o
    const TensorDim reverse_weight_hh_dim({unit, NUM_GATE * unit});
    wt_idx[LSTMParams::reverse_weight_hh] = context.requestWeight(
      reverse_weight_hh_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "reverse_weight_hh", true);
    if (!disable_bias) {
      if (integrate_bias) {
        // reverse_bias_h ( input bias, hidden bias are integrate to 1 bias
        // ) : [ 1, 1, 1, NUM_GATE * unit ] -> i, f, g, o
        const TensorDim reverse_bias_h_dim({NUM_GATE * unit});
        wt_idx[LSTMParams::reverse_bias_h] = context.requestWeight(
          reverse_bias_h_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
          bias_decay, "reverse_bias_h", true);
      } else {
        // reverse_bias_ih ( input bias ) : [ 1, 1, 1, NUM_GATE * unit ] ->
        // i, f, g, o
        const TensorDim reverse_bias_ih_dim({NUM_GATE * unit});
        wt_idx[LSTMParams::reverse_bias_ih] = context.requestWeight(
          reverse_bias_ih_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
          bias_decay, "reverse_bias_ih", true);
        // reverse_bias_hh ( hidden bias ) : [ 1, 1, 1, NUM_GATE * unit ] ->
        // i, f, g, o
        const TensorDim reverse_bias_hh_dim({NUM_GATE * unit});
        wt_idx[LSTMParams::reverse_bias_hh] = context.requestWeight(
          reverse_bias_hh_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
          bias_decay, "reverse_bias_hh", true);
      }
    }

    // reverse_hidden_state_dim : [ batch_size, 1, max_timestep, unit ]
    const TensorDim reverse_hidden_state_dim(batch_size, 1, max_timestep, unit);
    wt_idx[LSTMParams::reverse_hidden_state] = context.requestTensor(
      reverse_hidden_state_dim, "reverse_hidden_state",
      Tensor::Initializer::NONE, true, TensorLifespan::ITERATION_LIFESPAN);
    // reverse_cell_state_dim : [ batch_size, 1, max_timestep, unit ]
    const TensorDim reverse_cell_state_dim(batch_size, 1, max_timestep, unit);
    wt_idx[LSTMParams::reverse_cell_state] = context.requestTensor(
      reverse_cell_state_dim, "reverse_cell_state", Tensor::Initializer::NONE,
      true, TensorLifespan::ITERATION_LIFESPAN);

    // reverse_ifgo_dim : [ batch_size, 1, max_timestep, NUM_GATE * unit ]
    const TensorDim reverse_ifgo_dim(batch_size, 1, max_timestep,
                                     NUM_GATE * unit);
    wt_idx[LSTMParams::reverse_ifgo] = context.requestTensor(
      reverse_ifgo_dim, "reverse_ifgo", Tensor::Initializer::NONE, true,
      TensorLifespan::ITERATION_LIFESPAN);
  }

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
  LSTMCore::setProperty(remain_props);
}

void LSTMLayer::exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const {
  LSTMCore::exportTo(exporter, method);
  exporter.saveResult(lstm_props, method, this);
}

void LSTMLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstmcore_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcore_props).get();

  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const bool bidirectional = std::get<props::Bidirectional>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstm_props).get();

  const unsigned int bidirectional_constant = bidirectional ? 2 : 1;
  bool enable_dropout = dropout_rate > epsilon && training;

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim input_dim = input.getDim();
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

  Tensor &hidden_state = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &cell_state = context.getTensor(wt_idx[LSTMParams::cell_state]);
  Tensor &ifgo = context.getTensor(wt_idx[LSTMParams::ifgo]);

  Tensor &mask = enable_dropout
                   ? context.getTensor(wt_idx[LSTMParams::dropout_mask])
                   : empty;

  forwardingBatchFirstLSTM(NUM_GATE, batch_size, feature_size, disable_bias,
                           unit, integrate_bias, acti_func, recurrent_acti_func,
                           enable_dropout, dropout_rate, max_timestep, false,
                           input, weight_ih, weight_hh, bias_h, bias_ih,
                           bias_hh, hidden_state, cell_state, ifgo, mask);

  if (bidirectional) {
    const Tensor &reverse_weight_ih =
      context.getWeight(wt_idx[LSTMParams::reverse_weight_ih]);
    const Tensor &reverse_weight_hh =
      context.getWeight(wt_idx[LSTMParams::reverse_weight_hh]);
    const Tensor &reverse_bias_h =
      !disable_bias && integrate_bias
        ? context.getWeight(wt_idx[LSTMParams::reverse_bias_h])
        : empty;
    const Tensor &reverse_bias_ih =
      !disable_bias && !integrate_bias
        ? context.getWeight(wt_idx[LSTMParams::reverse_bias_ih])
        : empty;
    const Tensor &reverse_bias_hh =
      !disable_bias && !integrate_bias
        ? context.getWeight(wt_idx[LSTMParams::reverse_bias_hh])
        : empty;

    Tensor &reverse_hidden_state =
      context.getTensor(wt_idx[LSTMParams::reverse_hidden_state]);
    Tensor &reverse_cell_state =
      context.getTensor(wt_idx[LSTMParams::reverse_cell_state]);
    Tensor &reverse_ifgo = context.getTensor(wt_idx[LSTMParams::reverse_ifgo]);

    forwardingBatchFirstLSTM(
      NUM_GATE, batch_size, feature_size, disable_bias, unit, integrate_bias,
      acti_func, recurrent_acti_func, enable_dropout, dropout_rate,
      max_timestep, true, input, reverse_weight_ih, reverse_weight_hh,
      reverse_bias_h, reverse_bias_ih, reverse_bias_hh, reverse_hidden_state,
      reverse_cell_state, reverse_ifgo, mask);
  }

  if (return_sequences && !bidirectional) {
    std::copy(hidden_state.getData(),
              hidden_state.getData() + hidden_state.size(), output.getData());
  } else {
    unsigned int end_timestep = return_sequences ? max_timestep : 1;
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      for (unsigned int timestep = 0; timestep < end_timestep; ++timestep) {
        float *hidden_state_data = hidden_state.getAddress<float>(
          batch * max_timestep * unit +
          (return_sequences ? 0 : (max_timestep - 1) * unit) + timestep * unit);
        float *output_data = output.getAddress<float>(
          batch * (return_sequences ? max_timestep : 1) *
            bidirectional_constant * unit +
          timestep * bidirectional_constant * unit);
        std::copy(hidden_state_data, hidden_state_data + unit, output_data);

        if (bidirectional) {
          Tensor &reverse_hidden_state =
            context.getTensor(wt_idx[LSTMParams::reverse_hidden_state]);
          float *reverse_hidden_state_data =
            reverse_hidden_state.getAddress<float>(
              batch * max_timestep * unit +
              (return_sequences ? 0 : (max_timestep - 1) * unit) +
              timestep * unit);
          std::copy(reverse_hidden_state_data, reverse_hidden_state_data + unit,
                    output_data + unit);
        }
      }
    }
  }
}

void LSTMLayer::calcDerivative(RunLayerContext &context) {
  const bool bidirectional = std::get<props::Bidirectional>(lstm_props).get();

  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &weight_ih = context.getWeight(wt_idx[LSTMParams::weight_ih]);
  const Tensor &d_ifgos = context.getTensorGrad(wt_idx[LSTMParams::ifgo]);

  calcDerivativeLSTM(outgoing_derivative, weight_ih, d_ifgos);

  if (bidirectional) {
    const Tensor &reverse_weight_ih =
      context.getWeight(wt_idx[LSTMParams::reverse_weight_ih]);
    const Tensor &reverse_d_ifgos =
      context.getTensorGrad(wt_idx[LSTMParams::reverse_ifgo]);

    calcDerivativeLSTM(outgoing_derivative, reverse_weight_ih, reverse_d_ifgos,
                       1.0f);
  }
}

void LSTMLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstmcore_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcore_props).get();

  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const bool bidirectional = std::get<props::Bidirectional>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstm_props).get();

  bool enable_dropout = dropout_rate > epsilon;

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  const TensorDim input_dim = input.getDim();
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

  const Tensor &hidden_state =
    context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &d_hidden_state =
    context.getTensorGrad(wt_idx[LSTMParams::hidden_state]);
  const Tensor &cell_state = context.getTensor(wt_idx[LSTMParams::cell_state]);
  Tensor &d_cell_state = context.getTensorGrad(wt_idx[LSTMParams::cell_state]);

  const Tensor &ifgo = context.getTensor(wt_idx[LSTMParams::ifgo]);
  Tensor &d_ifgo = context.getTensorGrad(wt_idx[LSTMParams::ifgo]);

  const Tensor &mask = enable_dropout
                         ? context.getTensor(wt_idx[LSTMParams::dropout_mask])
                         : empty;

  calcGradientBatchFirstLSTM(
    NUM_GATE, batch_size, feature_size, disable_bias, unit, integrate_bias,
    acti_func, recurrent_acti_func, return_sequences, bidirectional,
    enable_dropout, dropout_rate, max_timestep, false, input,
    incoming_derivative, d_weight_ih, weight_hh, d_weight_hh, d_bias_h,
    d_bias_ih, d_bias_hh, hidden_state, d_hidden_state, cell_state,
    d_cell_state, ifgo, d_ifgo, mask);

  if (bidirectional) {
    Tensor &reverse_d_weight_ih =
      context.getWeightGrad(wt_idx[LSTMParams::reverse_weight_ih]);
    const Tensor &reverse_weight_hh =
      context.getWeight(wt_idx[LSTMParams::reverse_weight_hh]);
    Tensor &reverse_d_weight_hh =
      context.getWeightGrad(wt_idx[LSTMParams::reverse_weight_hh]);
    Tensor &reverse_d_bias_h =
      !disable_bias && integrate_bias
        ? context.getWeightGrad(wt_idx[LSTMParams::reverse_bias_h])
        : empty;
    Tensor &reverse_d_bias_ih =
      !disable_bias && !integrate_bias
        ? context.getWeightGrad(wt_idx[LSTMParams::reverse_bias_ih])
        : empty;
    Tensor &reverse_d_bias_hh =
      !disable_bias && !integrate_bias
        ? context.getWeightGrad(wt_idx[LSTMParams::reverse_bias_hh])
        : empty;

    const Tensor &reverse_hidden_state =
      context.getTensor(wt_idx[LSTMParams::reverse_hidden_state]);
    Tensor &reverse_d_hidden_state =
      context.getTensorGrad(wt_idx[LSTMParams::reverse_hidden_state]);
    const Tensor &reverse_cell_state =
      context.getTensor(wt_idx[LSTMParams::reverse_cell_state]);
    Tensor &reverse_d_cell_state =
      context.getTensorGrad(wt_idx[LSTMParams::reverse_cell_state]);

    const Tensor &reverse_ifgo =
      context.getTensor(wt_idx[LSTMParams::reverse_ifgo]);
    Tensor &reverse_d_ifgo =
      context.getTensorGrad(wt_idx[LSTMParams::reverse_ifgo]);

    calcGradientBatchFirstLSTM(
      NUM_GATE, batch_size, feature_size, disable_bias, unit, integrate_bias,
      acti_func, recurrent_acti_func, return_sequences, bidirectional,
      enable_dropout, dropout_rate, max_timestep, true, input,
      incoming_derivative, reverse_d_weight_ih, reverse_weight_hh,
      reverse_d_weight_hh, reverse_d_bias_h, reverse_d_bias_ih,
      reverse_d_bias_hh, reverse_hidden_state, reverse_d_hidden_state,
      reverse_cell_state, reverse_d_cell_state, reverse_ifgo, reverse_d_ifgo,
      mask);
  }
}

void LSTMLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  const bool bidirectional = std::get<props::Bidirectional>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();

  context.updateTensor(wt_idx[LSTMParams::hidden_state], batch);
  context.updateTensor(wt_idx[LSTMParams::cell_state], batch);
  context.updateTensor(wt_idx[LSTMParams::ifgo], batch);

  if (bidirectional) {
    context.updateTensor(wt_idx[LSTMParams::reverse_hidden_state], batch);
    context.updateTensor(wt_idx[LSTMParams::reverse_cell_state], batch);
    context.updateTensor(wt_idx[LSTMParams::reverse_ifgo], batch);
  }

  if (dropout_rate > epsilon) {
    context.updateTensor(wt_idx[LSTMParams::dropout_mask], batch);
  }
}

} // namespace nntrainer
