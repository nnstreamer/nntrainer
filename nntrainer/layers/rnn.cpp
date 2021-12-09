// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rnn.cpp
 * @date   17 March 2021
 * @brief  This is Recurrent Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <rnn.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// - weight_ih ( input to hidden )
// - weight_hh ( hidden to hidden )
// - bias_h ( input bias, hidden bias )
// - bias_ih ( input bias )
// - bias_hh ( hidden bias )
enum RNNParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  hidden_state,
  dropout_mask
};

RNNLayer::RNNLayer() :
  LayerImpl(),
  rnn_props(
    props::Unit(), props::HiddenStateActivation() = ActivationType::ACT_TANH,
    props::ReturnSequences(), props::DropOutRate(), props::IntegrateBias()),
  acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void RNNLayer::finalize(InitLayerContext &context) {
  const nntrainer::WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  const Tensor::Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  const Tensor::Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props);
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(rnn_props).get();
  const nntrainer::ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(rnn_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(rnn_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnn_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(rnn_props).get();

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("RNN layer takes only one input");
  }

  // input_dim = [ batch, 1, time_iteration, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const unsigned int batch_size = input_dim.batch();
  const unsigned int max_timestep = input_dim.height();
  const unsigned int feature_size = input_dim.width();

  // output_dim = [ batch, 1, (return_sequences ? time_iteration : 1), unit ]
  const TensorDim output_dim(batch_size, 1, return_sequences ? max_timestep : 1,
                             unit);

  context.setOutputDimensions({output_dim});

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // weight_ih_dim : [ 1, 1, feature_size, unit ]
  const TensorDim weight_ih_dim({feature_size, unit});
  wt_idx[RNNParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // weight_hh_dim : [ 1, 1, unit, unit ]
  const TensorDim weight_hh_dim({unit, unit});
  wt_idx[RNNParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // bias_h_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_h_dim({unit});
      wt_idx[RNNParams::bias_h] =
        context.requestWeight(bias_h_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_h", true);
    } else {
      // bias_ih_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_ih_dim({unit});
      wt_idx[RNNParams::bias_ih] =
        context.requestWeight(bias_ih_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_ih", true);
      // bias_hh_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_hh_dim({unit});
      wt_idx[RNNParams::bias_hh] =
        context.requestWeight(bias_hh_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_hh", true);
    }
  }

  // We do not need this if we reuse net_hidden[0]. But if we do, then the unit
  // test will fail. Becuase it modifies the data during gradient calculation
  // TODO : We could control with something like #define test to save memory

  // hidden_state_dim : [ batch_size, 1, max_timestep, unit ]
  const TensorDim hidden_state_dim(batch_size, 1, max_timestep, unit);
  wt_idx[RNNParams::hidden_state] = context.requestTensor(
    hidden_state_dim, "hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  if (dropout_rate > epsilon) {
    // dropout_mask_dim = [ batch, 1, (return_sequences ? time_iteration : 1),
    // unit ]
    const TensorDim dropout_mask_dim(batch_size, 1,
                                     return_sequences ? max_timestep : 1, unit);
    wt_idx[RNNParams::dropout_mask] = context.requestTensor(
      dropout_mask_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);

  if (!acti_func.supportInPlace())
    throw exception::not_supported(
      "Out of place activation functions not supported");
}

void RNNLayer::setProperty(const std::vector<std::string> &values) {
  const std::vector<std::string> &remain_props =
    loadProperties(values, rnn_props);
  LayerImpl::setProperty(remain_props);
}

void RNNLayer::exportTo(Exporter &exporter, const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rnn_props, method, this);
}

void RNNLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(rnn_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(rnn_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnn_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(rnn_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();
  const unsigned int max_timestep = input_dim.height();
  const unsigned int feature_size = input_dim.width();
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const Tensor &weight_ih = context.getWeight(wt_idx[RNNParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[RNNParams::weight_hh]);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[RNNParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[RNNParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[RNNParams::bias_hh])
                      : empty;

  Tensor &hidden_state = context.getTensor(wt_idx[RNNParams::hidden_state]);

  // TODO: swap batch and timestep index with transpose
  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    Tensor input_slice = input.getBatchSlice(batch, 1);
    Tensor hidden_state_slice = hidden_state.getBatchSlice(batch, 1);

    for (unsigned int timestep = 0; timestep < max_timestep; ++timestep) {
      Tensor in = input_slice.getSharedDataTensor({feature_size},
                                                  timestep * feature_size);
      Tensor hs =
        hidden_state_slice.getSharedDataTensor({unit}, timestep * unit);

      in.dot(weight_ih, hs);
      if (!disable_bias) {
        if (integrate_bias) {
          hs.add_i(bias_h);
        } else {
          hs.add_i(bias_ih);
          hs.add_i(bias_hh);
        }
      }

      if (timestep) {
        Tensor prev_hs =
          hidden_state_slice.getSharedDataTensor({unit}, (timestep - 1) * unit);
        prev_hs.dot(weight_hh, hs, false, false, 1.0);
      }

      // In-place calculation for activation
      acti_func.run_fn(hs, hs);

      if (dropout_rate > epsilon && training) {
        Tensor dropout_mask = context.getTensor(wt_idx[RNNParams::dropout_mask])
                                .getBatchSlice(batch, 1);
        Tensor dropout_mask_t =
          dropout_mask.getSharedDataTensor({unit}, timestep * unit);
        dropout_mask_t.dropout_mask(dropout_rate);
        hs.multiply_i(dropout_mask_t);
      }
    }
  }

  if (!return_sequences) {
    for (unsigned int batch = 0; batch < input_dim.batch(); ++batch) {
      float *hidden_state_data = hidden_state.getAddress(
        batch * unit * max_timestep + (max_timestep - 1) * unit);
      float *output_data = output.getAddress(batch * unit);
      std::copy(hidden_state_data, hidden_state_data + unit, output_data);
    }
  } else {
    output.copy(hidden_state);
  }
}

void RNNLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &hidden_state_derivative =
    context.getTensorGrad(wt_idx[RNNParams::hidden_state]);
  const Tensor &weight = context.getWeight(wt_idx[RNNParams::weight_ih]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  hidden_state_derivative.dot(weight, outgoing_derivative, false, true);
}

void RNNLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(rnn_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(rnn_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnn_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(rnn_props).get();

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();
  const unsigned int max_timestep = input_dim.height();
  Tensor &incoming_derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &djdweight_ih = context.getWeightGrad(wt_idx[RNNParams::weight_ih]);
  Tensor &weight_hh = context.getWeight(wt_idx[RNNParams::weight_hh]);
  Tensor &djdweight_hh = context.getWeightGrad(wt_idx[RNNParams::weight_hh]);
  Tensor empty;
  Tensor &djdbias_h = !disable_bias && integrate_bias
                        ? context.getWeightGrad(wt_idx[RNNParams::bias_h])
                        : empty;
  Tensor &djdbias_ih = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[RNNParams::bias_ih])
                         : empty;
  Tensor &djdbias_hh = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[RNNParams::bias_hh])
                         : empty;

  Tensor &hidden_state_derivative =
    context.getTensorGrad(wt_idx[RNNParams::hidden_state]);

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

  if (!return_sequences) {
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      float *hidden_state_derivative_data = hidden_state_derivative.getAddress(
        batch * unit * max_timestep + (max_timestep - 1) * unit);
      float *incoming_derivative_data =
        incoming_derivative.getAddress(batch * unit);
      std::copy(incoming_derivative_data, incoming_derivative_data + unit,
                hidden_state_derivative_data);
    }
  } else {
    hidden_state_derivative.copy(incoming_derivative);
  }

  if (dropout_rate > epsilon) {
    hidden_state_derivative.multiply_i(
      context.getTensor(wt_idx[RNNParams::dropout_mask]));
  }

  Tensor &hidden_state = context.getTensor(wt_idx[RNNParams::hidden_state]);

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    Tensor deriv_t = hidden_state_derivative.getBatchSlice(batch, 1);
    Tensor input_t = input.getBatchSlice(batch, 1);
    Tensor hidden_state_t = hidden_state.getBatchSlice(batch, 1);

    for (unsigned int timestep = max_timestep; timestep-- > 0;) {
      Tensor dh = deriv_t.getSharedDataTensor(
        TensorDim(1, 1, 1, deriv_t.width()), timestep * deriv_t.width());
      Tensor xs = input_t.getSharedDataTensor(
        TensorDim(1, 1, 1, input_t.width()), timestep * input_t.width());
      Tensor hs = hidden_state_t.getSharedDataTensor(
        TensorDim(1, 1, 1, hidden_state_t.width()),
        timestep * hidden_state_t.width());

      acti_func.run_prime_fn(hs, dh, dh);
      if (!disable_bias) {
        if (integrate_bias) {
          djdbias_h.add_i(dh);
        } else {
          djdbias_ih.add_i(dh);
          djdbias_hh.add_i(dh);
        }
      }
      xs.dot(dh, djdweight_ih, true, false, 1.0);

      if (timestep) {
        Tensor prev_hs = hidden_state_t.getSharedDataTensor(
          TensorDim(1, 1, 1, hidden_state_t.width()),
          (timestep - 1) * hidden_state_t.width());
        Tensor dh_t_1 =
          deriv_t.getSharedDataTensor(TensorDim(1, 1, 1, deriv_t.width()),
                                      (timestep - 1) * deriv_t.width());
        prev_hs.dot(dh, djdweight_hh, true, false, 1.0);
        dh.dot(weight_hh, dh_t_1, false, true, 1.0);
      }
    }
  }
}

void RNNLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[RNNParams::hidden_state], batch);

  if (std::get<props::DropOutRate>(rnn_props).get() > epsilon) {
    context.updateTensor(wt_idx[RNNParams::dropout_mask], batch);
  }
}

} // namespace nntrainer
