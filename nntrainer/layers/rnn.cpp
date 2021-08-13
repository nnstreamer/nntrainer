// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
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
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <rnn.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// - weight_xh ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) ]
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size)]
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size)]
enum RNNParams { weight_xh, weight_hh, bias_h, hidden_state, dropout_mask };

RNNLayer::RNNLayer() :
  LayerImpl(),
  rnn_props(props::Unit(), props::HiddenStateActivation(),
            props::ReturnSequences(), props::DropOutRate()),
  wt_idx({0}),
  acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {}

void RNNLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);

  auto unit = std::get<props::Unit>(rnn_props).get();
  auto &hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(rnn_props);
  bool return_sequences = std::get<props::ReturnSequences>(rnn_props);
  float dropout_rate = std::get<props::DropOutRate>(rnn_props);

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("RNN layer takes only one input");
  }

  TensorDim output_dim;
  const TensorDim &input_dim = context.getInputDimensions()[0];

  // input_dim = [ batch, 1, time_iteration, feature_size ]
  // outut_dim = [ batch, 1, time_iteration, hidden_size ( unit ) ]
  output_dim = input_dim;
  output_dim.width(unit);

  if (dropout_rate > epsilon) {
    wt_idx[RNNParams::dropout_mask] = context.requestTensor(
      output_dim, context.getName() + ":dropout_mask",
      Tensor::Initializer::NONE, false, TensorLifespan::ITERATION_LIFESPAN);
  }

  if (!return_sequences) {
    output_dim.height(1u);
  }

  context.setOutputDimensions({output_dim});

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit);

  TensorDim dim_xh = output_dim;
  dim_xh.height(input_dim.width());
  dim_xh.batch(1);

  TensorDim dim_hh = output_dim;
  dim_hh.height(unit);
  dim_hh.batch(1);

  // weight_initializer can be set sepeartely. weight_xh initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  wt_idx[RNNParams::weight_xh] = context.requestWeight(
    dim_xh, weight_initializer, weight_regularizer, weight_regularizer_constant,
    context.getName() + ":weight_xh", true);
  wt_idx[RNNParams::weight_hh] = context.requestWeight(
    dim_hh, weight_initializer, weight_regularizer, weight_regularizer_constant,
    context.getName() + ":weight_hh", true);
  wt_idx[RNNParams::bias_h] =
    context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                          1.0f, context.getName() + ":bias_h", true);

  // We do not need this if we reuse net_hidden[0]. But if we do, then the unit
  // test will fail. Becuase it modifies the date during gradient calculation
  // TODO : We could control with something like #define test to save memory
  TensorDim d = input_dim;
  d.width(unit);
  wt_idx[RNNParams::hidden_state] = context.requestTensor(
    d, context.getName() + ":hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  if (hidden_state_activation_type.get() == ActivationType::ACT_NONE) {
    hidden_state_activation_type.set(ActivationType::ACT_TANH);
  }
  acti_func.setActiFunc(hidden_state_activation_type.get());

  if (!acti_func.supportInPlace())
    throw exception::not_supported(
      "Out of place activation functions not supported");
}

void RNNLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, rnn_props);
  LayerImpl::setProperty(remain_props);
}

void RNNLayer::exportTo(Exporter &exporter, const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rnn_props, method, this);
}

void RNNLayer::forwarding(RunLayerContext &context, bool training) {
  bool return_sequences = std::get<props::ReturnSequences>(rnn_props);
  float dropout_rate = std::get<props::DropOutRate>(rnn_props);

  Tensor &weight_xh = context.getWeight(wt_idx[RNNParams::weight_xh]);
  Tensor &weight_hh = context.getWeight(wt_idx[RNNParams::weight_hh]);
  Tensor &bias_h = context.getWeight(wt_idx[RNNParams::bias_h]);

  Tensor &hidden_ = context.getTensor(wt_idx[RNNParams::hidden_state]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input_.getDim();

  // TODO: swap b and t index with transpose
  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);

    for (unsigned int t = 0; t < islice.height(); ++t) {
      Tensor xs =
        islice.getSharedDataTensor({islice.width()}, t * islice.width());
      Tensor hs =
        oslice.getSharedDataTensor({oslice.width()}, t * oslice.width());

      xs.dot(weight_xh, hs);
      hs.add_i(bias_h);

      if (t > 0) {
        Tensor hs_prev = oslice.getSharedDataTensor({oslice.width()},
                                                    (t - 1) * oslice.width());
        hs_prev.dot(weight_hh, hs, false, false, 1.0);
      }

      // In-place calculation for activation
      acti_func.run_fn(hs, hs);

      if (dropout_rate > epsilon && training) {
        Tensor mask_ = context.getTensor(wt_idx[RNNParams::dropout_mask])
                         .getBatchSlice(b, 1);
        Tensor msk =
          mask_.getSharedDataTensor({mask_.width()}, t * mask_.width());
        msk = hs.dropout_mask(dropout_rate);
        hs.multiply_i(msk);
      }
    }
  }

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  if (!return_sequences) {
    TensorDim d = hidden_.getDim();
    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      float *data = hidden_.getAddress(b * d.width() * d.height() +
                                       (d.height() - 1) * d.width());
      float *rdata = output.getAddress(b * d.width());
      std::copy(data, data + d.width(), rdata);
    }
  } else {
    output.copy(hidden_);
  }
}

void RNNLayer::calcDerivative(RunLayerContext &context) {
  Tensor &derivative_ = context.getTensorGrad(wt_idx[RNNParams::hidden_state]);
  Tensor &weight = context.getWeight(wt_idx[RNNParams::weight_xh]);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  derivative_.dot(weight, ret_, false, true);
}

void RNNLayer::calcGradient(RunLayerContext &context) {
  bool return_sequences = std::get<props::ReturnSequences>(rnn_props);
  float dropout_rate = std::get<props::DropOutRate>(rnn_props);

  Tensor &djdw_x = context.getWeightGrad(wt_idx[RNNParams::weight_xh]);
  Tensor &djdw_h = context.getWeightGrad(wt_idx[RNNParams::weight_hh]);
  Tensor &djdb_h = context.getWeightGrad(wt_idx[RNNParams::bias_h]);
  Tensor &weight_hh = context.getWeight(wt_idx[RNNParams::weight_hh]);

  Tensor &derivative_ = context.getTensorGrad(wt_idx[RNNParams::hidden_state]);
  Tensor &incoming_deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input_.getDim();

  djdw_x.setZero();
  djdw_h.setZero();
  djdb_h.setZero();
  derivative_.setZero();

  if (!return_sequences) {
    TensorDim d = derivative_.getDim();
    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      float *data = derivative_.getAddress(b * d.width() * d.height() +
                                           (d.height() - 1) * d.width());
      float *rdata = incoming_deriv.getAddress(b * d.width());
      std::copy(rdata, rdata + d.width(), data);
    }
  } else {
    derivative_.copy(incoming_deriv);
  }

  if (dropout_rate > epsilon) {
    derivative_.multiply_i(context.getTensor(wt_idx[RNNParams::dropout_mask]));
  }

  Tensor &hidden_ = context.getTensor(wt_idx[RNNParams::hidden_state]);

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor deriv_t = derivative_.getBatchSlice(b, 1);
    Tensor xs_t = input_.getBatchSlice(b, 1);
    Tensor hs_t = hidden_.getBatchSlice(b, 1);

    for (unsigned int t = deriv_t.height(); t-- > 0;) {
      Tensor dh = deriv_t.getSharedDataTensor(
        TensorDim(1, 1, 1, deriv_t.width()), t * deriv_t.width());
      Tensor xs = xs_t.getSharedDataTensor(TensorDim(1, 1, 1, xs_t.width()),
                                           t * xs_t.width());
      Tensor hs = hs_t.getSharedDataTensor(TensorDim(1, 1, 1, hs_t.width()),
                                           t * hs_t.width());

      acti_func.run_prime_fn(hs, dh, dh);
      djdb_h.add_i(dh);
      xs.dot(dh, djdw_x, true, false, 1.0);

      if (t > 0) {
        Tensor hs_prev = hs_t.getSharedDataTensor(
          TensorDim(1, 1, 1, hs_t.width()), (t - 1) * hs_t.width());
        Tensor dh_t_1 = deriv_t.getSharedDataTensor(
          TensorDim(1, 1, 1, deriv_t.width()), (t - 1) * deriv_t.width());
        hs_prev.dot(dh, djdw_h, true, false, 1.0);
        dh.dot(weight_hh, dh_t_1, false, true, 1.0);
      }
    }
  }
}

} // namespace nntrainer
