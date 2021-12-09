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

#include <cmath>
#include <layer_context.h>
#include <lstm.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

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
  lstm_props(props::Unit(),
             props::HiddenStateActivation() = ActivationType::ACT_TANH,
             props::RecurrentActivation() = ActivationType::ACT_SIGMOID,
             props::ReturnSequences(), props::DropOutRate(),
             props::IntegrateBias(), props::MaxTimestep(), props::Timestep()),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

// - weight_ih ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size) x NUM_GATE] -> f, g, i, o
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
  const ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(lstm_props).get();
  const ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(lstm_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(lstm_props).get();

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("LSTM layer takes only one input");
  }

  // input_dim = [ batch, 1, time_iteration, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  const unsigned int batch_size = input_dim.batch();
  unsigned int max_timestep = input_dim.height();
  if (!std::get<props::MaxTimestep>(lstm_props).empty())
    max_timestep =
      std::max(max_timestep, std::get<props::MaxTimestep>(lstm_props).get());
  std::get<props::MaxTimestep>(lstm_props).set(max_timestep);
  const unsigned int feature_size = input_dim.width();

  // if return_sequences == false :
  //      output_dim = [ batch, 1, 1, unit ]
  // else:
  //      output_dim = [ batch, 1, time_iteration, unit ]
  const TensorDim output_dim(batch_size, 1, return_sequences ? max_timestep : 1,
                             unit);
  context.setOutputDimensions({output_dim});

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // weight_ih_dim : [ 1, 1, feature_size, NUM_GATE * unit ]
  const TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[LSTMParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // weight_hh_dim : [ 1, 1, unit, NUM_GATE * unit ]
  const TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[LSTMParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // bias_h_dim : [ 1, 1, 1, NUM_GATE * unit ]
      const TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[LSTMParams::bias_h] =
        context.requestWeight(bias_h_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_h", true);
    } else {
      // bias_ih_dim : [ 1, 1, 1, NUM_GATE * unit ]
      const TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[LSTMParams::bias_ih] =
        context.requestWeight(bias_ih_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_ih", true);
      // bias_hh_dim : [ 1, 1, 1, NUM_GATE * unit ]
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
    TensorLifespan::ITERATION_LIFESPAN, false);
  // cell_state_dim : [ batch_size, 1, max_timestep, unit ]
  const TensorDim cell_state_dim(batch_size, 1, max_timestep, unit);
  wt_idx[LSTMParams::cell_state] = context.requestTensor(
    cell_state_dim, "cell_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);
  // ifgo_dim : [ batch_size, 1, max_timestep, NUM_GATE * unit ]
  const TensorDim ifgo_dim(batch_size, 1, max_timestep, NUM_GATE * unit);
  wt_idx[LSTMParams::ifgo] =
    context.requestTensor(ifgo_dim, "ifgo", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN, false);

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
  auto remain_props = loadProperties(values, lstm_props);
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
  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(lstm_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstm_props).get();
  const props::Timestep timestep = std::get<props::Timestep>(lstm_props);

  unsigned int start_timestep = 0;
  unsigned int end_timestep = max_timestep;
  if (!timestep.empty()) {
    const unsigned int current_timestep = timestep.get();
    if (current_timestep >= end_timestep)
      throw std::runtime_error("Timestep to run exceeds input dimensions");

    start_timestep = current_timestep;
    end_timestep = current_timestep + 1;
  }

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();

  const Tensor &weight_ih = context.getWeight(wt_idx[LSTMParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[LSTMParams::weight_hh]);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[LSTMParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[LSTMParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[LSTMParams::bias_hh])
                      : empty;

  Tensor &hidden_state = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &cell_state = context.getTensor(wt_idx[LSTMParams::cell_state]);
  Tensor &ifgo = context.getTensor(wt_idx[LSTMParams::ifgo]);

  if (!start_timestep) {
    hidden_state.setZero();
    cell_state.setZero();
  }

  /**
   * @note when the recurrent realization happens, different instances of lstm
   * will share the weights, hidden state, cell and ifgo memory. However, they
   * do not share the input, output and derivatives memory. The input/output
   * will be contain a single timestep data only.
   */

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    Tensor input_batch = input.getBatchSlice(batch, 1);
    Tensor hidden_state_batch = hidden_state.getBatchSlice(batch, 1);
    Tensor cell_state_batch = cell_state.getBatchSlice(batch, 1);
    Tensor ifgo_batch = ifgo.getBatchSlice(batch, 1);

    for (unsigned int t = start_timestep; t < end_timestep; ++t) {
      Tensor in;
      if (input_batch.height() != 1)
        in = input_batch.getSharedDataTensor({feature_size}, t * feature_size);
      else
        in = input_batch;

      Tensor hs = hidden_state_batch.getSharedDataTensor({unit}, t * unit);
      Tensor cs = cell_state_batch.getSharedDataTensor({unit}, t * unit);
      Tensor ifgo_t =
        ifgo_batch.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      in.dot(weight_ih, ifgo_t);
      if (!disable_bias) {
        if (integrate_bias) {
          ifgo_t.add_i(bias_h);
        } else {
          ifgo_t.add_i(bias_ih);
          ifgo_t.add_i(bias_hh);
        }
      }

      if (t) {
        Tensor prev_hs =
          hidden_state_batch.getSharedDataTensor({unit}, (t - 1) * unit);
        prev_hs.dot(weight_hh, ifgo_t, false, false, 1.0);
      }

      Tensor hi = ifgo_t.getSharedDataTensor({unit}, 0);
      Tensor hf = ifgo_t.getSharedDataTensor({unit}, unit);
      Tensor hg = ifgo_t.getSharedDataTensor({unit}, unit * 2);
      Tensor ho = ifgo_t.getSharedDataTensor({unit}, unit * 3);

      recurrent_acti_func.run_fn(hf, hf);
      recurrent_acti_func.run_fn(hi, hi);
      recurrent_acti_func.run_fn(ho, ho);
      acti_func.run_fn(hg, hg);

      if (t) {
        Tensor prev_cs =
          cell_state_batch.getSharedDataTensor({unit}, (t - 1) * unit);
        hf.multiply(prev_cs, cs);
      }
      cs.add_i(hg.multiply(hi));

      acti_func.run_fn(cs, hs);
      hs.multiply_i(ho);

      if (dropout_rate > epsilon && training) {
        Tensor mask_ = context.getTensor(wt_idx[LSTMParams::dropout_mask])
                         .getBatchSlice(batch, 1);
        Tensor msk =
          mask_.getSharedDataTensor({mask_.width()}, t * mask_.width());
        msk.dropout_mask(dropout_rate);
        hs.multiply_i(msk);
      }
    }
  }

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  if (start_timestep == 0 && end_timestep == max_timestep && return_sequences) {
    std::copy(hidden_state.getData(),
              hidden_state.getData() + hidden_state.size(), output.getData());
  } else {
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      float *hidden_state_data = hidden_state.getAddress(
        batch * unit * max_timestep + (end_timestep - 1) * unit);
      float *output_data = output.getAddress(batch * unit);
      std::copy(hidden_state_data, hidden_state_data + unit, output_data);
    }
  }
}

void LSTMLayer::calcDerivative(RunLayerContext &context) {
  /** get the timestep values */
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstm_props).get();
  const props::Timestep timestep = std::get<props::Timestep>(lstm_props);

  unsigned int start_timestep = 0;
  unsigned int end_timestep = max_timestep;
  if (!timestep.empty()) {
    const unsigned int cur_timestep = timestep.get();
    if (cur_timestep >= end_timestep)
      throw std::runtime_error("Timestep to run exceeds input dimensions");

    start_timestep = cur_timestep;
    end_timestep = cur_timestep + 1;
  }
  const unsigned int timestep_diff = end_timestep - start_timestep;

  Tensor &ifgo_derivative = context.getTensorGrad(wt_idx[LSTMParams::ifgo]);
  Tensor &weight_ih = context.getWeight(wt_idx[LSTMParams::weight_ih]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (start_timestep == 0 && end_timestep == max_timestep) {
    /**
     * this if is only for optimization purpose. The else calculates for
     * this scenario as well.
     */
    ifgo_derivative.dot(weight_ih, outgoing_derivative, false, true);
  } else {
    for (unsigned int b = 0; b < ifgo_derivative.batch(); ++b) {
      Tensor deriv_t = ifgo_derivative.getBatchSlice(b, 1);
      Tensor ret_deriv_t = outgoing_derivative.getBatchSlice(b, 1);
      Tensor dt, rdt;

      if (deriv_t.height() != 1) {
        dt = deriv_t.getSharedDataTensor({timestep_diff, deriv_t.width()},
                                         start_timestep * deriv_t.width());
      } else {
        dt = deriv_t;
      }

      if (ret_deriv_t.height() != 1) {
        rdt =
          ret_deriv_t.getSharedDataTensor({timestep_diff, ret_deriv_t.width()},
                                          start_timestep * ret_deriv_t.width());
      } else {
        rdt = ret_deriv_t;
      }

      dt.dot(weight_ih, rdt, false, true);
    }
  }
}

void LSTMLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstm_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(lstm_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(lstm_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(lstm_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(lstm_props).get();
  const props::Timestep timestep = std::get<props::Timestep>(lstm_props);

  unsigned int start_timestep = max_timestep - 1;
  int end_timestep = -1;
  if (!timestep.empty()) {
    const unsigned int cur_timestep = timestep.get();
    NNTR_THROW_IF(cur_timestep > start_timestep, std::runtime_error)
      << "Timestep to run exceeds input dimension current timestep"
      << cur_timestep << "start_timestep" << start_timestep;
    start_timestep = cur_timestep;
    end_timestep = cur_timestep - 1;
  }

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  Tensor &incoming_derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &djdweight_ih = context.getWeightGrad(wt_idx[LSTMParams::weight_ih]);
  Tensor &weight_hh = context.getWeight(wt_idx[LSTMParams::weight_hh]);
  Tensor &djdweight_hh = context.getWeightGrad(wt_idx[LSTMParams::weight_hh]);
  Tensor empty;
  Tensor &djdbias_h = !disable_bias && integrate_bias
                        ? context.getWeightGrad(wt_idx[LSTMParams::bias_h])
                        : empty;
  Tensor &djdbias_ih = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[LSTMParams::bias_ih])
                         : empty;
  Tensor &djdbias_hh = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[LSTMParams::bias_hh])
                         : empty;

  Tensor &hidden_state = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &hidden_state_derivative =
    context.getTensorGrad(wt_idx[LSTMParams::hidden_state]);
  Tensor &cell_state = context.getTensor(wt_idx[LSTMParams::cell_state]);
  Tensor &cell_state_derivative =
    context.getTensorGrad(wt_idx[LSTMParams::cell_state]);
  Tensor &ifgo = context.getTensor(wt_idx[LSTMParams::ifgo]);
  Tensor &ifgo_derivative = context.getTensorGrad(wt_idx[LSTMParams::ifgo]);

  if (start_timestep + 1 == max_timestep) {
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

    cell_state_derivative.setZero();
    hidden_state_derivative.setZero();
    ifgo_derivative.setZero();
  }

  if (start_timestep == max_timestep - 1 && end_timestep == -1 &&
      return_sequences) {
    std::copy(incoming_derivative.getData(),
              incoming_derivative.getData() + incoming_derivative.size(),
              hidden_state_derivative.getData());
  } else {
    TensorDim d = hidden_state_derivative.getDim();
    for (unsigned int batch = 0; batch < input_dim.batch(); ++batch) {
      Tensor data = hidden_state_derivative.getSharedDataTensor(
        {unit}, batch * unit * max_timestep + start_timestep * unit);

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
    hidden_state_derivative.multiply_i(
      context.getTensor(wt_idx[LSTMParams::dropout_mask]));
  }

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor deriv_t = hidden_state_derivative.getBatchSlice(b, 1);
    Tensor derivc_t = cell_state_derivative.getBatchSlice(b, 1);
    Tensor xs_t = input.getBatchSlice(b, 1);
    Tensor hs_t = hidden_state.getBatchSlice(b, 1);
    Tensor cs_t = cell_state.getBatchSlice(b, 1);

    Tensor dh;
    Tensor xs;
    Tensor hs_prev;
    Tensor cs_prev;
    Tensor cs;
    Tensor dc;
    Tensor difgo_ = ifgo_derivative.getBatchSlice(b, 1);
    Tensor ifgo_ = ifgo.getBatchSlice(b, 1);

    for (int t = start_timestep; t > end_timestep; t--) {
      dh = deriv_t.getSharedDataTensor({deriv_t.width()}, t * deriv_t.width());

      dc =
        derivc_t.getSharedDataTensor({derivc_t.width()}, t * derivc_t.width());

      if (xs_t.height() != 1)
        xs = xs_t.getSharedDataTensor({xs_t.width()}, t * xs_t.width());
      else
        xs = xs_t;

      cs = cs_t.getSharedDataTensor({cs_t.width()}, t * cs_t.width());

      Tensor difgo_t =
        difgo_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);
      Tensor ifgo_t =
        ifgo_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      if (t == 0) {
        hs_prev = Tensor(hs_t.width());
        hs_prev.setZero();
        cs_prev = Tensor(cs_t.width());
        cs_prev.setZero();
      } else {
        hs_prev =
          hs_t.getSharedDataTensor({hs_t.width()}, (t - 1) * hs_t.width());
        cs_prev =
          cs_t.getSharedDataTensor({cs_t.width()}, (t - 1) * cs_t.width());
      }

      Tensor dhi = difgo_t.getSharedDataTensor({unit}, 0);
      Tensor dhf = difgo_t.getSharedDataTensor({unit}, unit);
      Tensor dhg = difgo_t.getSharedDataTensor({unit}, unit * 2);
      Tensor dho = difgo_t.getSharedDataTensor({unit}, unit * 3);

      Tensor hi = ifgo_t.getSharedDataTensor({unit}, 0);
      Tensor hf = ifgo_t.getSharedDataTensor({unit}, unit);
      Tensor hg = ifgo_t.getSharedDataTensor({unit}, unit * 2);
      Tensor ho = ifgo_t.getSharedDataTensor({unit}, unit * 3);

      acti_func.run_fn(cs, dho);
      dho.multiply_i(dh);
      acti_func.run_fn(cs, cs);

      if ((unsigned)t + 1 == max_timestep) {
        acti_func.run_prime_fn(cs, dc, dh);
        dc.multiply_i(ho);
      } else {
        /// @todo optimize this by updating run_prime_fn to accumulate or make
        /// it inplace somehow
        Tensor dc_temp(dc.getDim());
        acti_func.run_prime_fn(cs, dc_temp, dh);
        dc_temp.multiply_i(ho);
        dc.add_i(dc_temp);
      }

      if (t > 0) {
        Tensor dc_nx = derivc_t.getSharedDataTensor({derivc_t.width()},
                                                    (t - 1) * derivc_t.width());
        dc.multiply(hf, dc_nx);
      }

      dc.multiply(cs_prev, dhf);
      dc.multiply(hg, dhi);
      dc.multiply(hi, dhg);

      recurrent_acti_func.run_prime_fn(ho, dho, dho);
      recurrent_acti_func.run_prime_fn(hf, dhf, dhf);
      recurrent_acti_func.run_prime_fn(hi, dhi, dhi);
      acti_func.run_prime_fn(hg, dhg, dhg);
      if (!disable_bias) {
        if (integrate_bias) {
          djdbias_h.add_i(difgo_t);
        } else {
          djdbias_ih.add_i(difgo_t);
          djdbias_hh.add_i(difgo_t);
        }
      }
      djdweight_ih.add_i(xs.dot(difgo_t, true, false));
      djdweight_hh.add_i(hs_prev.dot(difgo_t, true, false));
      if (t > 0) {
        Tensor dh_nx = deriv_t.getSharedDataTensor({deriv_t.width()},
                                                   (t - 1) * deriv_t.width());
        difgo_t.dot(weight_hh, dh_nx, false, true, 1.0f);
      }
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
