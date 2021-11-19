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
  weight_xh,
  weight_hh,
  bias_h,
  hidden_state,
  mem_cell,
  fgio,
  dropout_mask
};

#define NUM_GATE 4

LSTMLayer::LSTMLayer() :
  LayerImpl(),
  lstm_props(props::Unit(), props::HiddenStateActivation(),
             props::RecurrentActivation(), props::ReturnSequences(),
             props::DropOutRate(), props::MaxTimestep(), props::Timestep()),
  wt_idx({0}),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {}

// - weight_xh ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size) x NUM_GATE] -> f, g, i, o
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size) x NUM_GATE] -> f, g, i, o
void LSTMLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);

  NNTR_THROW_IF(std::get<props::Unit>(lstm_props).empty(),
                std::invalid_argument)
    << "unit property missing for lstm layer";
  auto unit = std::get<props::Unit>(lstm_props).get();
  auto &hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(lstm_props);
  auto &recurrent_activation_type =
    std::get<props::RecurrentActivation>(lstm_props);
  bool return_sequences = std::get<props::ReturnSequences>(lstm_props);
  float dropout_rate = std::get<props::DropOutRate>(lstm_props);

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("LSTM layer takes only one input");
  }

  TensorDim output_dim;
  const TensorDim &input_dim = context.getInputDimensions()[0];

  // input_dim = [ batch, 1, time_iteration, feature_size ]
  // if return_sequences == False :
  //      output_dim = [ batch, 1, 1, hidden_size (unit)]
  // else:
  //      output_dim = [ batch, 1, time_iteration, hidden_size ( unit ) ]
  output_dim = input_dim;
  output_dim.width(unit);

  if (dropout_rate > epsilon) {
    wt_idx[LSTMParams::dropout_mask] = context.requestTensor(
      output_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  if (!return_sequences) {
    output_dim.height(1);
  }

  context.setOutputDimensions({output_dim});

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit * NUM_GATE);

  TensorDim dim_xh = output_dim;
  dim_xh.height(input_dim.width());
  dim_xh.width(unit * NUM_GATE);
  dim_xh.batch(1);

  TensorDim dim_hh = output_dim;
  dim_hh.height(unit);
  dim_hh.width(unit * NUM_GATE);
  dim_hh.batch(1);

  // weight_initializer can be set sepeartely. weight_xh initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.
  wt_idx[LSTMParams::weight_xh] =
    context.requestWeight(dim_xh, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_xh", true);
  wt_idx[LSTMParams::weight_hh] =
    context.requestWeight(dim_hh, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  wt_idx[LSTMParams::bias_h] = context.requestWeight(
    bias_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, "bias_h", true);

  unsigned int max_timestep = input_dim.height();
  if (!std::get<props::MaxTimestep>(lstm_props).empty())
    max_timestep =
      std::max(max_timestep, std::get<props::MaxTimestep>(lstm_props).get());
  std::get<props::MaxTimestep>(lstm_props).set(max_timestep);

  TensorDim d = input_dim;
  d.height(max_timestep);
  d.width(unit);

  wt_idx[LSTMParams::hidden_state] =
    context.requestTensor(d, "hidden_state", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);
  wt_idx[LSTMParams::mem_cell] =
    context.requestTensor(d, "mem_cell", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  d.width(unit * NUM_GATE);
  wt_idx[LSTMParams::fgio] =
    context.requestTensor(d, "fgio", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  if (hidden_state_activation_type.get() == ActivationType::ACT_NONE) {
    hidden_state_activation_type.set(ActivationType::ACT_TANH);
  }
  acti_func.setActiFunc(hidden_state_activation_type.get());

  if (recurrent_activation_type.get() == ActivationType::ACT_NONE) {
    recurrent_activation_type.set(ActivationType::ACT_SIGMOID);
  }
  recurrent_acti_func.setActiFunc(recurrent_activation_type.get());
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
  auto unit = std::get<props::Unit>(lstm_props).get();
  bool return_sequences = std::get<props::ReturnSequences>(lstm_props);
  float dropout_rate = std::get<props::DropOutRate>(lstm_props);

  Tensor &weight_xh = context.getWeight(wt_idx[LSTMParams::weight_xh]);
  Tensor &weight_hh = context.getWeight(wt_idx[LSTMParams::weight_hh]);
  Tensor &bias_h = context.getWeight(wt_idx[LSTMParams::bias_h]);

  Tensor &hidden_ = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &m_cell_ = context.getTensor(wt_idx[LSTMParams::mem_cell]);
  Tensor &fgio = context.getTensor(wt_idx[LSTMParams::fgio]);
  const TensorDim &input_dim = input_.getDim();

  unsigned int max_timestep = std::get<props::MaxTimestep>(lstm_props).get();
  unsigned int start_timestep = 0;
  unsigned int end_timestep = max_timestep;

  auto ts = std::get<props::Timestep>(lstm_props);
  if (!ts.empty()) {
    auto cur_ts = ts.get();
    if (cur_ts >= end_timestep)
      throw std::runtime_error("Timestep to run exceeds input dimensions");

    start_timestep = cur_ts;
    end_timestep = cur_ts + 1;
  }

  if (start_timestep == 0) {
    hidden_.setZero();
    m_cell_.setZero();
  }

  /**
   * @note when the recurrent realization happens, different instances of lstm
   * will share the weights, hidden state, cell and fgio memory. However, they
   * do not share the input, output and derivatives memory. The input/output
   * will be contain a single timestep data only.
   */

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);
    Tensor cell = m_cell_.getBatchSlice(b, 1);
    Tensor fgio_ = fgio.getBatchSlice(b, 1);

    for (unsigned int t = start_timestep; t < end_timestep; ++t) {
      Tensor xs;
      if (islice.height() != 1)
        xs = islice.getSharedDataTensor({islice.width()}, t * islice.width());
      else
        xs = islice;

      Tensor hs =
        oslice.getSharedDataTensor({oslice.width()}, t * oslice.width());
      Tensor cs = cell.getSharedDataTensor({cell.width()}, t * cell.width());
      Tensor fgio_t =
        fgio_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      xs.dot(weight_xh, fgio_t);
      fgio_t.add_i(bias_h);

      if (t > 0) {
        Tensor hs_prev = oslice.getSharedDataTensor({oslice.width()},
                                                    (t - 1) * oslice.width());
        hs_prev.dot(weight_hh, fgio_t, false, false, 1.0);
      }

      Tensor hi = fgio_t.getSharedDataTensor({unit}, 0);
      Tensor hf = fgio_t.getSharedDataTensor({unit}, unit);
      Tensor hg = fgio_t.getSharedDataTensor({unit}, unit * 2);
      Tensor ho = fgio_t.getSharedDataTensor({unit}, unit * 3);

      recurrent_acti_func.run_fn(hf, hf);
      recurrent_acti_func.run_fn(hi, hi);
      recurrent_acti_func.run_fn(ho, ho);
      acti_func.run_fn(hg, hg);

      if (t > 0) {
        Tensor cs_prev =
          cell.getSharedDataTensor({cell.width()}, (t - 1) * cell.width());
        hf.multiply(cs_prev, cs);
      }
      cs.add_i(hg.multiply(hi));

      acti_func.run_fn(cs, hs);
      hs.multiply_i(ho);

      if (dropout_rate > epsilon && training) {
        Tensor mask_ = context.getTensor(wt_idx[LSTMParams::dropout_mask])
                         .getBatchSlice(b, 1);
        Tensor msk =
          mask_.getSharedDataTensor({mask_.width()}, t * mask_.width());
        msk.dropout_mask(dropout_rate);
        hs.multiply_i(msk);
      }
    }
  }

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  if (start_timestep == 0 && end_timestep == max_timestep && return_sequences) {
    std::copy(hidden_.getData(), hidden_.getData() + hidden_.size(),
              output.getData());
  } else {
    TensorDim d = hidden_.getDim();
    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      float *data = hidden_.getAddress(b * d.width() * d.height() +
                                       (end_timestep - 1) * d.width());
      float *rdata = output.getAddress(b * d.width());
      std::copy(data, data + d.width(), rdata);
    }
  }
}

void LSTMLayer::calcDerivative(RunLayerContext &context) {
  Tensor &derivative_ = context.getTensorGrad(wt_idx[LSTMParams::fgio]);
  Tensor &weight = context.getWeight(wt_idx[LSTMParams::weight_xh]);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  /** get the timestep values */
  unsigned int max_timestep = std::get<props::MaxTimestep>(lstm_props).get();
  unsigned int start_timestep = 0;
  unsigned int end_timestep = max_timestep;

  auto ts = std::get<props::Timestep>(lstm_props);
  if (!ts.empty()) {
    auto cur_ts = ts.get();
    if (cur_ts >= end_timestep)
      throw std::runtime_error("Timestep to run exceeds input dimensions");

    start_timestep = cur_ts;
    end_timestep = cur_ts + 1;
  }

  unsigned int timestep_diff = end_timestep - start_timestep;
  if (start_timestep == 0 && end_timestep == max_timestep) {
    /**
     * this if is only for optimization purpose. The else calculates for
     * this scenario as well.
     */
    derivative_.dot(weight, ret_, false, true);
  } else {
    for (unsigned int b = 0; b < derivative_.batch(); ++b) {
      Tensor deriv_t = derivative_.getBatchSlice(b, 1);
      Tensor ret_deriv_t = ret_.getBatchSlice(b, 1);
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

      dt.dot(weight, rdt, false, true);
    }
  }
}

void LSTMLayer::calcGradient(RunLayerContext &context) {
  auto unit = std::get<props::Unit>(lstm_props).get();
  bool return_sequences = std::get<props::ReturnSequences>(lstm_props);
  float dropout_rate = std::get<props::DropOutRate>(lstm_props);

  Tensor &djdw_x = context.getWeightGrad(wt_idx[LSTMParams::weight_xh]);
  Tensor &djdw_h = context.getWeightGrad(wt_idx[LSTMParams::weight_hh]);
  Tensor &djdb_h = context.getWeightGrad(wt_idx[LSTMParams::bias_h]);
  Tensor &weight_hh = context.getWeight(wt_idx[LSTMParams::weight_hh]);

  Tensor &derivative_ = context.getTensorGrad(wt_idx[LSTMParams::hidden_state]);
  Tensor &hidden_ = context.getTensor(wt_idx[LSTMParams::hidden_state]);
  Tensor &incoming_deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input_.getDim();
  Tensor &m_cell_ = context.getTensor(wt_idx[LSTMParams::mem_cell]);
  Tensor &dm_cell_ = context.getTensorGrad(wt_idx[LSTMParams::mem_cell]);
  Tensor &fgio = context.getTensor(wt_idx[LSTMParams::fgio]);
  Tensor &d_fgio = context.getTensorGrad(wt_idx[LSTMParams::fgio]);

  /** get the timestep values */
  unsigned int max_timestep = std::get<props::MaxTimestep>(lstm_props).get();
  unsigned int start_timestep = max_timestep - 1;
  int end_timestep = -1;

  auto ts = std::get<props::Timestep>(lstm_props);
  if (!ts.empty()) {
    auto cur_ts = ts.get();
    NNTR_THROW_IF(cur_ts > start_timestep, std::runtime_error)
      << "Timestep to run exceeds input dimension current timestep" << cur_ts
      << "start_timestep" << start_timestep;
    start_timestep = cur_ts;
    end_timestep = cur_ts - 1;
  }

  if (start_timestep + 1 == max_timestep) {
    djdw_x.setZero();
    djdw_h.setZero();
    djdb_h.setZero();

    dm_cell_.setZero();
    derivative_.setZero();
    d_fgio.setZero();
  }

  if (start_timestep == max_timestep - 1 && end_timestep == -1 &&
      return_sequences) {
    std::copy(incoming_deriv.getData(),
              incoming_deriv.getData() + incoming_deriv.size(),
              derivative_.getData());
  } else {
    TensorDim d = derivative_.getDim();
    for (unsigned int b = 0; b < input_dim.batch(); ++b) {
      Tensor data = derivative_.getSharedDataTensor(
        {d.width()}, b * d.width() * d.height() + start_timestep * d.width());

      Tensor rdata =
        incoming_deriv.getSharedDataTensor({d.width()}, b * d.width());
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
    derivative_.multiply_i(context.getTensor(wt_idx[LSTMParams::dropout_mask]));
  }

  for (unsigned int b = 0; b < input_dim.batch(); ++b) {
    Tensor deriv_t = derivative_.getBatchSlice(b, 1);
    Tensor derivc_t = dm_cell_.getBatchSlice(b, 1);
    Tensor xs_t = input_.getBatchSlice(b, 1);
    Tensor hs_t = hidden_.getBatchSlice(b, 1);
    Tensor cs_t = m_cell_.getBatchSlice(b, 1);

    Tensor dh;
    Tensor xs;
    Tensor hs_prev;
    Tensor cs_prev;
    Tensor cs;
    Tensor dc;
    Tensor dfgio_ = d_fgio.getBatchSlice(b, 1);
    Tensor fgio_ = fgio.getBatchSlice(b, 1);

    for (int t = start_timestep; t > end_timestep; t--) {
      dh = deriv_t.getSharedDataTensor({deriv_t.width()}, t * deriv_t.width());

      dc =
        derivc_t.getSharedDataTensor({derivc_t.width()}, t * derivc_t.width());

      if (xs_t.height() != 1)
        xs = xs_t.getSharedDataTensor({xs_t.width()}, t * xs_t.width());
      else
        xs = xs_t;

      cs = cs_t.getSharedDataTensor({cs_t.width()}, t * cs_t.width());

      Tensor dfgio_t =
        dfgio_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);
      Tensor fgio_t =
        fgio_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

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

      Tensor dhi = dfgio_t.getSharedDataTensor({unit}, 0);
      Tensor dhf = dfgio_t.getSharedDataTensor({unit}, unit);
      Tensor dhg = dfgio_t.getSharedDataTensor({unit}, unit * 2);
      Tensor dho = dfgio_t.getSharedDataTensor({unit}, unit * 3);

      Tensor hi = fgio_t.getSharedDataTensor({unit}, 0);
      Tensor hf = fgio_t.getSharedDataTensor({unit}, unit);
      Tensor hg = fgio_t.getSharedDataTensor({unit}, unit * 2);
      Tensor ho = fgio_t.getSharedDataTensor({unit}, unit * 3);

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
      djdb_h.add_i(dfgio_t);
      djdw_x.add_i(xs.dot(dfgio_t, true, false));
      djdw_h.add_i(hs_prev.dot(dfgio_t, true, false));
      if (t > 0) {
        Tensor dh_nx = deriv_t.getSharedDataTensor({deriv_t.width()},
                                                   (t - 1) * deriv_t.width());
        dfgio_t.dot(weight_hh, dh_nx, false, true, 1.0f);
      }
    }
  }
}

void LSTMLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[LSTMParams::hidden_state], batch);
  context.updateTensor(wt_idx[LSTMParams::mem_cell], batch);
  context.updateTensor(wt_idx[LSTMParams::fgio], batch);
  context.updateTensor(wt_idx[LSTMParams::dropout_mask], batch);
}

} // namespace nntrainer
