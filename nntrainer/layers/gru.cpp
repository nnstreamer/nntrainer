// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gru.cpp
 * @date   17 March 2021
 * @brief  This is Gated Recurrent Unit Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * h_prev --------d1------->[*]-------d0----->[+]---d0--> h
 * dh_nx    |  |             |                 | d0      dh
 *          | d14            | d2        d3    |
 *          |  |             +-----[1-]------>[*]
 *          | [*]<---+ d15   |d5               | d6
 *          |  |     |rt     | zt              |gt
 *          |  |    [sig]   [sig]            [tanh]
 *          |  |     |d16    | d7              |d8
 *          |  |    [+]      [+]              [+]
 *          |  |    / \d16   |  \ d7          / \ d8
 *          |  |  Whhr Wxhr Whhz Wxhz       Whhg Wxhg
 *          |  |  |d17  |d13 |d12 |d11       |d10 | d9
 *          +- |--+------|---+    |          |    |
 *             +---------|--------|----------+    |
 *   xs------------------+--------+---------------+
 */

#include <cmath>
#include <gru.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum GRUParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  hidden_state,
  zrg,
  h_prev,
  dropout_mask
};

GRULayer::GRULayer() :
  LayerImpl(),
  gru_props(props::Unit(),
            props::HiddenStateActivation() = ActivationType::ACT_TANH,
            props::RecurrentActivation() = ActivationType::ACT_SIGMOID,
            props::ReturnSequences(), props::DropOutRate(),
            props::IntegrateBias(), props::ResetAfter()),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void GRULayer::finalize(InitLayerContext &context) {
  const Tensor::Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props).get();
  const Tensor::Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props).get();
  const WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props).get();
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props).get();
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(gru_props).get();
  ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(gru_props).get();
  ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(gru_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(gru_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(gru_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(gru_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "GRU layer takes only one input";

  // input_dim = [ batch, 1, time_iteration, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[0];
  const unsigned int batch_size = input_dim.batch();
  const unsigned int max_timestep = input_dim.height();
  const unsigned int feature_size = input_dim.width();

  // if return_sequences == False :
  //      output_dim = [ batch, 1, 1, unit ]
  // else:
  //      output_dim = [ batch, 1, time_iteration, unit ]
  TensorDim output_dim(
    {batch_size, 1, return_sequences ? max_timestep : 1, unit});
  context.setOutputDimensions({output_dim});

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // - weight_ih ( input to hidden )
  // weight_ih_dim : [ 1, 1, feature_size, NUMGATE * unit ] -> z, r, g
  TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[GRUParams::weight_ih] = context.requestWeight(
    weight_ih_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_ih", true);
  // - weight_hh ( hidden to hidden )
  // weight_hh_dim : [ 1, 1, unit, NUM_GATE * unit ] -> z, r, g
  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[GRUParams::weight_hh] = context.requestWeight(
    weight_hh_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // - bias_h ( input bias, hidden bias are integrate to 1 bias )
      // bias_h_dim : [ 1, 1, 1, NUM_GATE * unit ] -> z, r, g
      TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[GRUParams::bias_h] = context.requestWeight(
        bias_h_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
        "bias_h", true);
    } else {
      // - bias_ih ( input bias )
      // bias_ih_dim : [ 1, 1, 1, NUM_GATE * unit ] -> z, r, g
      TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[GRUParams::bias_ih] = context.requestWeight(
        bias_ih_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_ih", true);
      // - bias_hh ( hidden bias )
      // bias_hh_dim : [ 1, 1, 1, NUM_GATE * unit ] -> z, r, g
      TensorDim bias_hh_dim({NUM_GATE * unit});
      wt_idx[GRUParams::bias_hh] = context.requestWeight(
        bias_hh_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_hh", true);
    }
  }

  // hidden_state_dim = [ batch, 1, max_timestep, unit ]
  TensorDim hidden_state_dim(batch_size, 1, max_timestep, unit);
  wt_idx[GRUParams::hidden_state] = context.requestTensor(
    hidden_state_dim, "hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  // zrg_dim = [ batch, 1, max_timestep, NUM_GATE * unit ]
  TensorDim zrg_dim(batch_size, 1, max_timestep, NUM_GATE * unit);
  wt_idx[GRUParams::zrg] =
    context.requestTensor(zrg_dim, "zrg", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  // h_prev_dim = [ batch, 1, 1, unit ]
  TensorDim h_prev_dim = TensorDim({batch_size, 1, 1, unit});
  wt_idx[GRUParams::h_prev] =
    context.requestTensor(h_prev_dim, "h_prev", Tensor::Initializer::NONE,
                          false, TensorLifespan::FORWARD_FUNC_LIFESPAN);

  if (dropout_rate > epsilon) {
    TensorDim dropout_mask_dim(batch_size, 1, max_timestep, unit);
    wt_idx[GRUParams::dropout_mask] = context.requestTensor(
      output_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);
  recurrent_acti_func.setActiFunc(recurrent_activation_type);
}

void GRULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gru_props);
  LayerImpl::setProperty(remain_props);
}

void GRULayer::exportTo(Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(gru_props, method, this);
}

void GRULayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(gru_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(gru_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(gru_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(gru_props).get();
  const bool reset_after = std::get<props::ResetAfter>(gru_props).get();

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();
  const unsigned int max_timestep = input_dim.height();
  const unsigned int feature_size = input_dim.width();
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const Tensor &weight_ih = context.getWeight(wt_idx[GRUParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[GRUParams::weight_hh]);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[GRUParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[GRUParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[GRUParams::bias_hh])
                      : empty;

  Tensor &hidden_state = context.getTensor(wt_idx[GRUParams::hidden_state]);
  Tensor &zrg = context.getTensor(wt_idx[GRUParams::zrg]);
  Tensor &h_prev = context.getTensor(wt_idx[GRUParams::h_prev]);

  hidden_state.setZero();
  zrg.setZero();
  h_prev.setZero();

  Tensor prev_hs;
  Tensor hs;

  // zt = sigma(W_hz.h_prev + W_xz.xs)
  // rt = sigma(W_hr.h_prev + W_xr.xs)
  // gt = tanh((h_prev*rt).W_hr + W_xg.xs)
  // h_nx = (1-zt)*gt + zt*h_prev

  for (unsigned int b = 0; b < batch_size; ++b) {
    Tensor islice = input.getBatchSlice(b, 1);
    Tensor oslice = hidden_state.getBatchSlice(b, 1);
    Tensor zrg_ = zrg.getBatchSlice(b, 1);

    for (unsigned int t = 0; t < max_timestep; ++t) {
      Tensor xs = islice.getSharedDataTensor({feature_size}, t * feature_size);

      /** @todo verify this dropout working */
      // if (dropout_rate > 0.0 && training) {
      //   xs.multiply_i(xs.dropout_mask(dropout_rate));
      // }
      hs = oslice.getSharedDataTensor({unit}, t * unit);
      Tensor zrg_t =
        zrg_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      if (t > 0) {
        prev_hs = oslice.getSharedDataTensor({unit}, (t - 1) * unit);
      } else {
        prev_hs = h_prev.getBatchSlice(b, 1);
      }

      xs.dot(weight_ih, zrg_t); // x_z, x_r, x_g

      Tensor ztrt = zrg_t.getSharedDataTensor({unit * 2}, 0);

      Tensor w_hh;
      w_hh.copy_with_stride(
        weight_hh.getSharedDataTensor({1, 1, unit, unit * 2}, 0, false));
      Tensor w_g;
      w_g.copy_with_stride(
        weight_hh.getSharedDataTensor({1, 1, unit, unit}, unit * 2, false));

      Tensor gt = zrg_t.getSharedDataTensor({unit}, unit * 2);

      ztrt.add_i(prev_hs.dot(w_hh));
      if (!disable_bias) {
        if (integrate_bias) {
          Tensor ztrt_bias_h = bias_h.getSharedDataTensor({unit * 2}, 0);
          ztrt.add_i(ztrt_bias_h);
        } else {
          Tensor ztrt_bias_ih = bias_ih.getSharedDataTensor({unit * 2}, 0);
          ztrt.add_i(ztrt_bias_ih);
          Tensor ztrt_bias_hh = bias_hh.getSharedDataTensor({unit * 2}, 0);
          ztrt.add_i(ztrt_bias_hh);
        }
      }

      recurrent_acti_func.run_fn(ztrt, ztrt);

      Tensor zt = ztrt.getSharedDataTensor({unit}, 0);
      Tensor rt = ztrt.getSharedDataTensor({unit}, unit);

      Tensor temp;
      if (reset_after) {
        prev_hs.dot(w_g, temp);
        if (!disable_bias && !integrate_bias) {
          Tensor bias_hh_g = bias_hh.getSharedDataTensor({unit}, 2 * unit);
          temp.add_i(bias_hh_g);
        }
        temp.multiply_i(rt);
        gt.add_i(temp);
      } else {
        rt.multiply(prev_hs, temp);
        temp.dot(w_g, gt, false, false, 1.0f);
        if (!disable_bias && !integrate_bias) {
          Tensor bias_hh_g = bias_hh.getSharedDataTensor({unit}, 2 * unit);
          gt.add_i(bias_hh_g);
        }
      }
      if (!disable_bias) {
        if (integrate_bias) {
          Tensor gt_bias_h = bias_h.getSharedDataTensor({unit}, unit * 2);
          gt.add_i(gt_bias_h);
        } else {
          Tensor gt_bias_ih = bias_ih.getSharedDataTensor({unit}, unit * 2);
          gt.add_i(gt_bias_ih);
        }
      }

      acti_func.run_fn(gt, gt);

      zt.multiply(prev_hs, hs);
      temp = zt.multiply(-1.0).add(1.0);
      hs.add_i(gt.multiply(temp));

      if (dropout_rate > epsilon && training) {
        Tensor mask_ = context.getTensor(wt_idx[GRUParams::dropout_mask])
                         .getBatchSlice(b, 1);
        Tensor msk = mask_.getSharedDataTensor({unit}, t * unit);
        msk.dropout_mask(dropout_rate);
        hs.multiply_i(msk);
      }
    }
  }

  if (!return_sequences) {
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      Tensor dest = output.getSharedDataTensor({unit}, batch * unit);
      Tensor src = hidden_state.getSharedDataTensor(
        {unit}, batch * unit * max_timestep + (max_timestep - 1) * unit);
      dest.copy(src);
    }
  } else {
    output.copy(hidden_state);
  }
}

void GRULayer::calcDerivative(RunLayerContext &context) {
  Tensor &zrg_derivative = context.getTensorGrad(wt_idx[GRUParams::zrg]);
  Tensor &weight_ih = context.getWeight(wt_idx[GRUParams::weight_ih]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  zrg_derivative.dot(weight_ih, outgoing_derivative, false, true);
}

void GRULayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(gru_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(gru_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(gru_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(gru_props).get();
  const bool reset_after = std::get<props::ResetAfter>(gru_props).get();

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();
  const unsigned int max_timestep = input_dim.height();
  const unsigned int feature_size = input_dim.width();
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &djdweight_ih = context.getWeightGrad(wt_idx[GRUParams::weight_ih]);
  Tensor &weight_hh = context.getWeight(wt_idx[GRUParams::weight_hh]);
  Tensor &djdweight_hh = context.getWeightGrad(wt_idx[GRUParams::weight_hh]);
  Tensor empty;
  Tensor &djdbias_h = !disable_bias && integrate_bias
                        ? context.getWeightGrad(wt_idx[GRUParams::bias_h])
                        : empty;
  Tensor &djdbias_ih = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[GRUParams::bias_ih])
                         : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[GRUParams::bias_hh])
                      : empty;
  Tensor &djdbias_hh = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[GRUParams::bias_hh])
                         : empty;

  Tensor djdweight_hh_zr = Tensor({1, 1, unit, unit * 2}, true);
  Tensor djdweight_hh_g = Tensor({1, 1, unit, unit}, true);
  Tensor &hidden_state_derivative =
    context.getTensorGrad(wt_idx[GRUParams::hidden_state]);
  Tensor &hidden_state = context.getTensor(wt_idx[GRUParams::hidden_state]);
  Tensor &zrg = context.getTensor(wt_idx[GRUParams::zrg]);
  Tensor &d_zrg = context.getTensorGrad(wt_idx[GRUParams::zrg]);

  djdweight_ih.setZero();
  djdweight_hh_zr.setZero();
  djdweight_hh_g.setZero();
  if (!disable_bias) {
    if (integrate_bias) {
      djdbias_h.setZero();
    } else {
      djdbias_ih.setZero();
      djdbias_hh.setZero();
    }
  }

  hidden_state_derivative.setZero();
  d_zrg.setZero();

  if (!return_sequences) {
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      Tensor dest = hidden_state_derivative.getSharedDataTensor(
        {unit}, batch * unit * max_timestep + (max_timestep - 1) * unit);
      Tensor src =
        incoming_derivative.getSharedDataTensor({unit}, batch * unit);
      dest.copy(src);
    }
  } else {
    hidden_state_derivative.copy(incoming_derivative);
  }

  if (dropout_rate > epsilon) {
    hidden_state_derivative.multiply_i(
      context.getTensor(wt_idx[GRUParams::dropout_mask]));
  }

  Tensor dh_nx = Tensor({unit});

  for (unsigned int b = 0; b < batch_size; ++b) {
    Tensor deriv_t = hidden_state_derivative.getBatchSlice(b, 1);
    Tensor xs_t = input.getBatchSlice(b, 1);
    Tensor hs_t = hidden_state.getBatchSlice(b, 1);

    dh_nx.setZero();

    Tensor dh;
    Tensor prev_hs;
    Tensor xs;
    Tensor dzrg_ = d_zrg.getBatchSlice(b, 1);
    Tensor zrg_ = zrg.getBatchSlice(b, 1);

    for (unsigned int t = max_timestep; t-- > 0;) {
      dh = deriv_t.getSharedDataTensor({unit}, t * unit);
      xs = xs_t.getSharedDataTensor({feature_size}, t * feature_size);

      Tensor dzrg_t =
        dzrg_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);
      Tensor zrg_t =
        zrg_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      if (t == 0) {
        prev_hs = Tensor({unit});
        prev_hs.setZero();
      } else {
        prev_hs = hs_t.getSharedDataTensor({unit}, (t - 1) * unit);
      }
      if (t < max_timestep - 1) {
        dh.add_i(dh_nx);
      }

      Tensor dhz = dzrg_t.getSharedDataTensor({unit}, 0);
      Tensor dhr = dzrg_t.getSharedDataTensor({unit}, unit);
      Tensor dhg = dzrg_t.getSharedDataTensor({unit}, unit * 2);

      Tensor zt = zrg_t.getSharedDataTensor({unit}, 0);
      Tensor rt = zrg_t.getSharedDataTensor({unit}, unit);
      Tensor gt = zrg_t.getSharedDataTensor({unit}, unit * 2);

      zt.multiply(dh, dh_nx);          // dh_nx = d1
      dh.multiply(prev_hs, dhz);       // dhz = d2
      dhz.subtract_i(gt.multiply(dh)); // dhz = d5
      zt.multiply(-1.0, dhg);
      dhg.add_i(1.0);
      dhg.multiply_i(dh); // dhg = d6

      recurrent_acti_func.run_prime_fn(zt, dhz, dhz); // dhz = d7
      acti_func.run_prime_fn(gt, dhg, dhg);           // dhg = d8

      Tensor dhzr = dzrg_t.getSharedDataTensor({unit * 2}, 0); // dhz+dhr

      Tensor wg_hh;
      wg_hh.copy_with_stride(
        weight_hh.getSharedDataTensor({1, 1, unit, unit}, unit * 2, false));
      Tensor wzr_hh;
      wzr_hh.copy_with_stride(
        weight_hh.getSharedDataTensor({1, 1, unit, unit * 2}, 0, false));

      Tensor temp = Tensor({unit});

      if (reset_after) {
        prev_hs.dot(wg_hh, temp);
        if (!disable_bias && !integrate_bias) {
          const Tensor bias_hh_g =
            bias_hh.getSharedDataTensor({unit}, 2 * unit);
          temp.add_i(bias_hh_g);
        }
        dhg.multiply(temp, dhr);

        // reset temp: dhg * rt for djdbias_hh_g, dh_nx and djdweight_hh_g
        dhg.multiply(rt, temp);
        if (!disable_bias && !integrate_bias) {
          Tensor djdbias_hh_g =
            djdbias_hh.getSharedDataTensor({unit}, 2 * unit);
          djdbias_hh_g.add_i(temp);
        }
        temp.dot(wg_hh, dh_nx, false, true, 1.0f); // dh_nx = d1 + d14
        djdweight_hh_g.add_i(prev_hs.dot(temp, true, false));
      } else {
        if (!disable_bias && !integrate_bias) {
          Tensor djdbias_hh_g =
            djdbias_hh.getSharedDataTensor({unit}, 2 * unit);
          djdbias_hh_g.add_i(dhg);
        }

        dhg.dot(wg_hh, temp, false, true); // temp = d10
        temp.multiply(prev_hs, dhr);       // dhr = d15s
        temp.multiply_i(rt);               // temp=d14
        dh_nx.add_i(temp);                 //  dh_nx = d1 + d14

        // reset temp : prev_hs * rt for djdweight_hh_g
        rt.multiply(prev_hs, temp);
        temp.dot(dhg, djdweight_hh_g, true, false, 1.0f);
      }

      recurrent_acti_func.run_prime_fn(rt, dhr, dhr); // dhr = d16

      if (!disable_bias) {
        if (integrate_bias) {
          djdbias_h.add_i(dzrg_t); // dzrg_t = d7+d16+d8
        } else {
          djdbias_ih.add_i(dzrg_t); // dzrg_t = d7+d16+d8
          Tensor djdbias_hh_zr = djdbias_hh.getSharedDataTensor({2 * unit}, 0);
          djdbias_hh_zr.add_i(dzrg_t.getSharedDataTensor({2 * unit}, 0));
        }
      }

      djdweight_hh_zr.add_i(prev_hs.dot(dhzr, true, false));
      xs.dot(dzrg_t, djdweight_ih, true, false, 1.0f);
      dhzr.dot(wzr_hh, dh_nx, false, true, 1.0); // dh_nx = d1 + d14 + d12 + d17
    }
  }
  for (unsigned int h = 0; h < unit; ++h) {
    float *data = djdweight_hh_zr.getAddress(h * unit * 2);
    float *rdata = djdweight_hh.getAddress(h * unit * NUM_GATE);
    std::copy(data, data + unit * 2, rdata);
  }

  for (unsigned int h = 0; h < unit; ++h) {
    float *data = djdweight_hh_g.getAddress(h * unit);
    float *rdata = djdweight_hh.getAddress(h * unit * NUM_GATE + unit * 2);
    std::copy(data, data + unit, rdata);
  }
}

void GRULayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[GRUParams::hidden_state], batch);
  context.updateTensor(wt_idx[GRUParams::zrg], batch);
  context.updateTensor(wt_idx[GRUParams::h_prev], batch);

  if (std::get<props::DropOutRate>(gru_props).get() > epsilon) {
    context.updateTensor(wt_idx[GRUParams::dropout_mask], batch);
  }
}

} // namespace nntrainer
