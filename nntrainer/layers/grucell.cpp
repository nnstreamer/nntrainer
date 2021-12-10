// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   grucell.cpp
 * @date   28 Oct 2021
 * @brief  This is Gated Recurrent Unit Cell Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
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

#include <grucell.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum GRUCellParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  hidden_state,
  zrg,
  dropout_mask
};

// Todo: handle with strided tensor more efficiently and reduce temporary
// tensors
GRUCellLayer::GRUCellLayer() :
  LayerImpl(),
  grucell_props(props::Unit(),
                props::HiddenStateActivation() = ActivationType::ACT_TANH,
                props::RecurrentActivation() = ActivationType::ACT_SIGMOID,
                props::DropOutRate(), props::IntegrateBias(),
                props::ResetAfter(), props::MaxTimestep(), props::Timestep()),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void GRUCellLayer::finalize(InitLayerContext &context) {
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

  const unsigned int unit = std::get<props::Unit>(grucell_props).get();
  const ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(grucell_props).get();
  const ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(grucell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(grucell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(grucell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(grucell_props).get();

  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("GRUCell layer takes only one input");
  }

  // input_dim = [ batch, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[0];
  if (input_dim.height() != 1 && input_dim.channel() != 1) {
    throw std::invalid_argument(
      "Input must be single time dimension for GRUCell");
  }

  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();

  // output_dim = [ batch, 1, 1, unit ]
  TensorDim output_dim(batch_size, 1, 1, unit);
  context.setOutputDimensions({output_dim});

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // - weight_ih ( input to hidden )
  // weight_ih_dim : [ 1, 1, feature_size, NUMGATE * unit ] -> z, r, g
  TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[GRUCellParams::weight_ih] =
    context.requestWeight(weight_ih_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_ih", true);
  // - weight_hh ( hidden to hidden )
  // weight_hh_dim : [ 1, 1, unit, NUM_GATE * unit ] -> z, r, g
  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[GRUCellParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // - bias_h ( input bias, hidden bias are integrate to 1 bias )
      // bias_h_dim : [ 1, 1, 1, NUM_GATE * unit ] -> z, r, g
      TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[GRUCellParams::bias_h] =
        context.requestWeight(bias_h_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_h", true);
    } else {
      // - bias_ih ( input bias )
      // bias_ih_dim : [ 1, 1, 1, NUM_GATE * unit ] -> z, r, g
      TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[GRUCellParams::bias_ih] =
        context.requestWeight(bias_ih_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_ih", true);
      // - bias_hh ( hidden bias )
      // bias_hh_dim : [ 1, 1, 1, NUM_GATE * unit ] -> z, r, g
      TensorDim bias_hh_dim({NUM_GATE * unit});
      wt_idx[GRUCellParams::bias_hh] =
        context.requestWeight(bias_hh_dim, bias_initializer,
                              WeightRegularizer::NONE, 1.0f, "bias_hh", true);
    }
  }

  // hidden_state_dim = [ max_timestep * batch, 1, 1, unit ]
  TensorDim hidden_state_dim(max_timestep * batch_size, 1, 1, unit);
  wt_idx[GRUCellParams::hidden_state] = context.requestTensor(
    hidden_state_dim, "hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN, false);

  // zrg_dim = [ max_timestep * batch, 1, 1, NUM_GATE * unit ]
  TensorDim zrg_dim(max_timestep * batch_size, 1, 1, NUM_GATE * unit);
  wt_idx[GRUCellParams::zrg] =
    context.requestTensor(zrg_dim, "zrg", Tensor::Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN, false);

  if (dropout_rate > epsilon) {
    // dropout_mask_dim = [ max_timestep * batch, 1, 1, unit ]
    TensorDim dropout_mask_dim(max_timestep * batch_size, 1, 1, unit);
    wt_idx[GRUCellParams::dropout_mask] = context.requestTensor(
      dropout_mask_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);
  recurrent_acti_func.setActiFunc(recurrent_activation_type);
}

void GRUCellLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, grucell_props);
  LayerImpl::setProperty(remain_props);
}

void GRUCellLayer::exportTo(Exporter &exporter,
                            const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(grucell_props, method, this);
}

void GRUCellLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(grucell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(grucell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(grucell_props).get();
  const bool reset_after = std::get<props::ResetAfter>(grucell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(grucell_props).get();
  const unsigned int timestep = std::get<props::Timestep>(grucell_props).get();

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();

  Tensor &weight_ih = context.getWeight(wt_idx[GRUCellParams::weight_ih]);
  Tensor &weight_hh = context.getWeight(wt_idx[GRUCellParams::weight_hh]);
  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[GRUCellParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[GRUCellParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[GRUCellParams::bias_hh])
                      : empty;

  Tensor &hidden_states =
    context.getTensor(wt_idx[GRUCellParams::hidden_state]);
  hidden_states.reshape({max_timestep, 1, batch_size, unit});
  Tensor prev_hidden_state;
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hidden_states.getBatchSlice(timestep - 1, 1);
  }
  Tensor hidden_state = hidden_states.getBatchSlice(timestep, 1);

  Tensor &zrg_gates = context.getTensor(wt_idx[GRUCellParams::zrg]);
  zrg_gates.reshape({max_timestep, 1, batch_size, NUM_GATE * unit});
  Tensor zrg_gate = zrg_gates.getBatchSlice(timestep, 1);

  input.dot(weight_ih, zrg_gate); // x_z, x_r, x_g

  Tensor zr_gate =
    zrg_gate.getSharedDataTensor({batch_size, 2 * unit}, 0, false);
  Tensor g_gate =
    zrg_gate.getSharedDataTensor({batch_size, unit}, 2 * unit, false);

  Tensor weight_hh_zr;
  Tensor weight_hh_g;
  weight_hh_zr.copy_with_stride(
    weight_hh.getSharedDataTensor({1, 1, unit, unit * 2}, 0, false));
  weight_hh_g.copy_with_stride(
    weight_hh.getSharedDataTensor({1, 1, unit, unit}, unit * 2, false));

  zr_gate.add_i_strided(prev_hidden_state.dot(weight_hh_zr));
  if (!disable_bias) {
    if (integrate_bias) {
      Tensor bias_h_zr = bias_h.getSharedDataTensor({2 * unit}, 0);
      zr_gate.add_i(bias_h_zr);
    } else {
      Tensor bias_ih_zr = bias_ih.getSharedDataTensor({2 * unit}, 0);
      zr_gate.add_i(bias_ih_zr);
      Tensor bias_hh_zr = bias_hh.getSharedDataTensor({2 * unit}, 0);
      zr_gate.add_i(bias_hh_zr);
    }
  }

  recurrent_acti_func.run_fn(zr_gate, zr_gate);

  Tensor z_gate = zr_gate.getSharedDataTensor({batch_size, unit}, 0, false);
  Tensor r_gate = zr_gate.getSharedDataTensor({batch_size, unit}, unit, false);

  Tensor temp;
  if (reset_after) {
    prev_hidden_state.dot(weight_hh_g, temp);
    if (!disable_bias && !integrate_bias) {
      Tensor bias_hh_g = bias_hh.getSharedDataTensor({unit}, 2 * unit);
      temp.add_i(bias_hh_g);
    }
    temp.multiply_i_strided(r_gate);
    g_gate.add_i_strided(temp);
  } else {
    r_gate.multiply_strided(prev_hidden_state, temp);
    temp.dot(weight_hh_g, g_gate, false, false, 1.0f);
    if (!disable_bias && !integrate_bias) {
      Tensor bias_hh_g = bias_hh.getSharedDataTensor({unit}, 2 * unit);
      g_gate.add_i(bias_hh_g);
    }
  }
  if (!disable_bias) {
    if (integrate_bias) {
      Tensor bias_h_g = bias_h.getSharedDataTensor({unit}, 2 * unit);
      g_gate.add_i(bias_h_g);
    } else {
      Tensor bias_ih_g = bias_ih.getSharedDataTensor({unit}, 2 * unit);
      g_gate.add_i(bias_ih_g);
    }
  }

  acti_func.run_fn(g_gate, g_gate);

  z_gate.multiply_strided(prev_hidden_state, hidden_state);
  temp = z_gate.multiply(-1.0).add(1.0);
  hidden_state.add_i(g_gate.multiply_strided(temp));

  if (dropout_rate > epsilon && training) {
    Tensor mask = context.getTensor(wt_idx[GRUCellParams::dropout_mask]);
    mask.dropout_mask(dropout_rate);
    hidden_state.multiply_i(mask);
  }

  output.copy(hidden_state);
}

void GRUCellLayer::calcDerivative(RunLayerContext &context) {
  const unsigned int unit = std::get<props::Unit>(grucell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(grucell_props).get();
  const unsigned int timestep = std::get<props::Timestep>(grucell_props).get();

  const unsigned int batch_size =
    context.getInput(SINGLE_INOUT_IDX).getDim().batch();

  Tensor &zrg_gates_derivatives =
    context.getTensorGrad(wt_idx[GRUCellParams::zrg]);
  zrg_gates_derivatives.reshape({max_timestep, 1, batch_size, NUM_GATE * unit});
  Tensor zrg_gate_derivative = zrg_gates_derivatives.getBatchSlice(timestep, 1);
  Tensor &weight_ih = context.getWeight(wt_idx[GRUCellParams::weight_ih]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  zrg_gate_derivative.dot(weight_ih, outgoing_derivative, false, true);
}

void GRUCellLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(grucell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(grucell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(grucell_props).get();
  const bool reset_after = std::get<props::ResetAfter>(grucell_props).get();
  const unsigned int max_timestep =
    std::get<props::MaxTimestep>(grucell_props).get();
  const unsigned int timestep = std::get<props::Timestep>(grucell_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const unsigned int batch_size = input.getDim().batch();

  Tensor &djdweight_ih =
    context.getWeightGrad(wt_idx[GRUCellParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[GRUCellParams::weight_hh]);
  Tensor &djdweight_hh =
    context.getWeightGrad(wt_idx[GRUCellParams::weight_hh]);

  Tensor empty;
  Tensor &djdbias_h = !disable_bias && integrate_bias
                        ? context.getWeightGrad(wt_idx[GRUCellParams::bias_h])
                        : empty;
  Tensor &djdbias_ih = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[GRUCellParams::bias_ih])
                         : empty;
  const Tensor &bias_hh = !disable_bias && !integrate_bias
                            ? context.getWeight(wt_idx[GRUCellParams::bias_hh])
                            : empty;
  Tensor &djdbias_hh = !disable_bias && !integrate_bias
                         ? context.getWeightGrad(wt_idx[GRUCellParams::bias_hh])
                         : empty;

  Tensor djdweight_hh_zr =
    djdweight_hh.getSharedDataTensor({unit, 2 * unit}, 0, false);
  Tensor djdweight_hh_g =
    djdweight_hh.getSharedDataTensor({unit, unit}, 2 * unit, false);
  Tensor &hidden_states =
    context.getTensor(wt_idx[GRUCellParams::hidden_state]);
  hidden_states.reshape({max_timestep, 1, batch_size, unit});
  Tensor &hidden_states_derivatives =
    context.getTensorGrad(wt_idx[GRUCellParams::hidden_state]);
  Tensor &incoming_derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &zrg_gates = context.getTensor(wt_idx[GRUCellParams::zrg]);
  zrg_gates.reshape({max_timestep, 1, batch_size, NUM_GATE * unit});
  Tensor zrg_gate = zrg_gates.getBatchSlice(timestep, 1);
  Tensor &zrg_gates_derivatives =
    context.getTensorGrad(wt_idx[GRUCellParams::zrg]);
  zrg_gates_derivatives.reshape({max_timestep, 1, batch_size, NUM_GATE * unit});
  Tensor zrg_gate_derivative = zrg_gates_derivatives.getBatchSlice(timestep, 1);

  hidden_states_derivatives.reshape({max_timestep, 1, batch_size, unit});
  Tensor hidden_state_derivative =
    hidden_states_derivatives.getBatchSlice(timestep, 1);
  if (timestep + 1 == max_timestep) {
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
  }

  hidden_state_derivative.reshape(
    incoming_derivative.getDim()); // reshape to incoming_derivative dim
  hidden_state_derivative.add_i(incoming_derivative);
  hidden_state_derivative.reshape(
    {1, 1, batch_size, unit}); // restore the dimension

  Tensor prev_hidden_state;
  Tensor dh_nx;
  if (timestep) {
    prev_hidden_state = hidden_states.getBatchSlice(timestep - 1, 1);
    dh_nx = hidden_states_derivatives.getBatchSlice(timestep - 1, 1);
  } else {
    dh_nx = Tensor(batch_size, unit);
    prev_hidden_state = Tensor(batch_size, unit);
    prev_hidden_state.setZero();
  }

  if (dropout_rate > epsilon) {
    hidden_states_derivatives.multiply_i(
      context.getTensor(wt_idx[GRUCellParams::dropout_mask]));
  }

  Tensor dhz =
    zrg_gate_derivative.getSharedDataTensor({batch_size, unit}, 0, false);
  Tensor dhr =
    zrg_gate_derivative.getSharedDataTensor({batch_size, unit}, unit, false);
  Tensor dhg = zrg_gate_derivative.getSharedDataTensor({batch_size, unit},
                                                       unit * 2, false);

  Tensor zt = zrg_gate.getSharedDataTensor({batch_size, unit}, 0, false);
  Tensor rt = zrg_gate.getSharedDataTensor({batch_size, unit}, unit, false);
  Tensor gt = zrg_gate.getSharedDataTensor({batch_size, unit}, unit * 2, false);

  hidden_state_derivative.multiply_strided(zt, dh_nx); // dh_nx = d1
  hidden_state_derivative.multiply_strided(prev_hidden_state, dhz); // dhz = d2
  dhz.add_i_strided(hidden_state_derivative.multiply_strided(gt),
                    -1.0f); // dhz = d5
  zt.multiply(-1.0, dhg);
  dhg.add_i(1.0);
  dhg.multiply_i_strided(hidden_state_derivative); // dhg = d6

  recurrent_acti_func.run_prime_fn(zt, dhz, dhz); // dhz = d7
  acti_func.run_prime_fn(gt, dhg, dhg);           // dhg = d8

  Tensor dhzr = zrg_gate_derivative.getSharedDataTensor({batch_size, unit * 2},
                                                        0, false); // dhz+dhr

  Tensor wg_hh;
  wg_hh.copy_with_stride(
    weight_hh.getSharedDataTensor({1, 1, unit, unit}, unit * 2, false));
  Tensor wzr_hh;
  wzr_hh.copy_with_stride(
    weight_hh.getSharedDataTensor({1, 1, unit, unit * 2}, 0, false));

  Tensor temp = Tensor(batch_size, unit);
  Tensor dhg_;
  dhg_.copy_with_stride(dhg);

  if (reset_after) {
    prev_hidden_state.dot(wg_hh, temp);
    if (!disable_bias && !integrate_bias) {
      const Tensor bias_hh_g = bias_hh.getSharedDataTensor({unit}, 2 * unit);
      temp.add_i(bias_hh_g);
    }
    dhg_.multiply_strided(temp, dhr); // dhr = d15

    // reset temp: dhg_ * rt for djdbias_hh_g, dh_nx and djdweight_hh_g
    dhg_.multiply_strided(rt, temp);
    if (!disable_bias && !integrate_bias) {
      Tensor djdbias_hh_g = djdbias_hh.getSharedDataTensor({unit}, 2 * unit);
      temp.sum(2, djdbias_hh_g, 1.0, 1.0);
    }
    temp.dot(wg_hh, dh_nx, false, true, 1.0); // dh_nx = d1 + d14
    djdweight_hh_g.add_i_strided(prev_hidden_state.dot(temp, true, false));
  } else {
    if (!disable_bias && !integrate_bias) {
      Tensor djdbias_hh_g = djdbias_hh.getSharedDataTensor({unit}, 2 * unit);
      dhg.sum(2, djdbias_hh_g, 1.0, 1.0);
    }

    dhg_.dot(wg_hh, temp, false, true);
    temp.multiply_strided(prev_hidden_state, dhr);
    temp.multiply_strided(rt, dh_nx, 1.0f);

    // reset temp: rt * prev_hidden_state for and djdweight_hh_g
    rt.multiply_strided(prev_hidden_state, temp);
    temp.dot(dhg_, djdweight_hh_g, true, false, 1.0f);
  }

  recurrent_acti_func.run_prime_fn(rt, dhr, dhr); // dhr = d16

  if (!disable_bias) {
    if (integrate_bias) {
      zrg_gate_derivative.sum(2, djdbias_h, 1.0, 1.0);
    } else {
      zrg_gate_derivative.sum(2, djdbias_ih, 1.0, 1.0);
      Tensor djdbias_hh_zr = djdbias_hh.getSharedDataTensor({2 * unit}, 0);
      djdbias_hh_zr.add_i(
        zrg_gate_derivative.sum(2).getSharedDataTensor({2 * unit}, 0));
    }
  }

  Tensor dhzr_;
  dhzr_.copy_with_stride(dhzr);
  djdweight_hh_zr.add_i_strided(prev_hidden_state.dot(dhzr_, true, false));
  input.dot(zrg_gate_derivative, djdweight_ih, true, false, 1.0f);
  dhzr_.dot(wzr_hh, dh_nx, false, true, 1.0); // dh_nx = d1 + d14 + d12 + d17
}

void GRUCellLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  unsigned int &max_timestep = std::get<props::MaxTimestep>(grucell_props);
  context.updateTensor(wt_idx[GRUCellParams::hidden_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[GRUCellParams::zrg], max_timestep * batch);

  const float dropout_rate = std::get<props::DropOutRate>(grucell_props);
  if (dropout_rate > epsilon) {
    context.updateTensor(wt_idx[GRUCellParams::dropout_mask],
                         max_timestep * batch);
  }
}

} // namespace nntrainer
