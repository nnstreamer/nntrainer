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
  weight_xh,
  weight_hh,
  bias_h,
  hidden_state,
  zrg,
  dropout_mask
};

#define ENABLE_BIAS_IH 1
// Todo: enable bias_hh
#define ENABLE_BIAS_HH 0

// Todo: handle with strided tensor more efficiently and reduce temporary
// tensors
GRUCellLayer::GRUCellLayer() :
  LayerImpl(),
  grucell_props(props::Unit(), props::HiddenStateActivation(),
                props::RecurrentActivation(), props::DropOutRate(),
                props::MaxTimestep(), props::Timestep()),
  wt_idx({0}),
  acti_func(ActivationType::ACT_NONE, true),
  recurrent_acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3) {}

// - weight_xh ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) x NUM_GATE] -> z, r, g
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size) x NUM_GATE] -> z, r, g
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size) x NUM_GATE] -> z, r, g
void GRUCellLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);

  const unsigned int unit = std::get<props::Unit>(grucell_props).get();
  auto &hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(grucell_props);
  auto &recurrent_activation_type =
    std::get<props::RecurrentActivation>(grucell_props);
  const float dropout_rate = std::get<props::DropOutRate>(grucell_props);
  const unsigned int max_timestep = std::get<props::MaxTimestep>(grucell_props);

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

  // output_dim = [ batch, 1, 1, hidden_size (unit)]
  TensorDim output_dim(batch_size, 1, 1, unit);
  context.setOutputDimensions({output_dim});

  if (dropout_rate > epsilon) {
    wt_idx[GRUCellParams::dropout_mask] = context.requestTensor(
      output_dim, "dropout_mask", Tensor::Initializer::NONE, false,
      TensorLifespan::ITERATION_LIFESPAN);
  }

  TensorDim weight_xh_dim({feature_size, NUM_GATE * unit});
  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  TensorDim bias_dim({NUM_GATE * unit});

  // weight_initializer can be set seperately. weight_xh initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.
  wt_idx[GRUCellParams::weight_xh] =
    context.requestWeight(weight_xh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_xh", true);
  wt_idx[GRUCellParams::weight_hh] =
    context.requestWeight(weight_hh_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, "weight_hh", true);
  wt_idx[GRUCellParams::bias_h] = context.requestWeight(
    bias_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, "bias_h", true);

  TensorDim hidden_state_dim(max_timestep * batch_size, 1, 1, unit);
  wt_idx[GRUCellParams::hidden_state] = context.requestTensor(
    hidden_state_dim, "hidden_state", Tensor::Initializer::NONE, true,
    TensorLifespan::ITERATION_LIFESPAN);

  TensorDim zrg_dim(max_timestep * batch_size, 1, 1, unit * NUM_GATE);
  wt_idx[GRUCellParams::zrg] =
    context.requestTensor(zrg_dim, "zrg", Tensor::Initializer::NONE, true,
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
  const unsigned int unit = std::get<props::Unit>(grucell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(grucell_props);
  const unsigned int max_timestep = std::get<props::MaxTimestep>(grucell_props);
  const unsigned int timestep = std::get<props::Timestep>(grucell_props);

  Tensor &weight_xh = context.getWeight(wt_idx[GRUCellParams::weight_xh]);
  Tensor &weight_hh = context.getWeight(wt_idx[GRUCellParams::weight_hh]);
  Tensor &bias_ih = context.getWeight(wt_idx[GRUCellParams::bias_h]);

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_states =
    context.getTensor(wt_idx[GRUCellParams::hidden_state]);
  Tensor &zrg_gates = context.getTensor(wt_idx[GRUCellParams::zrg]);
  Tensor prev_hidden_state;

  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();

  hidden_states.reshape({max_timestep, 1, batch_size, unit});
  zrg_gates.reshape({max_timestep, 1, batch_size, NUM_GATE * unit});

  Tensor hidden_state = hidden_states.getBatchSlice(timestep, 1);
  if (!timestep) {
    prev_hidden_state = Tensor(batch_size, unit);
    prev_hidden_state.setZero();
  } else {
    prev_hidden_state = hidden_states.getBatchSlice(timestep - 1, 1);
  }
  Tensor zrg_gate = zrg_gates.getBatchSlice(timestep, 1);

  input.dot(weight_xh, zrg_gate); // x_z, x_r, x_g

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

  if (timestep) {
    zr_gate.add_i_strided(prev_hidden_state.dot(weight_hh_zr));
  }
  Tensor bias_ih_zr = bias_ih.getSharedDataTensor({2 * unit}, 0);
  zr_gate.add_i(bias_ih_zr);

  recurrent_acti_func.run_fn(zr_gate, zr_gate);

  Tensor z_gate = zr_gate.getSharedDataTensor({batch_size, unit}, 0, false);
  Tensor r_gate = zr_gate.getSharedDataTensor({batch_size, unit}, unit, false);

  Tensor temp;
  prev_hidden_state.dot(weight_hh_g, temp, false, false);
#if ENABLE_BIAS_HH
  // Todo: fix this to get the bias_hh_g from bias_hh
  Tensor bias_hh_g = bias_ih.getSharedDataTensor({unit}, 2 * unit);
  temp.add_i(bias_hh_g);
#endif
  temp.multiply_i_strided(r_gate);
  g_gate.add_i_strided(temp);
#if ENABLE_BIAS_IH
  Tensor bias_ih_g = bias_ih.getSharedDataTensor({unit}, 2 * unit);
  g_gate.add_i(bias_ih_g);
#endif

  acti_func.run_fn(g_gate, g_gate);

  z_gate.multiply_strided(prev_hidden_state, hidden_state);
  temp = z_gate.multiply(-1.0).add(1.0);
  hidden_state.add_i(g_gate.multiply_strided(temp));

  if (dropout_rate > epsilon && training) {
    Tensor mask = context.getTensor(wt_idx[GRUCellParams::dropout_mask]);
    mask.dropout_mask(dropout_rate);
    hidden_state.multiply_i(mask);
  }

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  output.copy(hidden_state);
}

void GRUCellLayer::calcDerivative(RunLayerContext &context) {
  const unsigned int unit = std::get<props::Unit>(grucell_props).get();
  const unsigned int max_timestep = std::get<props::MaxTimestep>(grucell_props);
  const unsigned int timestep = std::get<props::Timestep>(grucell_props);
  const TensorDim &input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  const unsigned int batch_size = input_dim.batch();

  Tensor &zrg_gates_derivatives =
    context.getTensorGrad(wt_idx[GRUCellParams::zrg]);
  zrg_gates_derivatives.reshape({max_timestep, 1, batch_size, NUM_GATE * unit});
  Tensor zrg_gate_derivative = zrg_gates_derivatives.getBatchSlice(timestep, 1);
  Tensor &weight_xh = context.getWeight(wt_idx[GRUCellParams::weight_xh]);
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  zrg_gate_derivative.dot(weight_xh, outgoing_derivative, false, true);
}

void GRUCellLayer::calcGradient(RunLayerContext &context) {
  const unsigned int unit = std::get<props::Unit>(grucell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(grucell_props);
  const unsigned int max_timestep = std::get<props::MaxTimestep>(grucell_props);
  const unsigned int timestep = std::get<props::Timestep>(grucell_props);

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();

  Tensor &djdweight_xh =
    context.getWeightGrad(wt_idx[GRUCellParams::weight_xh]);
  Tensor &djdweight_hh =
    context.getWeightGrad(wt_idx[GRUCellParams::weight_hh]);
  Tensor &djdbias_ih = context.getWeightGrad(wt_idx[GRUCellParams::bias_h]);
  Tensor &weight_hh = context.getWeight(wt_idx[GRUCellParams::weight_hh]);
  Tensor &bias_h = context.getWeight(wt_idx[GRUCellParams::bias_h]);
  Tensor bias_h_g = bias_h.getSharedDataTensor({unit}, 2 * unit);

  Tensor djdw_zr_h =
    djdweight_hh.getSharedDataTensor({unit, 2 * unit}, 0, false);
  Tensor djdw_g_h =
    djdweight_hh.getSharedDataTensor({unit, unit}, 2 * unit, false);
  Tensor &hidden_states =
    context.getTensor(wt_idx[GRUCellParams::hidden_state]);
  hidden_states.reshape({max_timestep, 1, batch_size, unit});
  Tensor hidden_state = hidden_states.getBatchSlice(timestep, 1);
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
  hidden_state_derivative.reshape(incoming_derivative.getDim());
  if (timestep + 1 == max_timestep) {
    djdweight_xh.setZero();
    djdweight_hh.setZero();
    djdbias_ih.setZero();
    hidden_state_derivative.copyData(incoming_derivative);
  } else {
    hidden_state_derivative.add_i(incoming_derivative);
  }
  // restore the dimension
  hidden_state_derivative.reshape({1, 1, batch_size, unit});

  Tensor hs_prev;
  Tensor dh_nx;
  if (timestep) {
    hs_prev = hidden_states.getBatchSlice(timestep - 1, 1);
    dh_nx = hidden_states_derivatives.getBatchSlice(timestep - 1, 1);
  } else {
    dh_nx = Tensor(batch_size, unit);
    hs_prev = Tensor(batch_size, unit);
    hs_prev.setZero();
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

  hidden_state_derivative.multiply_strided(zt, dh_nx);    // dh_nx = d1
  hidden_state_derivative.multiply_strided(hs_prev, dhz); // dhz = d2
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
  hs_prev.dot(wg_hh, temp);
#if ENABLE_BIAS_HH
  temp.add_i(bias_h_g);
#endif
  dhg_.multiply_strided(temp, dhr); // dhr = d15

  // reset temp : hs_prev * rt for djdbias_hh_g and dh_nx
  dhg_.multiply_strided(rt, temp);
#if ENABLE_BIAS_HH
  // Todo: fix this to get the djdbias_hh_g from djdbias_hh
  Tensor djdbias_hh_g = djdbias_ih.getSharedDataTensor({unit}, 2 * unit);
  temp.sum(2, djdbias_hh_g, 1.0, 1.0);
#endif
  temp.dot(wg_hh, dh_nx, false, true, 1.0); // dh_nx = d1 + d14

  recurrent_acti_func.run_prime_fn(rt, dhr, dhr); // dhr = d16

#if ENABLE_BIAS_HH
  // Todo: fix this to get the djdbias_hh_zr from djdbias_hh
  Tensor djdbias_hh_zr = djdbias_ih.getSharedDataTensor({2 * unit}, 0);
  djdbias_hh_zr.add_i(
    zrg_gate_derivative.sum(2).getSharedDataTensor({2 * unit}, 0));
#endif
#if ENABLE_BIAS_IH
  zrg_gate_derivative.sum(2, djdbias_ih, 1.0, 1.0);
#endif

  djdweight_xh.add_i(input.dot(zrg_gate_derivative, true, false));

  Tensor dhzr_;
  dhzr_.copy_with_stride(dhzr);
  djdw_zr_h.add_i_strided(hs_prev.dot(dhzr_, true, false));
  djdw_g_h.add_i_strided(hs_prev.dot(temp, true, false));
  dhzr_.dot(wzr_hh, dh_nx, false, true, 1.0); // dh_nx = d1 + d14 + d12 + d17
}

void GRUCellLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  unsigned int &max_timestep = std::get<props::MaxTimestep>(grucell_props);
  context.updateTensor(wt_idx[GRUCellParams::hidden_state],
                       max_timestep * batch);
  context.updateTensor(wt_idx[GRUCellParams::zrg], max_timestep * batch);
  context.updateTensor(wt_idx[GRUCellParams::dropout_mask], batch);
}

} // namespace nntrainer
