// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   mol_attention_layer.cpp
 * @date   11 November 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MoL Attention Layer Class for Neural Network
 *
 */

#include <cmath>

#include <layer_context.h>
#include <mol_attention_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

MoLAttentionLayer::MoLAttentionLayer() : wt_idx({0}) {}

MoLAttentionLayer::~MoLAttentionLayer() {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum AttentionParams { query = 0, value = 1, state = 2, mlp_w, mlp_proj_w };

void MoLAttentionLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 3)
    throw std::runtime_error("MoL Attention layer needs 3 inputs.");

  auto const &all_dims = context.getInputDimensions();
  auto const &query_dim = all_dims[AttentionParams::query];

  wt_idx[AttentionParams::query] = AttentionParams::query;
  wt_idx[AttentionParams::value] = AttentionParams::value;
  wt_idx[AttentionParams::state] = AttentionParams::state;

  softmax.setActiFunc(ActivationType::ACT_SOFTMAX);
  tanh.setActiFunc(ActivationType::ACT_TANH);
  sigmoid.setActiFunc(ActivationType::ACT_SIGMOID);

  NNTR_THROW_IF(std::get<props::Unit>(mol_props).empty(), std::invalid_argument)
    << "Number of units not provided for layer " << context.getName();
  auto unit = std::get<props::Unit>(mol_props).get();

  NNTR_THROW_IF(std::get<props::MoL_K>(mol_props).empty(),
                std::invalid_argument)
    << "MoL_K property not provided for layer " << context.getName();
  auto mol_k = std::get<props::MoL_K>(mol_props).get();

  TensorDim mlp_w_dim = {query_dim.width(), unit};
  wt_idx[AttentionParams::mlp_w] =
    context.requestWeight(mlp_w_dim, Tensor::Initializer::XAVIER_UNIFORM,
                          WeightRegularizer::NONE, 0.0, "mlp_w", true);

  TensorDim mlp_proj_w_dim = {unit, 3 * mol_k};
  wt_idx[AttentionParams::mlp_proj_w] =
    context.requestWeight(mlp_proj_w_dim, Tensor::Initializer::XAVIER_UNIFORM,
                          WeightRegularizer::NONE, 0.0, "mlp_proj_w", true);

  context.setOutputDimensions({query_dim});
}

void MoLAttentionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &query = context.getInput(wt_idx[AttentionParams::query]);
  Tensor &value = context.getInput(wt_idx[AttentionParams::value]);
  Tensor &state = context.getInput(wt_idx[AttentionParams::state]);

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &mlp_w = context.getWeight(wt_idx[AttentionParams::mlp_w]);
  Tensor &mlp_proj_w = context.getWeight(wt_idx[AttentionParams::mlp_proj_w]);

  const TensorDim &input_dim = query.getDim();
  unsigned int batch = input_dim.batch();
  auto mol_k = std::get<props::MoL_K>(mol_props).get();

  Tensor mlp_out = query.dot(mlp_w);

  Tensor mlp_tanh;
  tanh.run_fn(mlp_out, mlp_tanh);

  Tensor mlp_proj_out = mlp_tanh.dot(mlp_proj_w);

  /** @note kappa_hat, beta_hat are strided */
  Tensor kappa_hat =
    mlp_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, 0, false);
  Tensor beta_hat =
    mlp_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k, false);
  Tensor alpha_hat;
  alpha_hat.copy_with_stride(
    mlp_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k * 2, false));

  Tensor kappa = kappa_hat.apply(static_cast<float (*)(float)>(&std::exp));
  Tensor beta = beta_hat.apply(static_cast<float (*)(float)>(&std::exp));

  Tensor alpha;
  softmax.run_fn(alpha_hat, alpha);

  Tensor m = state.add(kappa);

  /** @todo cache u_base, u_pos, u_neg */
  Tensor u_base = Tensor(TensorDim({batch, 1, value.height(), mol_k}));
  for (unsigned int b = 0; b < batch; b++) {
    for (unsigned int h = 0; h < u_base.height(); h++) {
      float *u_data = u_base.getAddress(b, 0, h, 0);
      std::fill(u_data, u_data + u_base.width(), h + 1);
    }
  }

  Tensor u_pos = u_base.add(0.5);
  u_base.add_i(-0.5);
  Tensor u_neg = u_base;

  Tensor integral_left, integral_right;
  sigmoid.run_fn(u_pos.subtract(state).divide(beta.add(1e-8f)), integral_left);
  sigmoid.run_fn(u_neg.subtract(state).divide(beta.add(1e-8f)), integral_right);
  Tensor integral = integral_left.subtract(integral_right);

  Tensor integral_scaled = integral.multiply(alpha);
  Tensor scores = integral_scaled.sum(3);
  scores.reshape(TensorDim({scores.batch(), 1, 1, scores.height()}));

  scores.dotBatched(value, output);
}

void MoLAttentionLayer::calcDerivative(RunLayerContext &context) { /** NYI */
}

void MoLAttentionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mol_props);
  AttentionLayer::setProperty(remain_props);
}

void MoLAttentionLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  /** NYI */
}

} /* namespace nntrainer */
