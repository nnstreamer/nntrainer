/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	bn_layer.cpp
 * @date	14 May 2020
 * @brief	This is Batch Normalization Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <bn_layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum BNParams { mu, var, gamma, beta, deviation };

BatchNormalizationLayer::BatchNormalizationLayer(int axis_) :
  Layer(),
  axis(axis_),
  wt_idx({0}),
  bn_props(props::Epsilon(), props::BNPARAMS_MU_INIT(),
           props::BNPARAMS_VAR_INIT(), props::BNPARAMS_BETA_INIT(),
           props::BNPARAMS_GAMMA_INIT(), props::Momentum()) {}

/// @todo add multiple axis support
void BatchNormalizationLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument(
      "Only one input is allowed for batch normalization layer");
  }

  auto &bnparams_mu = std::get<props::BNPARAMS_MU_INIT>(bn_props);
  auto &bnparams_var = std::get<props::BNPARAMS_VAR_INIT>(bn_props);
  auto &bnparams_beta = std::get<props::BNPARAMS_BETA_INIT>(bn_props);
  auto &bnparams_gamma = std::get<props::BNPARAMS_GAMMA_INIT>(bn_props);

  std::vector<TensorDim> output_dims(1);

  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  context.setOutputDimensions(context.getInputDimensions());

  TensorDim dim;

  /// @note this logic cannot tell channel is actually 1 or it is just not used.
  if (axis == -1)
    axis = in_dim.channel() > 1 ? 1 : 3;

  dim.setTensorDim(axis, in_dim.getTensorDim(axis));

  for (int i = 0; i < 4; ++i) {
    if (axis != i)
      axes_to_reduce.push_back(i);
  }

  wt_idx[BNParams::mu] =
    context.requestWeight(dim, bnparams_mu, WeightRegularizer::NONE, 1.0f,
                          context.getName() + ":moving_mean", false);
  wt_idx[BNParams::var] =
    context.requestWeight(dim, bnparams_var, WeightRegularizer::NONE, 1.0f,
                          context.getName() + ":moving_variance", false);
  wt_idx[BNParams::gamma] =
    context.requestWeight(dim, bnparams_gamma, WeightRegularizer::NONE, 1.0f,
                          context.getName() + ":gamma", true);
  wt_idx[BNParams::beta] =
    context.requestWeight(dim, bnparams_beta, WeightRegularizer::NONE, 1.0f,
                          context.getName() + ":beta", true);

  wt_idx[BNParams::deviation] = context.requestTensor(
    in_dim, context.getName() + ":deviation", Tensor::Initializer::NONE, false,
    TensorLifespan::ITERATION_LIFESPAN);
}

void BatchNormalizationLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, bn_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[BNLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void BatchNormalizationLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  float epsilon = std::get<props::Epsilon>(bn_props);
  float momentum = std::get<props::Momentum>(bn_props);

  Tensor &mu = context.getWeight(wt_idx[BNParams::mu]);
  Tensor &var = context.getWeight(wt_idx[BNParams::var]);
  Tensor &gamma = context.getWeight(wt_idx[BNParams::gamma]);
  Tensor &beta = context.getWeight(wt_idx[BNParams::beta]);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &deviation = context.getTensor(wt_idx[BNParams::deviation]);

  if (training) {
    /**
     * @todo support average with preallocated tensors,
     * and then register cmu as a temporary tensor
     */
    Tensor cmu = input_.average(axes_to_reduce);
    input_.subtract(cmu, deviation);

    Tensor t1 = deviation.pow(2.0f);
    cvar = t1.average(axes_to_reduce);

    mu.multiply_i(momentum);
    mu.add_i(cmu, 1 - momentum);
    var.multiply_i(momentum);
    var.add_i(cvar, 1 - momentum);

    cvar.add_i(epsilon);
    cvar.pow(-0.5f, invstd);
  } else {
    input_.subtract(mu, deviation);
    var.add(epsilon, invstd);
    invstd.pow_i(-0.5f);
  }

  deviation.multiply(invstd, hidden_);
  hidden_.multiply_i(gamma);
  hidden_.add_i(beta);
}

void BatchNormalizationLayer::calcDerivative(RunLayerContext &context) {

  Tensor &gamma = context.getWeight(wt_idx[BNParams::gamma]);
  Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &deviation = context.getTensor(wt_idx[BNParams::deviation]);

  Tensor dx_1 = gamma.multiply(invstd);
  Tensor dx_2 = deriv.subtract(deriv.average(axes_to_reduce));

  Tensor t1 = deviation.multiply(deriv);
  Tensor t2 = t1.average(axes_to_reduce);
  deviation.divide_i(cvar);
  deviation.multiply_i(t2);
  dx_2.subtract_i(deviation);

  dx_2.multiply(dx_1, dx);
}

void BatchNormalizationLayer::calcGradient(RunLayerContext &context) {

  Tensor &dgamma = context.getWeightGrad(wt_idx[BNParams::gamma]);
  Tensor &dbeta = context.getWeightGrad(wt_idx[BNParams::beta]);
  Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &deviation = context.getTensor(wt_idx[BNParams::deviation]);

  deriv.sum(axes_to_reduce, dbeta);
  Tensor dev = deviation.multiply(deriv);
  dev.multiply_i(invstd);
  dev.sum(axes_to_reduce, dgamma);
}

void BatchNormalizationLayer::exportTo(Exporter &exporter,
                                       const ExportMethods &method) const {
  exporter.saveResult(bn_props, method, this);
}

} /* namespace nntrainer */
