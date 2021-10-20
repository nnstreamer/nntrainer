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

enum BNParams {
  mu,
  var,
  gamma,
  beta,
  deviation,
  invstd,
  cvar,
  t_reduced,
  t_full
};

BatchNormalizationLayer::BatchNormalizationLayer() :
  Layer(),
  divider(0),
  wt_idx({0}),
  bn_props(props::Epsilon(), props::BNPARAMS_MU_INIT(),
           props::BNPARAMS_VAR_INIT(), props::BNPARAMS_BETA_INIT(),
           props::BNPARAMS_GAMMA_INIT(), props::Momentum(), props::Axis()) {}

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
  auto &axis_prop = std::get<props::Axis>(bn_props);
  unsigned int axis;
  if (axis_prop.empty())
    axis = in_dim.channel() > 1 ? 1 : 3;
  else
    axis = axis_prop.get();

  /**
   * @todo This can be speedup by employing transpose for convolution. With
   * transpose, the channel dimension can be made last, and the remaining
   * dimensions can be squeezed. This would allow the sum and average to be
   * faster, and no temporary allocations inside them.
   */

  dim.setTensorDim(axis, in_dim.getTensorDim(axis));

  divider = 1;
  for (unsigned int i = 0; i < 4; ++i) {
    if (axis != i) {
      axes_to_reduce.push_back(i);
      divider *= in_dim.getTensorDim(i);
    }
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

  /**
   * caches the deviation -> input - avg(input)
   * @todo check if avoiding this storage and adding dependency on input (no
   * more in-place calculation) can save memory during memory optimization.
   */
  wt_idx[BNParams::deviation] = context.requestTensor(
    in_dim, context.getName() + ":deviation", Tensor::Initializer::NONE, false,
    TensorLifespan::ITERATION_LIFESPAN);
  /** caches the inverse standard deviation */
  wt_idx[BNParams::invstd] = context.requestTensor(
    dim, context.getName() + ":invstd", Tensor::Initializer::NONE, false,
    TensorLifespan::ITERATION_LIFESPAN);
  /**
   * Temporary tensor to store the full sized tensors in order to allow batch
   * norm to execute in-place. Running in-place leads to same memory footprint
   * for this layer in its backwarding, but reduces the peak memory requirement
   * as the output of this layer need not be stored all the time.
   */
  wt_idx[BNParams::t_full] = context.requestTensor(
    in_dim, context.getName() + ":tesnor_full", Tensor::Initializer::NONE,
    false, TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  /**
   * caches variance + epsilon as well.
   */
  wt_idx[BNParams::cvar] = context.requestTensor(
    dim, context.getName() + ":cvar", Tensor::Initializer::NONE, false,
    TensorLifespan::ITERATION_LIFESPAN);
  /**
   * Temporary tensor to store the reduced tensors along the axes_to_reduce.
   */
  wt_idx[BNParams::t_reduced] = context.requestTensor(
    dim, context.getName() + ":tensor_reduced", Tensor::Initializer::NONE,
    false, TensorLifespan::BACKWARD_FUNC_LIFESPAN);
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
  Tensor &invstd = context.getTensor(wt_idx[BNParams::invstd]);

  /** @todo these are not needed for inference, support optimizing these */
  Tensor &t_reduced = context.getTensor(wt_idx[BNParams::t_reduced]);
  /** use hidden_ as temporary tensor before setting the result in hidden */
  Tensor t_full = hidden_;
  Tensor &cvar = context.getTensor(wt_idx[BNParams::cvar]);

  if (training) {
    input_.average(axes_to_reduce, t_reduced);
    input_.subtract(t_reduced, deviation);

    mu.multiply_i(momentum);
    mu.add_i(t_reduced, 1 - momentum);

    deviation.pow(2.0f, t_full);
    t_full.average(axes_to_reduce, cvar);

    var.multiply_i(momentum);
    var.add_i(cvar, 1 - momentum);

    cvar.add_i(epsilon);
    cvar.pow(-0.5f, invstd);
  } else {
    input_.subtract(mu, deviation);
    /** @todo do below 2 lines only for first iteration */
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
  Tensor &invstd = context.getTensor(wt_idx[BNParams::invstd]);
  Tensor &cvar = context.getTensor(wt_idx[BNParams::cvar]);

  Tensor &t_reduced = context.getTensor(wt_idx[BNParams::t_reduced]);
  Tensor &t_full = context.getTensor(wt_idx[BNParams::t_full]);

  deviation.multiply(deriv, t_full);
  t_full.average(axes_to_reduce, t_reduced);
  t_reduced.divide_i(cvar);
  deviation.multiply_i(t_reduced);

  if (context.getTrainable()) {
    /**
     * This calculates dgamma tensor.
     */
    Tensor &dgamma = context.getWeightGrad(wt_idx[BNParams::gamma]);
    t_full.multiply_i(invstd);
    t_full.sum(axes_to_reduce, dgamma);

    /**
     * This implementation depends on the pre-calculated dbeta calculated.
     */
    Tensor &dbeta = context.getWeightGrad(wt_idx[BNParams::beta]);
    dbeta.divide(divider, t_reduced);
  } else {
    deriv.average(axes_to_reduce, t_reduced);
  }

  deriv.subtract(t_reduced, dx);
  dx.subtract_i(deviation);

  invstd.multiply_i(gamma);
  dx.multiply_i(invstd);
}

void BatchNormalizationLayer::calcGradient(RunLayerContext &context) {
  /** dgamma is calculated in calcDerivative. dbeta is calculated here */
  Tensor &dbeta = context.getWeightGrad(wt_idx[BNParams::beta]);
  Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  deriv.sum(axes_to_reduce, dbeta);
}

void BatchNormalizationLayer::exportTo(Exporter &exporter,
                                       const ExportMethods &method) const {
  exporter.saveResult(bn_props, method, this);
}

} /* namespace nntrainer */
