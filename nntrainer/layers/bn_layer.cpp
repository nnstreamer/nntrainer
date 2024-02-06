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
#include <layer_context.h>
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
  bn_props(props::Epsilon(), props::BNPARAMS_MU_INIT(),
           props::BNPARAMS_VAR_INIT(), props::BNPARAMS_BETA_INIT(),
           props::BNPARAMS_GAMMA_INIT(), props::Momentum(), props::Axis(),
           props::WeightDecay(), props::BiasDecay()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

/// @todo add multiple axis support
void BatchNormalizationLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Only one input is allowed for batch normalization layer";

  auto &bnparams_mu = std::get<props::BNPARAMS_MU_INIT>(bn_props);
  auto &bnparams_var = std::get<props::BNPARAMS_VAR_INIT>(bn_props);
  auto &bnparams_beta = std::get<props::BNPARAMS_BETA_INIT>(bn_props);
  auto &bnparams_gamma = std::get<props::BNPARAMS_GAMMA_INIT>(bn_props);
  auto &weight_decay = std::get<props::WeightDecay>(bn_props);
  auto &bias_decay = std::get<props::BiasDecay>(bn_props);

  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  context.setOutputDimensions(context.getInputDimensions());

  TensorDim dim(context.getFormat(), context.getWeightDataType());

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
    context.requestWeight(dim, bnparams_mu, WeightRegularizer::NONE, 1.0f, 0.0f,
                          "moving_mean", false);
  wt_idx[BNParams::var] =
    context.requestWeight(dim, bnparams_var, WeightRegularizer::NONE, 1.0f,
                          0.0f, "moving_variance", false);
  wt_idx[BNParams::gamma] =
    context.requestWeight(dim, bnparams_gamma, WeightRegularizer::NONE, 1.0f,
                          weight_decay, "gamma", true);
  wt_idx[BNParams::beta] =
    context.requestWeight(dim, bnparams_beta, WeightRegularizer::NONE, 1.0f,
                          bias_decay, "beta", true);

  /**
   * caches the deviation -> input - avg(input)
   * @todo check if avoiding this storage and adding dependency on input (no
   * more in-place calculation) can save memory during memory optimization.
   */
  wt_idx[BNParams::deviation] =
    context.requestTensor(in_dim, "deviation", Tensor::Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  /** caches the inverse standard deviation */
  wt_idx[BNParams::invstd] =
    context.requestTensor(dim, "invstd", Tensor::Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  /**
   * Temporary tensor to store the full sized tensors in order to allow batch
   * norm to execute in-place. Running in-place leads to same memory footprint
   * for this layer in its backwarding, but reduces the peak memory requirement
   * as the output of this layer need not be stored all the time.
   */
  wt_idx[BNParams::t_full] =
    context.requestTensor(in_dim, "tensor_full", Tensor::Initializer::NONE,
                          false, TensorLifespan::CALC_DERIV_LIFESPAN);
  /**
   * caches variance + epsilon as well.
   */
  wt_idx[BNParams::cvar] =
    context.requestTensor(dim, "cvar", Tensor::Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  /**
   * Temporary tensor to store the reduced tensors along the axes_to_reduce.
   */
  wt_idx[BNParams::t_reduced] =
    context.requestTensor(dim, "tensor_reduced", Tensor::Initializer::NONE,
                          false, TensorLifespan::FORWARD_DERIV_LIFESPAN);
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
  if (input_.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    TensorDim mu_dim = mu.getDim();
    mu_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor mu32(mu_dim, true);
    mu32.copyData(mu);

    TensorDim var_dim = var.getDim();
    var_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor var32(var_dim, true);
    var32.copyData(var);

    TensorDim gamma_dim = gamma.getDim();
    gamma_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor gamma32(gamma_dim, true);
    gamma32.copyData(gamma);

    TensorDim beta_dim = beta.getDim();
    beta_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor beta32(beta_dim, true);
    beta32.copyData(beta);

    TensorDim input_dim = input_.getDim();
    input_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor input_32(input_dim, true);
    input_32.copyData(input_);

    TensorDim hidden_dim = hidden_.getDim();
    hidden_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor hidden_32(hidden_dim, true);
    hidden_32.copyData(hidden_);
    Tensor t_full32 = hidden_32;

    TensorDim deviation_dim = deviation.getDim();
    deviation_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor deviation32(deviation_dim, true);
    deviation32.copyData(deviation);

    TensorDim dim_invstd = invstd.getDim();
    dim_invstd.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor invstd32(dim_invstd, true);
    invstd32.copyData(invstd);

    TensorDim t_reduced_dim = t_reduced.getDim();
    t_reduced_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor t_reduced32(t_reduced_dim, true);
    t_reduced32.copyData(t_reduced);

    TensorDim cvar_dim = cvar.getDim();
    cvar_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor cvar32(cvar_dim, true);
    cvar32.copyData(cvar);

    if (training) {
      input_32.average(axes_to_reduce, t_reduced32);
      input_32.subtract(t_reduced32, deviation32);

      mu32.multiply_i(momentum);
      mu32.add_i(t_reduced32, 1 - momentum);

      deviation32.pow(2.0f, t_full32);
      t_full32.average(axes_to_reduce, cvar32);

      var32.multiply_i(momentum);
      var32.add_i(cvar32, 1 - momentum);

      cvar32.add_i(epsilon);
      cvar32.pow(-0.5f, invstd32);
    } else {
      input_32.subtract(mu32, deviation32);
      /** @todo do below 2 lines only for first iteration */
      var32.add(epsilon, invstd32);
      invstd32.pow_i(-0.5f);
    }

    deviation32.multiply(invstd32, hidden_32);
    hidden_32.multiply_i(gamma32);
    hidden_32.add_i(beta32);

    mu.copyData(mu32);
    var.copyData(var32);
    gamma.copyData(gamma32);
    beta.copyData(beta32);
    input_.copyData(input_32);
    hidden_.copyData(hidden_32);
    deviation.copyData(deviation32);
    invstd.copyData(invstd32);
    t_reduced.copyData(t_reduced32);
    cvar.copyData(cvar32);
#else
    throw std::runtime_error("enable-fp16 is not enabled");
#endif
  } else {
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
}

void BatchNormalizationLayer::calcDerivative(RunLayerContext &context) {

  Tensor &gamma = context.getWeight(wt_idx[BNParams::gamma]);
  const Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &deviation = context.getTensor(wt_idx[BNParams::deviation]);
  Tensor &invstd = context.getTensor(wt_idx[BNParams::invstd]);
  Tensor &cvar = context.getTensor(wt_idx[BNParams::cvar]);

  Tensor &t_reduced = context.getTensor(wt_idx[BNParams::t_reduced]);
  Tensor &t_full = context.getTensor(wt_idx[BNParams::t_full]);
  if (deriv.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    TensorDim gamma_dim = gamma.getDim();
    gamma_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor gamma32(gamma_dim, true);
    gamma32.copyData(gamma);

    TensorDim deriv_dim = deriv.getDim();
    deriv_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor deriv32(deriv_dim, true);
    deriv32.copyData(deriv);

    TensorDim dx_dim = dx.getDim();
    dx_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor dx32(dx_dim, true);
    dx32.copyData(dx);

    TensorDim deviation_dim = deviation.getDim();
    deviation_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor deviation32(deviation_dim, true);
    deviation32.copyData(deviation);

    TensorDim invstd_dim = invstd.getDim();
    invstd_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor invstd32(invstd_dim, true);
    invstd32.copyData(invstd);

    TensorDim cvar_dim = cvar.getDim();
    cvar_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor cvar32(cvar_dim, true);
    cvar32.copyData(cvar);

    TensorDim t_reduced_dim = t_reduced.getDim();
    t_reduced_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor t_reduced32(t_reduced_dim, true);
    t_reduced32.copyData(t_reduced);

    TensorDim t_full_dim = t_full.getDim();
    t_full_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor t_full32(t_full_dim, true);
    t_full32.copyData(t_full);

    deviation32.multiply(deriv32, t_full32);
    t_full32.average(axes_to_reduce, t_reduced32);
    t_reduced32.divide_i(cvar32);
    deviation32.multiply_i(t_reduced32);

    if (context.getTrainable()) {
      /**
       * This calculates dgamma tensor.
       */
      Tensor &dgamma = context.getWeightGrad(wt_idx[BNParams::gamma]);
      TensorDim dgamma_dim = dgamma.getDim();
      dgamma_dim.setDataType(ml::train::TensorDim::DataType::FP32);
      Tensor dgamma32(dgamma_dim, true);
      dgamma32.copyData(dgamma);

      t_full32.multiply_i(invstd32);
      t_full32.sum(axes_to_reduce, dgamma32);
      dgamma.copyData(dgamma32);

      /**
       * This implementation depends on the pre-calculated dbeta calculated.
       */
      Tensor &dbeta = context.getWeightGrad(wt_idx[BNParams::beta]);
      TensorDim dbeta_dim = dbeta.getDim();
      dbeta_dim.setDataType(ml::train::TensorDim::DataType::FP32);
      Tensor dbeta32(dbeta_dim, true);
      dbeta32.copyData(dbeta);
      dbeta32.divide(divider, t_reduced32);
    } else {
      deriv32.average(axes_to_reduce, t_reduced32);
    }

    deriv32.subtract(t_reduced32, dx32);
    dx32.subtract_i(deviation32);

    invstd32.multiply_i(gamma32);
    dx32.multiply_i(invstd32);

    gamma.copyData(gamma32);
    dx.copyData(dx32);
    deviation.copyData(deviation32);
    invstd.copyData(invstd32);
    cvar.copyData(cvar32);
    t_reduced.copyData(t_reduced32);
    t_full.copyData(t_full32);
#else
    throw std::runtime_error("enable-fp16 is not enabled");
#endif
  } else {
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
}

void BatchNormalizationLayer::calcGradient(RunLayerContext &context) {
  /** dgamma is calculated in calcDerivative. dbeta is calculated here */
  Tensor &dbeta = context.getWeightGrad(wt_idx[BNParams::beta]);
  const Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  if (deriv.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    TensorDim dbeta_dim = dbeta.getDim();
    dbeta_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor dbeta32(dbeta_dim, true);
    dbeta32.copyData(dbeta);

    TensorDim deriv_dim = deriv.getDim();
    deriv_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    Tensor deriv32(deriv_dim, true);
    deriv32.copyData(deriv);

    deriv32.sum(axes_to_reduce, dbeta32);
    dbeta.copyData(dbeta32);
#else
    throw std::runtime_error("enable-fp16 is not enabled");
#endif
  } else {
    deriv.sum(axes_to_reduce, dbeta);
  }
}

void BatchNormalizationLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(bn_props, method, this);
}

void BatchNormalizationLayer::setBatch(RunLayerContext &context,
                                       unsigned int batch) {
  context.updateTensor(wt_idx[BNParams::deviation], batch);
  context.updateTensor(wt_idx[BNParams::t_full], batch);

  /// reset divider
  divider = 1;
  auto input_dim = context.getInput(0).getDim();
  for (auto axis : axes_to_reduce) {
    if (axis == 0) {
      /// @note input dim batch is not updated, it will be more sensible we
      /// update batch before any node comes to this spot
      divider *= batch;
    }
    divider *= input_dim.getTensorDim(axis);
  }
}

} /* namespace nntrainer */
