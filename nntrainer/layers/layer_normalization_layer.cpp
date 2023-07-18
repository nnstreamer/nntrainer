// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   layer_normalization_layer.cpp
 * @date   25 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1607.06450
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Layer Normalization Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <numeric>

#include <layer_context.h>
#include <layer_normalization_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LNParams {
  gamma,
  beta,
  deviation,
  variance,
  inv_std_dev,
  temp_origin_size,
  temp_normalized_size,
};

LayerNormalizationLayer::LayerNormalizationLayer() :
  Layer(),
  layer_normalization_props(
    std::vector<props::Axis>(), props::Epsilon(), props::BNPARAMS_GAMMA_INIT(),
    props::BNPARAMS_BETA_INIT(), props::WeightDecay(), props::BiasDecay()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void LayerNormalizationLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument(
      "Only one input is allowed for layer normalization layer");
  }

  auto gamma_initializer =
    std::get<props::BNPARAMS_GAMMA_INIT>(layer_normalization_props).get();
  auto beta_initializer =
    std::get<props::BNPARAMS_BETA_INIT>(layer_normalization_props).get();
  auto weight_decay = std::get<props::WeightDecay>(layer_normalization_props);
  auto bias_decay = std::get<props::BiasDecay>(layer_normalization_props);

  auto const &input_dim = context.getInputDimensions()[0];
  context.setOutputDimensions({input_dim});

  std::vector<props::Axis> axes_prop =
    std::get<std::vector<props::Axis>>(layer_normalization_props);

  NNTR_THROW_IF(axes_prop.empty(), std::invalid_argument)
    << "[Layer normalization]axis property is empty";

  normalize_axes.insert(normalize_axes.end(), axes_prop.begin(),
                        axes_prop.end());
  std::sort(normalize_axes.begin(), normalize_axes.end());
  normalize_axes.erase(
    std::unique(normalize_axes.begin(), normalize_axes.end()),
    normalize_axes.end());

  TensorDim normalize_dim(context.getFormat(), context.getWeightDataType());
  for (unsigned int axis : normalize_axes) {
    normalize_dim.setTensorDim(axis, input_dim.getTensorDim(axis));
  }

  wt_idx[LNParams::gamma] = context.requestWeight(
    normalize_dim, gamma_initializer, WeightRegularizer::NONE, 1.0f,
    weight_decay, "gamma", true);
  wt_idx[LNParams::beta] = context.requestWeight(
    normalize_dim, beta_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
    "beta", true);

  TensorDim remain_dim(context.getFormat(), context.getWeightDataType());
  std::vector<unsigned int> total_axes;
  total_axes.resize(ml::train::TensorDim::MAXDIM);
  std::iota(total_axes.begin(), total_axes.end(), 0u);
  std::set_difference(total_axes.begin(), total_axes.end(),
                      normalize_axes.begin(), normalize_axes.end(),
                      std::back_inserter(remain_axes));
  for (unsigned int axis : remain_axes) {
    remain_dim.setTensorDim(axis, input_dim.getTensorDim(axis));
  }

  /** caches the deviation -> input - avg(input) */
  wt_idx[LNParams::deviation] =
    context.requestTensor(input_dim, "deviation", Tensor::Initializer::NONE,
                          false, TensorLifespan::ITERATION_LIFESPAN);
  /** caches variance + epsilon as well */
  wt_idx[LNParams::variance] =
    context.requestTensor(remain_dim, "variance", Tensor::Initializer::NONE,
                          false, TensorLifespan::ITERATION_LIFESPAN);
  /** caches the inverse standard deviation */
  wt_idx[LNParams::inv_std_dev] =
    context.requestTensor(remain_dim, "inv_std_dev", Tensor::Initializer::NONE,
                          false, TensorLifespan::ITERATION_LIFESPAN);

  /** temporary tensor (origin size) */
  wt_idx[LNParams::temp_origin_size] = context.requestTensor(
    input_dim, "temp_origin_size", Tensor::Initializer::NONE, false,
    TensorLifespan::CALC_DERIV_LIFESPAN);
  /** temporary tensor (normalized size) */
  wt_idx[LNParams::temp_normalized_size] = context.requestTensor(
    remain_dim, "temp_normalized_size", Tensor::Initializer::NONE, false,
    TensorLifespan::CALC_DERIV_LIFESPAN);
}

void LayerNormalizationLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, layer_normalization_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[Layer Normalization Layer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void LayerNormalizationLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  const float epsilon =
    std::get<props::Epsilon>(layer_normalization_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &gamma = context.getWeight(wt_idx[LNParams::gamma]);
  Tensor &beta = context.getWeight(wt_idx[LNParams::beta]);

  Tensor &deviation = context.getTensor(wt_idx[LNParams::deviation]);
  Tensor &variance = context.getTensor(wt_idx[LNParams::variance]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[LNParams::inv_std_dev]);

  Tensor &temp_full_size = output;
  Tensor &temp_norm_size = inv_std_dev;

  input.average(normalize_axes, temp_norm_size);
  input.subtract(temp_norm_size, deviation);

  deviation.pow(2.0, temp_full_size);
  temp_full_size.average(normalize_axes, variance);

  variance.add_i(epsilon);
  variance.pow(-0.5, inv_std_dev);

  deviation.multiply(inv_std_dev, output);
  output.multiply_i(gamma);
  output.add_i(beta);
}

void LayerNormalizationLayer::incremental_forwarding(RunLayerContext &context,
                                                     unsigned int from,
                                                     unsigned int to,
                                                     bool training) {
  const float epsilon =
    std::get<props::Epsilon>(layer_normalization_props).get();

  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &gamma = context.getWeight(wt_idx[LNParams::gamma]);
  Tensor &beta = context.getWeight(wt_idx[LNParams::beta]);

  Tensor &deviation = context.getTensor(wt_idx[LNParams::deviation]);
  Tensor &variance = context.getTensor(wt_idx[LNParams::variance]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[LNParams::inv_std_dev]);

  // @todo: consider NHWC format
  bool is_height_normalize =
    std::find(normalize_axes.begin(), normalize_axes.end(), 1) !=
        normalize_axes.end()
      ? true
      : false;

  TensorDim input_dim = input.getDim();
  TensorDim output_dim = output.getDim();
  TensorDim normalize_dim = gamma.getDim();
  TensorDim remain_dim = variance.getDim();

  TensorDim input_step_dim = {input_dim.batch(), input_dim.channel(), to - from,
                              input_dim.width()};
  TensorDim output_step_dim = {output_dim.batch(), output_dim.channel(),
                               to - from, output_dim.width()};
  TensorDim normalize_step_dim = {
    normalize_dim.batch(), normalize_dim.channel(),
    is_height_normalize ? to - from : 1, normalize_dim.width()};
  TensorDim remain_step_dim = {remain_dim.batch(), remain_dim.channel(),
                               is_height_normalize ? 1 : to - from,
                               remain_dim.width()};

  // @todo: set reset stride as false. This implementation only works when batch
  // size is 1
  const Tensor input_step = input.getSharedDataTensor(
    input_step_dim, from * input_step_dim.width(), true);
  Tensor output_step = output.getSharedDataTensor(
    output_step_dim, from * output_step_dim.width(), true);

  Tensor gamma_step = gamma.getSharedDataTensor(
    normalize_step_dim,
    is_height_normalize ? from * normalize_step_dim.width() : 0, true);
  Tensor beta_step = beta.getSharedDataTensor(
    normalize_step_dim,
    is_height_normalize ? from * normalize_step_dim.width() : 0, true);

  Tensor deviation_step = deviation.getSharedDataTensor(
    input_step_dim, from * input_step_dim.width(), true);
  Tensor variance_step = variance.getSharedDataTensor(
    remain_step_dim, is_height_normalize ? 0 : from * remain_step_dim.width(),
    true);
  Tensor inv_std_dev_step = inv_std_dev.getSharedDataTensor(
    remain_step_dim, is_height_normalize ? 0 : from * remain_step_dim.width(),
    true);

  Tensor &temp_full_size = output_step;
  Tensor &temp_norm_size = inv_std_dev_step;

  input_step.average(normalize_axes, temp_norm_size);
  input_step.subtract(temp_norm_size, deviation_step);

  deviation_step.pow(2.0f, temp_full_size);
  temp_full_size.average(normalize_axes, variance_step);

  variance_step.add_i(epsilon);
  variance_step.pow(-0.5f, inv_std_dev_step);

  deviation_step.multiply(inv_std_dev_step, output_step);
  output_step.multiply_i(gamma_step);
  output_step.add_i(beta_step);
}

void LayerNormalizationLayer::calcDerivative(RunLayerContext &context) {
  const bool trainable = context.getTrainable();

  TensorDim::TensorType weight_tensor_type =
    context.getWeight(wt_idx[LNParams::gamma]).getTensorType();

  Tensor empty;
  empty.setTensorType(weight_tensor_type);

  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  const Tensor &gamma = context.getWeight(wt_idx[LNParams::gamma]);
  Tensor &d_gamma =
    trainable ? context.getWeightGrad(wt_idx[LNParams::gamma]) : empty;

  Tensor &deviation = context.getTensor(wt_idx[LNParams::deviation]);
  Tensor &variance = context.getTensor(wt_idx[LNParams::variance]);
  Tensor &inv_std_dev = context.getTensor(wt_idx[LNParams::inv_std_dev]);

  Tensor &temp_origin_size =
    context.getTensor(wt_idx[LNParams::temp_origin_size]);
  Tensor &temp_normalized_size =
    context.getTensor(wt_idx[LNParams::temp_normalized_size]);

  incoming_derivative.multiply(deviation, temp_origin_size);
  temp_origin_size.average(normalize_axes, temp_normalized_size);
  temp_normalized_size.divide_i(variance);
  deviation.multiply_i(temp_normalized_size);

  if (trainable) {
    /** calculate d_gamma */
    temp_origin_size.multiply_i(inv_std_dev);
    temp_origin_size.sum(remain_axes, d_gamma);
  }
  incoming_derivative.average(normalize_axes, temp_normalized_size);
  incoming_derivative.subtract(temp_normalized_size, outgoing_derivative);
  outgoing_derivative.subtract_i(deviation);

  inv_std_dev.multiply_i(gamma);
  outgoing_derivative.multiply_i(inv_std_dev);
}

void LayerNormalizationLayer::calcGradient(RunLayerContext &context) {
  /** d_gamma is calculated in calcDerivative. d_beta is calculated here */
  const Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &d_beta = context.getWeightGrad(wt_idx[LNParams::beta]);

  incoming_derivative.sum(remain_axes, d_beta);
}

void LayerNormalizationLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(layer_normalization_props, method, this);
}

void LayerNormalizationLayer::setBatch(RunLayerContext &context,
                                       unsigned int batch) {
  context.updateTensor(wt_idx[LNParams::deviation], batch);
  context.updateTensor(wt_idx[LNParams::variance], batch);
  context.updateTensor(wt_idx[LNParams::inv_std_dev], batch);
  context.updateTensor(wt_idx[LNParams::temp_origin_size], batch);
  context.updateTensor(wt_idx[LNParams::temp_normalized_size], batch);
}

} /* namespace nntrainer */
