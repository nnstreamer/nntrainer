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
 * @file	qkv_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <qkv_layer.h>

#include <cstring>
#include <vector>

#include "htp_decode_hook.h"

#include <cpu_backend.h>
#include <engine.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum QKVParams { Q, K, V };
/** weight_idx slots; the file's order is q, q_gamma, k, k_gamma, v */
enum QKVWeights { WQ, WQ_GAMMA, WK, WK_GAMMA, WV };

QKVLayer::QKVLayer() :
  LayerImpl(),
  qkv_props(props::QUnit(), props::KUnit(), props::VUnit(),
            props::FeatureSize(), nntrainer::props::Epsilon()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

void QKVLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &q_unit = std::get<props::QUnit>(qkv_props).get();
  const auto &k_unit = std::get<props::KUnit>(qkv_props).get();
  const auto &v_unit = std::get<props::VUnit>(qkv_props).get();

  std::vector<nntrainer::TensorDim> output_dims(3);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];

  /** Q out */
  output_dims[QKVParams::Q] = in_dim;
  is_nchw ? output_dims[QKVParams::Q].width(q_unit)
          : output_dims[QKVParams::Q].channel(q_unit);
  output_dims[QKVParams::Q].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** K out */
  output_dims[QKVParams::K] = in_dim;
  is_nchw ? output_dims[QKVParams::K].width(k_unit)
          : output_dims[QKVParams::K].channel(k_unit);
  output_dims[QKVParams::K].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** V out */
  output_dims[QKVParams::V] = in_dim;
  is_nchw ? output_dims[QKVParams::V].width(v_unit)
          : output_dims[QKVParams::V].channel(v_unit);
  output_dims[QKVParams::V].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  feature_size = std::get<props::FeatureSize>(qkv_props).empty()
                   ? 0
                   : std::get<props::FeatureSize>(qkv_props).get();
  NNTR_THROW_IF(feature_size != 0 &&
                  (q_unit % feature_size != 0 || k_unit % feature_size != 0),
                std::invalid_argument)
    << "qkv_layer: feature_size must divide q_unit and k_unit";

  // gamma is unquantized FP32 on disk, requested FP32 regardless of the
  // activation dtype -- ReshapedRMSNormLayer's rule, and its weight.
  const nntrainer::TensorDim gamma_dim(
    1, 1, 1, feature_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  auto request_gamma = [&](const char *name) {
    return context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, name, true);
  };

  /** Q */
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : q_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? q_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[WQ] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "qweight", true);
  if (feature_size)
    weight_idx[WQ_GAMMA] = request_gamma("q_norm_gamma");

  /** K */
  weight_dim.width(k_unit);
  weight_idx[WK] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "kweight", true);
  if (feature_size)
    weight_idx[WK_GAMMA] = request_gamma("k_norm_gamma");

  /** V */
  weight_dim.width(v_unit);
  weight_idx[WV] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "vweight", true);

  // The norm kernel is not in-place (its pointers are __restrict), so the
  // projections land here first and the outputs hold the normed rows.
  if (feature_size) {
    tensor_idx[QKVParams::Q] = context.requestTensor(
      output_dims[QKVParams::Q], "q_raw", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
    tensor_idx[QKVParams::K] = context.requestTensor(
      output_dims[QKVParams::K], "k_raw", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }
}

/**
 * @brief out = rms_norm(in) * gamma per feature_size-wide head, over the
 *        first @a rows rows. ReshapedRMSNormLayer::incremental_forwarding's
 *        FP32 arithmetic, call for call.
 */
static void headNorm(nntrainer::Tensor &in, nntrainer::Tensor &out,
                     nntrainer::Tensor &gamma, unsigned int rows,
                     unsigned int feature_size, float epsilon) {
  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "qkv_layer: the folded norm is FP32 only";
  const unsigned int width = in.getDim().width();
  ml::train::TensorDim dim(1, 1, rows * (width / feature_size), feature_size);
  nntrainer::Tensor in_step = in.getSharedDataTensor(dim, 0, true);
  nntrainer::Tensor out_step = out.getSharedDataTensor(dim, 0, true);
  nntrainer::rms_norm_wrt_width_fp32_intrinsic(
    in_step.getData<float>(), out_step.getData<float>(), dim.height(),
    dim.width(), epsilon);
  out_step.multiply_i(gamma);
}

void QKVLayer::exportTo(nntrainer::Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(qkv_props, method, this);
}

void QKVLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, qkv_props);
  LayerImpl::setProperty(remain_props);
}

void QKVLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  incremental_forwarding(context, 0,
                         context.getInput(SINGLE_INOUT_IDX).height(), training);
}

void QKVLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {
  nntrainer::Tensor &Qweight = context.getWeight(weight_idx[WQ]);
  nntrainer::Tensor &Kweight = context.getWeight(weight_idx[WK]);
  nntrainer::Tensor &Vweight = context.getWeight(weight_idx[WV]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  // With the norm folded in, the projections go to q_raw / k_raw and the
  // outputs receive the normed rows below.
  nntrainer::Tensor &Qhidden_ = feature_size
                                  ? context.getTensor(tensor_idx[QKVParams::Q])
                                  : context.getOutput(QKVParams::Q);
  nntrainer::Tensor &Khidden_ = feature_size
                                  ? context.getTensor(tensor_idx[QKVParams::K])
                                  : context.getOutput(QKVParams::K);
  nntrainer::Tensor &Vhidden_ = context.getOutput(QKVParams::V);

  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  nntrainer::Tensor input_step =
    input_.getSharedDataTensor(input_step_dim, 0, true);

  nntrainer::TensorDim Qhidden_dim = Qhidden_.getDim();
  nntrainer::TensorDim Qhidden_step_dim = Qhidden_.getDim();
  Qhidden_step_dim.batch(1);
  Qhidden_step_dim.height(to - from);
  nntrainer::Tensor Qhidden_step =
    Qhidden_.getSharedDataTensor(Qhidden_step_dim, 0, true);

  nntrainer::TensorDim Khidden_dim = Khidden_.getDim();
  nntrainer::TensorDim Khidden_step_dim = Khidden_.getDim();
  Khidden_step_dim.batch(1);
  Khidden_step_dim.height(to - from);
  nntrainer::Tensor Khidden_step =
    Khidden_.getSharedDataTensor(Khidden_step_dim, 0, true);

  nntrainer::TensorDim Vhidden_dim = Vhidden_.getDim();
  nntrainer::TensorDim Vhidden_step_dim = Vhidden_.getDim();
  Vhidden_step_dim.batch(1);
  Vhidden_step_dim.height(to - from);
  nntrainer::Tensor Vhidden_step =
    Vhidden_.getSharedDataTensor(Vhidden_step_dim, 0, true);

  std::vector<nntrainer::Tensor *> Weights({&Qweight, &Kweight, &Vweight});
  std::vector<nntrainer::Tensor *> Outputs(
    {&Qhidden_step, &Khidden_step, &Vhidden_step});

  input_step.dot(Weights, Outputs);

  if (feature_size) {
    const float epsilon = std::get<nntrainer::props::Epsilon>(qkv_props).get();
    // [#130] one decode row: with QK_NORM resident the HTP norms q | k and
    // keeps the row for the attention hook, so the outputs stay unwritten
    if (to - from == 1 && input_dim.batch() == 1 &&
        Qhidden_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const unsigned int wq = Qhidden_step_dim.width(),
                         wk = Khidden_step_dim.width(),
                         wv = Vhidden_step_dim.width();
      static thread_local std::vector<float> row, gammas;
      row.resize(wq + wk + wv);
      std::memcpy(row.data(), Qhidden_step.getData<float>(),
                  wq * sizeof(float));
      std::memcpy(row.data() + wq, Khidden_step.getData<float>(),
                  wk * sizeof(float));
      std::memcpy(row.data() + wq + wk, Vhidden_step.getData<float>(),
                  wv * sizeof(float));
      gammas.resize(2 * feature_size);
      std::memcpy(gammas.data(),
                  context.getWeight(weight_idx[WQ_GAMMA]).getData<float>(),
                  feature_size * sizeof(float));
      std::memcpy(gammas.data() + feature_size,
                  context.getWeight(weight_idx[WK_GAMMA]).getData<float>(),
                  feature_size * sizeof(float));
      if (htpDecodeQkNorm(from, row.data(), wq + wk + wv, gammas.data(),
                          2 * feature_size, epsilon))
        return;
    }
    headNorm(Qhidden_, context.getOutput(QKVParams::Q),
             context.getWeight(weight_idx[WQ_GAMMA]), to - from, feature_size,
             epsilon);
    headNorm(Khidden_, context.getOutput(QKVParams::K),
             context.getWeight(weight_idx[WK_GAMMA]), to - from, feature_size,
             epsilon);
  }
}

void QKVLayer::calcDerivative(nntrainer::RunLayerContext &context) { return; }

void QKVLayer::calcGradient(nntrainer::RunLayerContext &context) { return; }

void QKVLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim Qoutput_dim = context.getOutput(QKVParams::Q).getDim();
  ml::train::TensorDim Koutput_dim = context.getOutput(QKVParams::K).getDim();
  ml::train::TensorDim Voutput_dim = context.getOutput(QKVParams::V).getDim();

  input_dim.height(input_dimensions[0].height());
  Qoutput_dim.height(input_dimensions[0].height());
  Koutput_dim.height(input_dimensions[0].height());
  Voutput_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(QKVParams::Q, Qoutput_dim);
  context.updateOutput(QKVParams::K, Koutput_dim);
  context.updateOutput(QKVParams::V, Voutput_dim);
  if (feature_size) {
    context.updateTensor(tensor_idx[QKVParams::Q], Qoutput_dim);
    context.updateTensor(tensor_idx[QKVParams::K], Koutput_dim);
  }
}
} // namespace causallm
