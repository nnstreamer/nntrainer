// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.cpp
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <embedding.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

EmbeddingLayer::EmbeddingLayer() :
  LayerImpl(),
  embedding_props(props::InDim(), props::OutDim()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

void EmbeddingLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  const TensorDim &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  NNTR_THROW_IF(input_dim.getDataType() != TensorDim::DataType::FP32,
                std::invalid_argument)
    << "Embedding layer takes only FP32 input data";

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  unsigned int in_dim = std::get<props::InDim>(embedding_props);
  unsigned int out_dim = std::get<props::OutDim>(embedding_props);

  TensorDim output_dim = input_dim;

  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  TensorDim dim = output_dim;

  dim.setTensorType({context.getFormat(), context.getWeightDataType()});

  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  weight_idx = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void EmbeddingLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, embedding_props);
  LayerImpl::setProperty(remain_props);
}

void EmbeddingLayer::forwarding(RunLayerContext &context, bool training) {
  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim = std::get<props::InDim>(embedding_props);
  unsigned int out_dim = std::get<props::OutDim>(embedding_props);

  Tensor &weight = context.getWeight(weight_idx);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  TensorDim out_tensor_dim = TensorDim({1, 1, 1, out_dim});

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());

    Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);
    for (unsigned int i = 0; i < input_.width(); ++i) {
      uint embed_idx = ((uint *)(in_data))[i];
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
      Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * i);
      // float *out_data =
      //   hidden_.getAddress(b * hidden_.getDim().getFeatureLen() + i *
      //   out_dim);
      // float *weight_data =
      //   weight.getAddress(embed_idx * out_dim);
      // Tensor cur_weight = Tensor::Map(weight_data,
      // out_tensor_dim.getDataLen() * 4, out_tensor_dim); Tensor out_tensor =
      // Tensor::Map(out_data, out_tensor_dim.getDataLen() * 4, out_tensor_dim);
      out_tensor.copyData(cur_weight);

      // Assume padding is 0 and index always start from 1.
      // If in_data[i] - 1 < 0, then it skips.
      // if (embed_idx == 0)
      //   continue;

      // float *weight_data =
      //   weight.getAddress(embed_idx * out_dim);
      // float *out_data =
      //   hidden_.getAddress(b * hidden_.getDim().getFeatureLen() + i *
      //   out_dim);

      // std::copy(weight_data, weight_data + out_dim, out_data);
    }
  }
}

void EmbeddingLayer::incremental_forwarding(RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim = std::get<props::InDim>(embedding_props);
  unsigned int out_dim = std::get<props::OutDim>(embedding_props);

  Tensor &weight = context.getWeight(weight_idx);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  TensorDim out_tensor_dim = TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    float *in_data = input_.getAddress<float>(b * input_.getDim().getFeatureLen());

    Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);
    for (unsigned int i = from; i < to; ++i) {
      uint embed_idx = ((uint *)(in_data))[i];
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
      Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * i);
      // float *out_data =
      //   hidden_.getAddress(b * hidden_.getDim().getFeatureLen() + i *
      //   out_dim);
      // float *weight_data =
      //   weight.getAddress(embed_idx * out_dim);
      // Tensor cur_weight = Tensor::Map(weight_data,
      // out_tensor_dim.getDataLen() * 4, out_tensor_dim); Tensor out_tensor =
      // Tensor::Map(out_data, out_tensor_dim.getDataLen() * 4, out_tensor_dim);
      out_tensor.copyData(cur_weight);

      // Assume padding is 0 and index always start from 1.
      // If in_data[i] - 1 < 0, then it skips.
      // if (embed_idx == 0)
      //   continue;

      // float *weight_data =
      //   weight.getAddress(embed_idx * out_dim);
      // float *out_data =
      //   hidden_.getAddress(b * hidden_.getDim().getFeatureLen() + i *
      //   out_dim);

      // std::copy(weight_data, weight_data + out_dim, out_data);
    }
    //hidden_.print(std::cout);
  }
}

void EmbeddingLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void EmbeddingLayer::calcGradient(RunLayerContext &context) {
  unsigned int out_dim = std::get<props::OutDim>(embedding_props);

  Tensor &djdw = context.getWeightGrad(weight_idx);
  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  djdw.setZero();

  // TODO:
  // This is to calculate gradient with current implementation of optimizer.
  // In order to accelerate, we need to better way like using index to weight.

  for (unsigned int b = 0; b < input_.batch(); ++b) {
    float *in_data = input_.getAddress<float>(b * input_.getDim().getFeatureLen());

    for (unsigned int i = 0; i < input_.width(); ++i) {
      uint embed_idx = ((uint *)(in_data))[i];
      // Assume padding is 0 and index always start from 1.
      // If in_data[i] - 1 < 0, then it skips.
      // if (embed_idx == 0)
      //   continue;

      float *djdw_data = djdw.getAddress<float>(embed_idx * out_dim);
      const float *grad_data = derivative_.getAddress<float>(
        b * derivative_.getDim().getFeatureLen() + i * out_dim);

      std::transform(djdw_data, djdw_data + out_dim, grad_data, djdw_data,
                     std::plus<float>());
    }
  }
}

void EmbeddingLayer::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(embedding_props, method, this);
}

} // namespace nntrainer
