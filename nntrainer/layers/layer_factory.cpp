// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_factory.cpp
 * @date   11 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer factory.
 */

#include <layer_factory.h>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <concat_layer.h>
#include <conv2d_layer.h>
#include <cross_entropy_loss_layer.h>
#include <cross_entropy_sigmoid_loss_layer.h>
#include <cross_entropy_softmax_loss_layer.h>
#include <embedding.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <mse_loss_layer.h>
#include <nntrainer_error.h>
#include <output_layer.h>
#include <parse_util.h>
#include <pooling2d_layer.h>
#include <rnn.h>
#include <split_layer.h>
#include <time_dist.h>

#ifdef ENABLE_TFLITE_BACKBONE
#include <tflite_layer.h>
#endif

#ifdef ENABLE_NNSTREAMER_BACKBONE
#include <nnstreamer_layer.h>
#endif

namespace nntrainer {

const std::string layerGetStrType(const LayerType &type) {
  switch (type) {
  case LayerType::LAYER_IN:
    return InputLayer::type;
  case LayerType::LAYER_MULTIOUT:
    return OutputLayer::type;
  case LayerType::LAYER_FC:
    return FullyConnectedLayer::type;
  case LayerType::LAYER_BN:
    return BatchNormalizationLayer::type;
  case LayerType::LAYER_CONV2D:
    return Conv2DLayer::type;
  case LayerType::LAYER_POOLING2D:
    return Pooling2DLayer::type;
  case LayerType::LAYER_FLATTEN:
    return FlattenLayer::type;
  case LayerType::LAYER_ACTIVATION:
    return ActivationLayer::type;
  case LayerType::LAYER_ADDITION:
    return AdditionLayer::type;
  case LayerType::LAYER_CONCAT:
    return ConcatLayer::type;
#ifdef ENABLE_NNSTREAMER_BACKBONE
  case LayerType::LAYER_BACKBONE_NNSTREAMER:
    return NNStreamerLayer::type;
#endif
#ifdef ENABLE_TFLITE_BACKBONE
  case LayerType::LAYER_BACKBONE_TFLITE:
    return TfLiteLayer::type;
#endif
  case LayerType::LAYER_EMBEDDING:
    return EmbeddingLayer::type;
  case LayerType::LAYER_TIME_DIST:
    return TimeDistLayer::type;
  case LayerType::LAYER_SPLIT:
    return SplitLayer::type;
  /** Loss layers */
  case LayerType::LAYER_LOSS_MSE:
    return MSELossLayer::type;
  case LayerType::LAYER_LOSS_CROSS_ENTROPY:
    return CrossEntropyLossLayer::type;
  case LayerType::LAYER_LOSS_CROSS_ENTROPY_SOFTMAX:
    return CrossEntropySoftmaxLossLayer::type;
  case LayerType::LAYER_LOSS_CROSS_ENTROPY_SIGMOID:
    return CrossEntropySigmoidLossLayer::type;
  case LayerType::LAYER_UNKNOWN:
    /** fallthrough intended */
  default:
    throw std::invalid_argument("Unknown type for the layer");
  }

  throw std::runtime_error("Control should not reach here");
}

/**
 * @brief Factory creator with constructor
 */
// std::unique_ptr<Layer> createLayer(const std::string &type) {
//
//   if (istrequal(type, InputLayer::type))
//     return std::make_unique<InputLayer>();
//   //  if (istrequal(type, OutputLayer::type))
//   //    return std::make_unique<OutputLayer>();
//   if (istrequal(type, FullyConnectedLayer::type))
//     return std::make_unique<FullyConnectedLayer>();
//   //   if (istrequal(type, BatchNormalizationLayer::type))
//   //     return std::make_unique<BatchNormalizationLayer>();
//   //   if (istrequal(type, Conv2DLayer::type))
//   //     return std::make_unique<Conv2DLayer>();
//   //   if (istrequal(type, Pooling2DLayer::type))
//   //     return std::make_unique<Pooling2DLayer>();
//   //   if (istrequal(type, FlattenLayer::type))
//   //     return std::make_unique<FlattenLayer>();
//   //   if (istrequal(type, ActivationLayer::type))
//   //     return std::make_unique<ActivationLayer>();
//   //   if (istrequal(type, AdditionLayer::type))
//   //     return std::make_unique<AdditionLayer>();
//   //   if (istrequal(type, ConcatLayer::type))
//   //     return std::make_unique<ConcatLayer>();
//   // #ifdef ENABLE_NNSTREAMER_BACKBONE
//   //   if (istrequal(type, NNStreamerLayer::type))
//   //     return std::make_unique<NNStreamerLayer>();
//   // #endif
//   // #ifdef ENABLE_TFLITE_BACKBONE
//   //   if (istrequal(type, TfLiteLayer::type))
//   //     return std::make_unique<TfLiteLayer>();
//   // #endif
//   //   if (istrequal(type, ConcatLayer::type))
//   //     return std::make_unique<ConcatLayer>();
//   //   if (istrequal(type, OutputLayer::type))
//   //     return std::make_unique<OutputLayer>();
//   //   if (istrequal(type, EmbeddingLayer::type))
//   //     return std::make_unique<EmbeddingLayer>();
//   //   if (istrequal(type, RNNLayer::type))
//   //     return std::make_unique<RNNLayer>();
//   //   if (istrequal(type, TimeDistLayer::type))
//   //     return std::make_unique<TimeDistLayer>();
//   std::stringstream ss;
//   ss << "Unsupported type given, type: " << type;
//   throw std::invalid_argument(ss.str().c_str());
// }

} // namespace nntrainer
