// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	layer_factory.h
 * @date	7 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is the layer factory.
 */

#ifndef __LAYER_FACTORY_H__
#define __LAYER_FACTORY_H__
#ifdef __cplusplus

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <conv2d_layer.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <layer_internal.h>
#include <loss_layer.h>
#include <pooling2d_layer.h>

namespace nntrainer {

/**
 * @brief Factory creator with constructor
 */
template <typename... Args>
std::unique_ptr<Layer> createLayer(LayerType type, Args... args) {
  switch (type) {
  case LayerType::LAYER_IN:
    return std::make_unique<InputLayer>(args...);
  case LayerType::LAYER_FC:
    return std::make_unique<FullyConnectedLayer>(args...);
  case LayerType::LAYER_BN:
    return std::make_unique<BatchNormalizationLayer>(args...);
  case LayerType::LAYER_CONV2D:
    return std::make_unique<Conv2DLayer>(args...);
  case LayerType::LAYER_POOLING2D:
    return std::make_unique<Pooling2DLayer>(args...);
  case LayerType::LAYER_FLATTEN:
    return std::make_unique<FlattenLayer>(args...);
  case LayerType::LAYER_ACTIVATION:
    return std::make_unique<ActivationLayer>(args...);
  case LayerType::LAYER_ADDITION:
    return std::make_unique<AdditionLayer>(args...);
  case LayerType::LAYER_LOSS:
    return std::make_unique<LossLayer>(args...);
  case LayerType::LAYER_UNKNOWN:
    /** fallthrough intended */
  default:
    throw std::invalid_argument("Unknown type for the layer");
  }
}

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LAYER_FACTORY_H__ */
