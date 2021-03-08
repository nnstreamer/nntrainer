// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file nntrainer_util.cpp
 * @date 10 July 2020
 * @brief NNTrainer/Utilizer C-API Wrapper.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <nntrainer_error.h>
#include <nntrainer_internal.h>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <concat_layer.h>
#include <conv2d_layer.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <loss_layer.h>
#include <output_layer.h>
#include <pooling2d_layer.h>

/**
 * @brief Convert nntrainer API optimizer type to neural network optimizer type
 */
const std::string
ml_optimizer_to_nntrainer_type(ml_train_optimizer_type_e type) {
  switch (type) {
  case ML_TRAIN_OPTIMIZER_TYPE_ADAM:
    return "adam";
  case ML_TRAIN_OPTIMIZER_TYPE_SGD:
    return "sgd";
  case ML_TRAIN_OPTIMIZER_TYPE_UNKNOWN:
  /// fall through intended
  default:
    throw nntrainer::exception::not_supported(
      "[ml_optimmizer_to_nntrainer_type] Not supported type given");
  }

  throw nntrainer::exception::not_supported(
    "[ml_optimmizer_to_nntrainer_type] Not supported type given");
}

/**
 * @brief Convert nntrainer API layer type to neural network layer type
 */
const std::string ml_layer_to_nntrainer_type(ml_train_layer_type_e type) {
  switch (type) {
  case ML_TRAIN_LAYER_TYPE_FC:
    return nntrainer::FullyConnectedLayer::type;
  case ML_TRAIN_LAYER_TYPE_INPUT:
    return nntrainer::InputLayer::type;
  case ML_TRAIN_LAYER_TYPE_BN:
    return nntrainer::BatchNormalizationLayer::type;
  case ML_TRAIN_LAYER_TYPE_CONV2D:
    return nntrainer::Conv2DLayer::type;
  case ML_TRAIN_LAYER_TYPE_POOLING2D:
    return nntrainer::Pooling2DLayer::type;
  case ML_TRAIN_LAYER_TYPE_FLATTEN:
    return nntrainer::FlattenLayer::type;
  case ML_TRAIN_LAYER_TYPE_ACTIVATION:
    return nntrainer::ActivationLayer::type;
  case ML_TRAIN_LAYER_TYPE_ADDITION:
    return nntrainer::AdditionLayer::type;
  case ML_TRAIN_LAYER_TYPE_CONCAT:
    return nntrainer::ConcatLayer::type;
  case ML_TRAIN_LAYER_TYPE_MULTIOUT:
    return nntrainer::OutputLayer::type;
  case ML_TRAIN_LAYER_TYPE_UNKNOWN:
  /// fall through intended
  default:
    throw nntrainer::exception::not_supported(
      "[ml_layer_to_nntrainer_type] Not supported type given");
  }

  throw std::logic_error(
    "[ml_layer_to_nntrainer_type] Control shouldn't reach here");
}
