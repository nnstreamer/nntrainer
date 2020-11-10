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
  case ML_TRAIN_LAYER_TYPE_UNKNOWN:
  /// fall through intended
  default:
    throw nntrainer::exception::not_supported(
      "[ml_layer_to_nntrainer_type] Not supported type given");
  }

  throw std::logic_error(
    "[ml_layer_to_nntrainer_type] Control shouldn't reach here");
}
