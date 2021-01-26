// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file	 app_context.cpp
 * @date	 10 November 2020
 * @brief	 This file contains app context related functions and classes that
 * manages the global configuration of the current environment
 * @see		 https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 */
#include <dirent.h>
#include <iostream>
#include <sstream>

#include <app_context.h>
#include <nntrainer_log.h>
#include <util_func.h>

#include <adam.h>
#include <sgd.h>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <concat_layer.h>
#include <conv2d_layer.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <loss_layer.h>
#include <nntrainer_error.h>
#include <output_layer.h>
#include <parse_util.h>
#include <pooling2d_layer.h>

#ifdef ENABLE_TFLITE_BACKBONE
#include <tflite_layer.h>
#endif

#ifdef ENABLE_NNSTREAMER_BACKBONE
#include <nnstreamer_layer.h>
#endif

namespace nntrainer {

std::mutex factory_mutex;

/**
 * @brief initiate global context
 *
 */
static void init_global_context_nntrainer(void) __attribute__((constructor));

/**
 * @brief finialize global context
 *
 */
static void fini_global_context_nntrainer(void) __attribute__((destructor));

static void init_global_context_nntrainer(void) {
  /// @note all layers should be added to the app_context to gaurantee that
  /// createLayer/createOptimizer class is created
  auto &ac = AppContext::Global();

  using OptType = ml::train::OptimizerType;
  ac.registerFactory(ml::train::createOptimizer<SGD>, SGD::type, OptType::SGD);
  ac.registerFactory(ml::train::createOptimizer<Adam>, Adam::type,
                     OptType::ADAM);
  ac.registerFactory(AppContext::unknownFactory<ml::train::Optimizer>,
                     "unknown", OptType::UNKNOWN);

  using LayerType = ml::train::LayerType;
  ac.registerFactory(ml::train::createLayer<InputLayer>, InputLayer::type,
                     LayerType::LAYER_IN);
  ac.registerFactory(ml::train::createLayer<FullyConnectedLayer>,
                     FullyConnectedLayer::type, LayerType::LAYER_FC);
  ac.registerFactory(ml::train::createLayer<BatchNormalizationLayer>,
                     BatchNormalizationLayer::type, LayerType::LAYER_BN);
  ac.registerFactory(ml::train::createLayer<Conv2DLayer>, Conv2DLayer::type,
                     LayerType::LAYER_CONV2D);
  ac.registerFactory(ml::train::createLayer<Pooling2DLayer>,
                     Pooling2DLayer::type, LayerType::LAYER_POOLING2D);
  ac.registerFactory(ml::train::createLayer<FlattenLayer>, FlattenLayer::type,
                     LayerType::LAYER_FLATTEN);
  ac.registerFactory(ml::train::createLayer<ActivationLayer>,
                     ActivationLayer::type, LayerType::LAYER_ACTIVATION);
  ac.registerFactory(ml::train::createLayer<AdditionLayer>, AdditionLayer::type,
                     LayerType::LAYER_ADDITION);
  ac.registerFactory(ml::train::createLayer<OutputLayer>, OutputLayer::type,
                     LayerType::LAYER_MULTIOUT);
  ac.registerFactory(ml::train::createLayer<ConcatLayer>, ConcatLayer::type,
                     LayerType::LAYER_CONCAT);
  ac.registerFactory(ml::train::createLayer<LossLayer>, LossLayer::type,
                     LayerType::LAYER_LOSS);
#ifdef ENABLE_NNSTREAMER_BACKBONE
  ac.registerFactory(ml::train::createLayer<NNStreamerLayer>,
                     NNStreamerLayer::type,
                     LayerType::LAYER_BACKBONE_NNSTREAMER);
#endif
#ifdef ENABLE_TFLITE_BACKBONE
  ac.registerFactory(ml::train::createLayer<TfLiteLayer>, TfLiteLayer::type,
                     LayerType::LAYER_BACKBONE_TFLITE);
#endif
  ac.registerFactory(AppContext::unknownFactory<ml::train::Layer>, "unknown",
                     LayerType::LAYER_UNKNOWN);
}

static void fini_global_context_nntrainer(void) {}

AppContext &AppContext::Global() {
  static AppContext instance;
  return instance;
}

static const std::string func_tag = "[AppContext::setWorkingDirectory] ";

void AppContext::setWorkingDirectory(const std::string &base) {
  DIR *dir = opendir(base.c_str());

  if (!dir) {
    std::stringstream ss;
    ss << func_tag << "path is not directory or has no permission: " << base;
    throw std::invalid_argument(ss.str().c_str());
  }
  closedir(dir);

  char *ret = realpath(base.c_str(), nullptr);

  if (ret == nullptr) {
    std::stringstream ss;
    ss << func_tag << "failed to get canonical path for the path: ";
    throw std::invalid_argument(ss.str().c_str());
  }

  working_path_base = std::string(ret);
  ml_logd("working path base has set: %s", working_path_base.c_str());
  free(ret);
}

const std::string AppContext::getWorkingPath(const std::string &path) {

  /// if path is absolute, return path
  if (path[0] == '/') {
    return path;
  }

  if (working_path_base == std::string()) {
    return path == std::string() ? "." : path;
  }

  return path == std::string() ? working_path_base
                               : working_path_base + "/" + path;
}

} // namespace nntrainer
