// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_pooling.cpp
 * @date 7 July 2021
 * @brief Activation Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <activation_layer.h>
#include <layers_common_tests.h>

auto semantic_activation_relu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=relu"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_swish = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=swish"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_gelu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=gelu"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_sigmoid = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=sigmoid"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_softmax = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=softmax"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_tanh = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=tanh"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_activation_none = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ActivationLayer>,
  nntrainer::ActivationLayer::type, {"activation=none"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(
  Activation, LayerSemantics,
  ::testing::Values(semantic_activation_relu, semantic_activation_swish,
                    semantic_activation_gelu, semantic_activation_sigmoid,
                    semantic_activation_softmax, semantic_activation_tanh,
                    semantic_activation_none));
