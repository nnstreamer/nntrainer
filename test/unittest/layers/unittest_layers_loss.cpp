// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_loss.cpp
 * @date 15 June 2021
 * @brief Loss Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <cross_entropy_loss_layer.h>
#include <cross_entropy_sigmoid_loss_layer.h>
#include <cross_entropy_softmax_loss_layer.h>
#include <layers_common_tests.h>
#include <mse_loss_layer.h>

auto semantic_loss_cross_sigmoid = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CrossEntropySigmoidLossLayer>,
  nntrainer::CrossEntropySigmoidLossLayer::type, {}, 0, false);

auto semantic_loss_cross_softmax = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CrossEntropySoftmaxLossLayer>,
  nntrainer::CrossEntropySoftmaxLossLayer::type, {}, 0, false);

auto semantic_loss_mse =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::MSELossLayer>,
                          nntrainer::MSELossLayer::type, {}, 0, false);

auto semantic_loss_cross = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CrossEntropyLossLayer>,
  nntrainer::CrossEntropyLossLayer::type, {}, 0, true);

INSTANTIATE_TEST_CASE_P(LossCross, LayerSemantics,
                        ::testing::Values(semantic_loss_cross,
                                          semantic_loss_mse,
                                          semantic_loss_cross_softmax,
                                          semantic_loss_cross_sigmoid));
