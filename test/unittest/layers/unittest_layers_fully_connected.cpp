// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_layers_fully_connected.cpp
 * @date 15 June 2021
 * @brief Fully Connected Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <fc_layer.h>
#include <layers_common_tests.h>

auto semantic_fc = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>,
  nntrainer::FullyConnectedLayer::type, {"unit=1"}, 0, false);

INSTANTIATE_TEST_CASE_P(FullyConnected, LayerSemantics,
                        ::testing::Values(semantic_fc));

auto fc_basic_plain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>, {"unit=5"},
  "3:1:1:10", "fc_golden_plain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);
auto fc_basic_single_batch = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedLayer>, {"unit=4"},
  "1:1:1:10", "fc_golden_single_batch.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

INSTANTIATE_TEST_CASE_P(FullyConnected, LayerGoldenTest,
                        ::testing::Values(fc_basic_plain,
                                          fc_basic_single_batch));
