// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_dropout.cpp
 * @date 15 October 2021
 * @brief Dropout Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <dropout.h>
#include <layers_common_tests.h>

auto semantic_dropout = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>,
  nntrainer::DropOutLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Dropout, LayerSemantics,
                     ::testing::Values(semantic_dropout));

auto dropout_inference_option =
  LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
  LayerGoldenTestParamOptions::FORWARD_MODE_INFERENCE;

auto dropout_20_training = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.2"},
  "2:3:2:3", "dropout_20_training.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT |
    LayerGoldenTestParamOptions::DROPOUT_MATCH_60_PERCENT,
  "nchw", "fp32");

auto dropout_20_inference = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.2"},
  "2:3:2:3", "dropout_20_inference.nnlayergolden", dropout_inference_option,
  "nchw", "fp32");

auto dropout_0_training = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=0.0"},
  "2:3:2:3", "dropout_0_training.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto dropout_100_training = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::DropOutLayer>, {"dropout_rate=1.0"},
  "2:3:2:3", "dropout_100_training.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

GTEST_PARAMETER_TEST(Dropout, LayerGoldenTest,
                     ::testing::Values(dropout_20_training, dropout_0_training,
                                       dropout_100_training,
                                       dropout_20_inference));
