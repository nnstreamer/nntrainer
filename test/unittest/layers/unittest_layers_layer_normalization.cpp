// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file unittest_layers_loss.cpp
 * @date 29 July 2022
 * @brief Layer Normalization Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layer_normalization_layer.h>
#include <layers_common_tests.h>

auto semantic_layer_normalization = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>,
  nntrainer::LayerNormalizationLayer::type, {"axis=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(LayerNormalization, LayerSemantics,
                     ::testing::Values(semantic_layer_normalization));

auto ln_axis_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1"},
  "2:4:2:3", "ln_axis_1.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto ln_axis_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=2"},
  "2:4:2:3", "ln_axis_2.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto ln_axis_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=3"},
  "2:4:2:3", "ln_axis_3.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto ln_axis_1_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 2"},
  "2:4:2:3", "ln_axis_1_2.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto ln_axis_2_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=2, 3"},
  "2:4:2:3", "ln_axis_2_3.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto ln_axis_1_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 3"},
  "2:4:2:3", "ln_axis_1_3.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto ln_axis_1_2_3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LayerNormalizationLayer>, {"axis=1, 2, 3"},
  "2:4:2:3", "ln_axis_1_2_3.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

GTEST_PARAMETER_TEST(LayerNormalization, LayerGoldenTest,
                     ::testing::Values(ln_axis_1, ln_axis_2, ln_axis_3,
                                       ln_axis_1_2, ln_axis_2_3, ln_axis_1_3,
                                       ln_axis_1_2_3));
