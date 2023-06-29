// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_loss.cpp
 * @date 15 June 2021
 * @brief Batch Normalization Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <bn_layer.h>
#include <layers_common_tests.h>

auto semantic_bn = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::BatchNormalizationLayer>,
  nntrainer::BatchNormalizationLayer::type, {},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(BatchNormalization, LayerSemantics,
                     ::testing::Values(semantic_bn));

auto bn_inference_option = LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
                           LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
                           LayerGoldenTestParamOptions::FORWARD_MODE_INFERENCE;

auto bn_basic_channels_training = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::BatchNormalizationLayer>, {}, "2:4:2:3",
  "bn_channels_training.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto bn_basic_channels_inference = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::BatchNormalizationLayer>, {}, "2:4:2:3",
  "bn_channels_inference.nnlayergolden", bn_inference_option, "nchw", "fp32");

auto bn_basic_width_training = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::BatchNormalizationLayer>, {}, "2:1:1:10",
  "bn_width_training.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto bn_basic_width_inference = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::BatchNormalizationLayer>, {}, "2:1:1:10",
  "bn_width_inference.nnlayergolden", bn_inference_option, "nchw", "fp32");

GTEST_PARAMETER_TEST(BatchNormalization, LayerGoldenTest,
                     ::testing::Values(bn_basic_channels_training,
                                       bn_basic_channels_inference,
                                       bn_basic_width_training,
                                       bn_basic_width_inference));
