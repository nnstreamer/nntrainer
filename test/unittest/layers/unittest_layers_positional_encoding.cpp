// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file unittest_layers_positional_encoding.cpp
 * @date 24 August 2022
 * @brief PositionalEncodingLayer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <positional_encoding_layer.h>

auto semantic_positional_encoding = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PositionalEncodingLayer>,
  nntrainer::PositionalEncodingLayer::type, {"max_timestep=10"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

INSTANTIATE_TEST_CASE_P(PositionalEncoding, LayerSemantics,
                        ::testing::Values(semantic_positional_encoding));

auto positional_encoding_partial = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::PositionalEncodingLayer>,
  {"max_timestep=10"}, "3:1:7:6", "positional_encoding_partial.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto positional_encoding = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::PositionalEncodingLayer>,
  {"max_timestep=10"}, "3:1:10:6", "positional_encoding.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

INSTANTIATE_TEST_CASE_P(PositionalEncoding, LayerGoldenTest,
                        ::testing::Values(positional_encoding_partial,
                                          positional_encoding));
