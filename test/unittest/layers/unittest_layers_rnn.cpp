// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_rnn.cpp
 * @date 8 July 2021
 * @brief RNN Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <rnn.h>

auto semantic_rnn = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::RNNLayer>, nntrainer::RNNLayer::type,
  {"unit=1"}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false,
  1);

GTEST_PARAMETER_TEST(RNN, LayerSemantics, ::testing::Values(semantic_rnn));

auto rnn_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::RNNLayer>,
  {"unit=5", "return_sequences=false", "integrate_bias=true"}, "3:1:1:7",
  "rnn_single_step.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

GTEST_PARAMETER_TEST(RNN, LayerGoldenTest, ::testing::Values(rnn_single_step));
