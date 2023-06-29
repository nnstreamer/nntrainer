// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_concat.cpp
 * @date 7 July 2021
 * @brief Concat Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <concat_layer.h>
#include <layers_common_tests.h>

auto semantic_concat = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ConcatLayer>, nntrainer::ConcatLayer::type,
  {}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Concat, LayerSemantics,
                     ::testing::Values(semantic_concat));

auto concat_dim3 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayer>, {"axis=3"},
  "2:3:3:2, 2:3:3:3", "concat_dim3.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto concat_dim2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayer>, {"axis=2"},
  "2:3:2:3, 2:3:3:3", "concat_dim2.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto concat_dim1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConcatLayer>, {"axis=1"},
  "2:2:3:3, 2:3:3:3", "concat_dim1.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

GTEST_PARAMETER_TEST(Concat, LayerGoldenTest,
                     ::testing::Values(concat_dim3, concat_dim2, concat_dim1));
