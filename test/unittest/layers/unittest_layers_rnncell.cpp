// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file unittest_layers_rnncell.cpp
 * @date 1 November 2021
 * @brief RNNCell Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <rnncell.h>

auto semantic_rnncell = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::RNNCellLayer>,
  nntrainer::RNNCellLayer::type, {"unit=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(RNNCell, LayerSemantics,
                     ::testing::Values(semantic_rnncell));

auto rnncell_single_step =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::RNNCellLayer>,
                           {"unit=5", "integrate_bias=true"}, "3:1:1:7,3:1:1:5",
                           "rnncell_single_step.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

GTEST_PARAMETER_TEST(RNNCell, LayerGoldenTest,
                     ::testing::Values(rnncell_single_step));
