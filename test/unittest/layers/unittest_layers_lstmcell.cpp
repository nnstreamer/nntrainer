// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_lstmcell.cpp
 * @date 22 October 2021
 * @brief LSTMCell Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <lstmcell.h>

auto semantic_lstmcell = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::LSTMCellLayer>,
  nntrainer::LSTMCellLayer::type, {"unit=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 3);

GTEST_PARAMETER_TEST(LSTMCell, LayerSemantics,
                     ::testing::Values(semantic_lstmcell));

auto lstmcell_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMCellLayer>,
  {"unit=5", "integrate_bias=true"}, "3:1:1:7,3:1:1:5,3:1:1:5",
  "lstmcell_single_step.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

GTEST_PARAMETER_TEST(LSTMCell, LayerGoldenTest,
                     ::testing::Values(lstmcell_single_step));
