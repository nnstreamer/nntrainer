// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file unittest_layers_grucell.cpp
 * @date 09 November 2021
 * @brief GRUCell Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <grucell.h>
#include <layers_common_tests.h>

auto semantic_grucell = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::GRUCellLayer>,
  nntrainer::GRUCellLayer::type,
  {"unit=1", "integrate_bias=false", "reset_after=true"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(GRUCell, LayerSemantics,
                     ::testing::Values(semantic_grucell));

auto grucell_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRUCellLayer>,
  {"unit=5", "integrate_bias=true", "reset_after=false"}, "3:1:1:7,3:1:1:5",
  "grucell_single_step.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto grucell_reset_after_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRUCellLayer>,
  {"unit=5", "integrate_bias=false", "reset_after=true"}, "3:1:1:7,3:1:1:5",
  "grucell_reset_after_single_step.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto grucell_single_step_act = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRUCellLayer>,
  {"unit=5", "integrate_bias=true", "reset_after=false",
   "hidden_state_activation=sigmoid", "recurrent_activation=tanh"},
  "3:1:1:7,3:1:1:5", "grucell_single_step_act.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

GTEST_PARAMETER_TEST(GRUCell, LayerGoldenTest,
                     ::testing::Values(grucell_single_step,
                                       grucell_reset_after_single_step,
                                       grucell_single_step_act));
