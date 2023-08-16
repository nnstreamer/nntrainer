// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file unittest_layers_zoneout_lstmcell.cpp
 * @date 14 June 2022
 * @brief ZoneoutLSTMCell Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <lstmcell.h>

auto semantic_zoneout_lstmcell = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::LSTMCellLayer>,
  nntrainer::LSTMCellLayer::type,
  {"unit=1", "hidden_state_zoneout_rate=0.1", "cell_state_zoneout_rate=0.0"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 3);

INSTANTIATE_TEST_CASE_P(LSTMCell, LayerSemantics,
                        ::testing::Values(semantic_zoneout_lstmcell));

auto zoneout_lstmcell_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMCellLayer>,
  {"unit=5", "integrate_bias=true", "hidden_state_zoneout_rate=0.1",
   "cell_state_zoneout_rate=0.0"},
  "3:1:1:7,3:1:1:5,3:1:1:5", "zoneout_lstmcell_single_step.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

INSTANTIATE_TEST_CASE_P(LSTMCell, LayerGoldenTest,
                        ::testing::Values(zoneout_lstmcell_single_step));
