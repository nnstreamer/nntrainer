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
  nntrainer::LSTMCellLayer::type, {"unit=1", "timestep=0", "max_timestep=1"}, 0,
  false, 1);

INSTANTIATE_TEST_CASE_P(LSTMCell, LayerSemantics,
                        ::testing::Values(semantic_lstmcell));

auto lstmcell_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LSTMCellLayer>,
  {"unit=5", "timestep=0", "max_timestep=1"}, "3:1:1:7",
  "lstm_single_step.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

INSTANTIATE_TEST_CASE_P(LSTMCell, LayerGoldenTest,
                        ::testing::Values(lstmcell_single_step));
