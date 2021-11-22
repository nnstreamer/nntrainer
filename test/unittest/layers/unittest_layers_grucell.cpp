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
  nntrainer::GRUCellLayer::type, {"unit=1", "max_timestep=1", "timestep=0"}, 0,
  false, 1);

INSTANTIATE_TEST_CASE_P(GRUCell, LayerSemantics,
                        ::testing::Values(semantic_grucell));

auto grucell_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRUCellLayer>,
  {"unit=5", "max_timestep=1", "timestep=0"}, "3:1:1:7",
  "gru_single_step.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

INSTANTIATE_TEST_CASE_P(GRUCell, LayerGoldenTest,
                        ::testing::Values(grucell_single_step));
