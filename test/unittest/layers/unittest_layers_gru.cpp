// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_gru.cpp
 * @date 11 July 2021
 * @brief GRU Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <gru.h>
#include <layers_common_tests.h>

auto semantic_gru =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::GRULayer>,
                          nntrainer::GRULayer::type, {"unit=1"}, 0, false, 1);

INSTANTIATE_TEST_CASE_P(GRU, LayerSemantics, ::testing::Values(semantic_gru));
