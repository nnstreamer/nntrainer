// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_fully_connected.cpp
 * @date 15 June 2021
 * @brief Input Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <input_layer.h>
#include <layers_common_tests.h>

auto semantic_input =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::InputLayer>,
                          nntrainer::InputLayer::type, {}, 0, false);

INSTANTIATE_TEST_CASE_P(Input, LayerSemantics,
                        ::testing::Values(semantic_input));
