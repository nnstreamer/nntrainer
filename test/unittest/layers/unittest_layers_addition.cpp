// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_addition.cpp
 * @date 7 July 2021
 * @brief Addition Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <addition_layer.h>
#include <layers_common_tests.h>

auto semantic_addition =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::AdditionLayer>,
                          nntrainer::AdditionLayer::type, {}, 0, false);

INSTANTIATE_TEST_CASE_P(Addition, LayerSemantics,
                        ::testing::Values(semantic_addition));
