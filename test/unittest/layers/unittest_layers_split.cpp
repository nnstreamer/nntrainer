// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_split.cpp
 * @date 12 June 2021
 * @brief Split Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <split_layer.h>

auto semantic_split =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::SplitLayer>,
                          nntrainer::SplitLayer::type, {"axis=3"}, 0, false, 1);

GTEST_PARAMETER_TEST(Split, LayerSemantics, ::testing::Values(semantic_split));
