// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_permute.cpp
 * @date 7 July 2021
 * @brief Permute Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <permute_layer.h>

auto semantic_permute = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PermuteLayer>,
  nntrainer::PermuteLayer::type, {"direction=3,2,1"}, 0, false, 1);

INSTANTIATE_TEST_CASE_P(Permute, LayerSemantics,
                        ::testing::Values(semantic_permute));
