// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_concat.cpp
 * @date 7 July 2021
 * @brief Concat Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <concat_layer.h>
#include <layers_common_tests.h>

auto semantic_concat =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::ConcatLayer>,
                          nntrainer::ConcatLayer::type, {}, 0, false);

INSTANTIATE_TEST_CASE_P(Concat, LayerSemantics,
                        ::testing::Values(semantic_concat));
