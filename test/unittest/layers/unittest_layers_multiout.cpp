// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_multiout.cpp
 * @date 7 July 2021
 * @brief Output Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <multiout_layer.h>

auto semantic_output =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::MultiOutLayer>,
                          nntrainer::MultiOutLayer::type, {}, 0, false);

INSTANTIATE_TEST_CASE_P(Output, LayerSemantics,
                        ::testing::Values(semantic_output));
