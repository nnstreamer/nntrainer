// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_loss.cpp
 * @date 15 June 2021
 * @brief Batch Normalization Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <bn_layer.h>
#include <layers_common_tests.h>

auto semantic_bn = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::BatchNormalizationLayer>,
  nntrainer::BatchNormalizationLayer::type, {}, {}, 0, false);

INSTANTIATE_TEST_CASE_P(BatchNormalization, LayerSemantics,
                        ::testing::Values(semantic_bn));
