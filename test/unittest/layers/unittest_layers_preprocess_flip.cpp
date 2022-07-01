// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_preprocess_flip.cpp
 * @date 11 June Flip
 * @brief Preprocess flip Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <preprocess_flip_layer.h>

auto semantic_flip = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessFlipLayer>,
  nntrainer::PreprocessFlipLayer::type, {}, 0, false, 1);

GTEST_PARAMETER_TEST(PreprocessFlip, LayerSemantics,
                     ::testing::Values(semantic_flip));
