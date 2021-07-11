// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_fully_connected.cpp
 * @date 11 June 2021
 * @brief Preprocess Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <preprocess_flip_layer.h>
#include <preprocess_translate_layer.h>

auto semantic_flip = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessFlipLayer>,
  nntrainer::PreprocessFlipLayer::type, {}, 0, false);

auto semantic_translate = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessTranslateLayer>,
  nntrainer::PreprocessTranslateLayer::type, {}, 0, false);

INSTANTIATE_TEST_CASE_P(Preprocess, LayerSemantics,
                        ::testing::Values(semantic_flip, semantic_translate));
