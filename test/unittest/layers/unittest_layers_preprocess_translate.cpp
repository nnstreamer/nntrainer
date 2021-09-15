// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_preprocess_translate.cpp
 * @date 11 June 2021
 * @brief Preprocess Translate Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <preprocess_translate_layer.h>

auto semantic_translate = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessTranslateLayer>,
  nntrainer::PreprocessTranslateLayer::type, {"random_translate=0.1"}, 0,
  false);

INSTANTIATE_TEST_CASE_P(PreprocessTranslate, LayerSemantics,
                        ::testing::Values(semantic_translate));
