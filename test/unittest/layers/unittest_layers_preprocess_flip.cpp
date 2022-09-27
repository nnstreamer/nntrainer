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

auto flip_prop1 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessFlipLayer>,
  nntrainer::PreprocessFlipLayer::type,
  {"flip_direction=vertical_and_horizontal"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto flip_prop2 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessFlipLayer>,
  nntrainer::PreprocessFlipLayer::type, {"flip_direction=flip"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto flip_prop3 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessFlipLayer>,
  nntrainer::PreprocessFlipLayer::type, {"flip_direction=horizontal&vertical"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto flip_prop4 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessFlipLayer>,
  nntrainer::PreprocessFlipLayer::type, {"flip_direction=horizontal&&vertical"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto flip_prop5 = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessFlipLayer>,
  nntrainer::PreprocessFlipLayer::type, {"flip_direction=horizontal+vertical"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(PreprocessFlip, LayerPropertySemantics,
                     ::testing::Values(flip_prop1, flip_prop2, flip_prop3,
                                       flip_prop4, flip_prop5));
