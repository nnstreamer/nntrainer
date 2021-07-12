// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_tflite.cpp
 * @date 12 June 2021
 * @brief TfLite Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <tflite_layer.h>

auto semantic_tflite = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::TfLiteLayer>, nntrainer::TfLiteLayer::type,
  {"modelfile=../test/test_models/models/add.tflite"}, 0, false);

INSTANTIATE_TEST_CASE_P(TfLite, LayerSemantics,
                        ::testing::Values(semantic_tflite));
