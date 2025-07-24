// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_nnstreamer.cpp
 * @date 12 June 2021
 * @brief NNStreamer Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <nnstreamer_layer.h>

// auto semantic_nnstreamer = LayerSemanticsParamType(
//   nntrainer::createLayer<nntrainer::NNStreamerLayer>,
//   nntrainer::NNStreamerLayer::type,
//   {"model_path=../test/test_models/models/add.tflite"},
//   LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

// GTEST_PARAMETER_TEST(NNStreamer, LayerSemantics,
//                      ::testing::Values(semantic_nnstreamer));
