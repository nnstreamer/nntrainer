// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_pooling.cpp
 * @date 13 July 2021
 * @brief Simpleshot Layers Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <centering.h>
#include <centroid_knn.h>
#include <layers_common_tests.h>
#include <preprocess_l2norm_layer.h>

/// @todo move below test to the main repo
auto semantic_activation_l2norm = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PreprocessL2NormLayer>,
  nntrainer::PreprocessL2NormLayer::type, {}, 0, false, 1);

auto semantic_activation_centroid_knn = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::CentroidKNN>, nntrainer::CentroidKNN::type,
  {"num_class=1"}, 0, false, 1);

auto semantic_activation_centering = LayerSemanticsParamType(
  nntrainer::createLayer<simpleshot::layers::CenteringLayer>,
  simpleshot::layers::CenteringLayer::type,
  {"feature_path=../Applications/SimpleShot/backbones/"
   "conv4_60classes_feature_vector.bin"},
  0, false, 1);

GTEST_PARAMETER_TEST(L2NormLayer, LayerSemantics,
                     ::testing::Values(semantic_activation_l2norm));

GTEST_PARAMETER_TEST(CentroidKNN, LayerSemantics,
                     ::testing::Values(semantic_activation_centroid_knn));

GTEST_PARAMETER_TEST(CenteringLayer, LayerSemantics,
                     ::testing::Values(semantic_activation_centering));
