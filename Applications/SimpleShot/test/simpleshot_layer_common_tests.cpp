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
#include <l2norm.h>
#include <layers_common_tests.h>

auto semantic_activation_l2norm = LayerSemanticsParamType(
  nntrainer::createLayer<simpleshot::layers::L2NormLayer>,
  simpleshot::layers::L2NormLayer::type, {}, 0, false);

auto semantic_activation_centroid_knn = LayerSemanticsParamType(
  nntrainer::createLayer<simpleshot::layers::CentroidKNN>,
  simpleshot::layers::CentroidKNN::type, {"num_class=1"}, 0, false);

auto semantic_activation_centering = LayerSemanticsParamType(
  nntrainer::createLayer<simpleshot::layers::CenteringLayer>,
  simpleshot::layers::CenteringLayer::type,
  {"feature_path=../Applications/SimpleShot/backbones/"
   "conv4_60classes_feature_vector.bin"},
  0, false);

INSTANTIATE_TEST_CASE_P(L2NormLayer, LayerSemantics,
                        ::testing::Values(semantic_activation_l2norm));

INSTANTIATE_TEST_CASE_P(CentroidKNN, LayerSemantics,
                        ::testing::Values(semantic_activation_centroid_knn));

INSTANTIATE_TEST_CASE_P(CenteringLayer, LayerSemantics,
                        ::testing::Values(semantic_activation_centering));
