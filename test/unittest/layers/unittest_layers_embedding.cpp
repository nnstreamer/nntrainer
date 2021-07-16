// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_embedding.cpp
 * @date 12 June 2021
 * @brief Embedding Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <embedding.h>
#include <layers_common_tests.h>

auto semantic_embedding = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::EmbeddingLayer>,
  nntrainer::EmbeddingLayer::type, {"in_length=1", "out_dim=1", "in_dim=1"}, {},
  0, false);

INSTANTIATE_TEST_CASE_P(Embedding, LayerSemantics,
                        ::testing::Values(semantic_embedding));
