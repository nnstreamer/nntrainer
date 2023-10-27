// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_embedding.cpp
 * @date 12 June 2021
 * @brief Embedding Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <embedding.h>
#include <layers_common_tests.h>

auto skip_derivative_option = LayerGoldenTestParamOptions::SKIP_CALC_DERIV;

auto semantic_embedding = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::EmbeddingLayer>,
  nntrainer::EmbeddingLayer::type, {"out_dim=1", "in_dim=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Embedding, LayerSemantics,
                     ::testing::Values(semantic_embedding));

auto embedding_fp32 =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::EmbeddingLayer>,
                           {"out_dim=10", "in_dim=10"}, "1:1:1:10",
                           "embedding_w32a32_single_batch.nnlayergolden",
                           skip_derivative_option, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(Embedding32, LayerGoldenTest,
                     ::testing::Values(embedding_fp32));

#ifdef ENABLE_FP16
auto embedding_mixed_single_batch =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::EmbeddingLayer>,
                           {"out_dim=10", "in_dim=10"}, "1:1:1:10",
                           "embedding_mixed_single_batch.nnlayergolden",
                           skip_derivative_option, "nchw", "fp16", "fp16");

auto embedding_mixed_double_batch =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::EmbeddingLayer>,
                           {"out_dim=10", "in_dim=20"}, "2:1:1:10",
                           "embedding_mixed_double_batch.nnlayergolden",
                           skip_derivative_option, "nchw", "fp16", "fp16");

auto embedding_mixed_many =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::EmbeddingLayer>,
                           {"out_dim=64", "in_dim=1000"}, "5:1:1:10",
                           "embedding_mixed_many.nnlayergolden",
                           skip_derivative_option, "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(Embedding16, LayerGoldenTest,
                     ::testing::Values(embedding_mixed_single_batch,
                                       embedding_mixed_double_batch,
                                       embedding_mixed_many));
#endif