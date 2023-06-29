// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file unittest_layers_mol_attention.cpp
 * @date 13 July 2022
 * @brief Multi Head Attention Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <multi_head_attention_layer.h>

auto semantic_multi_head_attention = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>,
  nntrainer::MultiHeadAttentionLayer::type,
  {"num_heads=1", "projected_key_dim=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 3);

auto semantic_multi_head_attention_with_mask = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>,
  nntrainer::MultiHeadAttentionLayer::type,
  {"num_heads=1", "projected_key_dim=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 4);

GTEST_PARAMETER_TEST(
  MultiHeadAttention, LayerSemantics,
  ::testing::Values(semantic_multi_head_attention,
                    semantic_multi_head_attention_with_mask));

auto multi_head_attention_single_batch = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>,
  {"num_heads=2", "projected_key_dim=3"}, "1:1:5:7,1:1:3:7,1:1:3:7",
  "multi_head_attention_single_batch.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto multi_head_attention = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>,
  {"num_heads=2", "projected_key_dim=3"}, "2:1:5:7,2:1:3:7,2:1:3:7",
  "multi_head_attention.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto multi_head_attention_return_attention_scores = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>,
  {"num_heads=2", "projected_key_dim=3", "return_attention_weight=before",
   "average_attention_weight=false"},
  "2:1:5:7,2:1:3:7,2:1:3:7",
  "multi_head_attention_return_attention_scores.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto multi_head_attention_value_dim = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>,
  {"num_heads=2", "projected_key_dim=3", "projected_value_dim=5"},
  "2:1:5:7,2:1:3:7,2:1:3:7", "multi_head_attention_value_dim.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto multi_head_attention_output_shape = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::MultiHeadAttentionLayer>,
  {"num_heads=2", "projected_key_dim=3", "output_shape=5"},
  "2:1:5:7,2:1:3:7,2:1:3:7", "multi_head_attention_output_shape.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

GTEST_PARAMETER_TEST(
  MultiHeadAttention, LayerGoldenTest,
  ::testing::Values(multi_head_attention_single_batch, multi_head_attention,
                    multi_head_attention_return_attention_scores,
                    multi_head_attention_value_dim,
                    multi_head_attention_output_shape));
