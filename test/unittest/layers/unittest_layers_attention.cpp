// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_attention.cpp
 * @date 1 October 2021
 * @brief Attention Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <attention_layer.h>
#include <layers_common_tests.h>

auto semantic_attention =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::AttentionLayer>,
                          nntrainer::AttentionLayer::type, {}, 0, false, 2);

INSTANTIATE_TEST_CASE_P(Attention, LayerSemantics,
                        ::testing::Values(semantic_attention));

auto attention_shared_kv = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {}, "1:1:5:7,1:1:3:7",
  "attention_shared_kv.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto attention_shared_kv_batched = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::AttentionLayer>, {}, "2:1:5:7,2:1:3:7",
  "attention_shared_kv_batched.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

INSTANTIATE_TEST_CASE_P(Attention, LayerGoldenTest,
                        ::testing::Values(attention_shared_kv,
                                          attention_shared_kv_batched));
