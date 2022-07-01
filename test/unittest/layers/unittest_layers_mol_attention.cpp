// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_mol_attention.cpp
 * @date 26 November 2021
 * @brief MoL Attention Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <mol_attention_layer.h>

auto semantic_mol_attention = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MoLAttentionLayer>,
  nntrainer::MoLAttentionLayer::type, {"unit=5", "mol_k=1"}, 0, false, 3);

GTEST_PARAMETER_TEST(MoLAttention, LayerSemantics,
                     ::testing::Values(semantic_mol_attention));
