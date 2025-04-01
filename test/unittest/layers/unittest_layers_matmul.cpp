// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_matmul.cpp
 * @date 1 April 2025
 * @brief MatMul Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <matmul_layer.h>

auto semantic_matmul = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MatMulLayer>, nntrainer::MatMulLayer::type,
  {}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

auto semantic_matmul_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::MatMulLayer>, nntrainer::MatMulLayer::type,
  {}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(MatMul, LayerSemantics,
                     ::testing::Values(semantic_matmul, semantic_matmul_multi));
