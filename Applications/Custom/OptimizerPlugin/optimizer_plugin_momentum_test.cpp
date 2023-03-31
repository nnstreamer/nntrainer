// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   optimizer_plugin_momentun_test.cpp
 * @date   30 March 2023
 * @brief  This file contains the execution part of momentum optimizer in
 * LayerPlugin example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <tuple>

#include <gtest/gtest.h>

#include <momentum.h>
#include <optimizer_plugin_common_test.h>

#ifdef GTEST_BACKPORT
#define GTEST_PARAMETER_TEST INSTANTIATE_TEST_CASE_P
#else
#define GTEST_PARAMETER_TEST INSTANTIATE_TEST_SUITE_P
#endif

GTEST_PARAMETER_TEST(
  Momentum, OptimizerPluginCommonTest,
  ::testing::Values(std::make_tuple("libmomentum_optimizer.so", "momentum")));
auto semantic_momentum =
  OptimizerSemanticsParamType(nntrainer::createOptimizer<custom::Momentum>,
                              custom::Momentum::type, {}, 0, false, 1);

auto semantic_momentum_m = OptimizerSemanticsParamType(
  nntrainer::createOptimizer<custom::Momentum>, custom::Momentum::type,
  {"momentum=0.03"}, 0, false, 1);

GTEST_PARAMETER_TEST(Momentum, OptimizerSemantics,
                     ::testing::Values(semantic_momentum, semantic_momentum_m));
