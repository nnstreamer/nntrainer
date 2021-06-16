// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_layer_impl.cpp
 * @date 16 June 2021
 * @brief Layer Impl test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>

#include <layer_impl.h>

namespace {
/**
 * @brief Minimal implementation of layer impl to test layer impl itself
 *
 */
class MockLayer : public nntrainer::LayerImpl {};
} // namespace

INSTANTIATE_TEST_CASE_P(
  LayerImpl, LayerCreateDestroyTest,
  ::testing::Values(
    "test") /**< format of type, properties, num_batch, golden file name */);

INSTANTIATE_TEST_CASE_P(
  LayerImpl, LayerGoldenTest,
  ::testing::Values(
    "test") /**< format of type, properties, num_batch, golden file name */);
