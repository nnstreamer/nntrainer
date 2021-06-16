// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_layer_fully_connected.cpp
 * @date 15 June 2021
 * @brief Fully Connected Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>

INSTANTIATE_TEST_CASE_P(
  FullyConnected, LayerCreateDestroyTest,
  ::testing::Values(
    "golden1") /**< format of type, properties, num_batch, golden file name */);

INSTANTIATE_TEST_CASE_P(
  FullyConnected, LayerGoldenTest,
  ::testing::Values(
    "golden1") /**< format of type, properties, num_batch, golden file name */);
