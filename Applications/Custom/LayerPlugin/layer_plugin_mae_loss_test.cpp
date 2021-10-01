// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_plugin_mae_loss_test.cpp
 * @date   10 June 2021
 * @brief  This file contains the execution part of mae loss layer in
 * LayerPlugin example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layer_plugin_common_test.h>
#include <layers_common_tests.h>
#include <mae_loss.h>

INSTANTIATE_TEST_CASE_P(
  MaeLossLayer, LayerPluginCommonTest,
  ::testing::Values(std::make_tuple("libmae_loss_layer.so", "mae_loss")));

auto semantic_mae =
  LayerSemanticsParamType(nntrainer::createLayer<custom::MaeLossLayer>,
                          custom::MaeLossLayer::type, {}, 0, false, 1);

INSTANTIATE_TEST_CASE_P(MaeLossLayer, LayerSemantics,
                        ::testing::Values(semantic_mae));
