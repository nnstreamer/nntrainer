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
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <string>

#include <app_context.h>
#include <layer.h>
#include <layer_internal.h>

static const char *NNTRAINER_PATH = std::getenv("NNTRAINER_PATH");

TEST(MaeLossLayer, DlRegisterOpen_p) {
  ASSERT_NE(NNTRAINER_PATH, nullptr)
    << "NNTRAINER_PATH environment value must be set";
  auto ac = nntrainer::AppContext();

  ac.registerLayer("libmae_loss_layer.so", NNTRAINER_PATH);

  auto layer = ac.createObject<nntrainer::LayerV1>("mae_loss");

  EXPECT_EQ(layer->getType(), "mae_loss");
}
