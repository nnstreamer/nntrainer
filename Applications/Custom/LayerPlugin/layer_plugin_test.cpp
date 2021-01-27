// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   test.cpp
 * @date   26 January 2021
 * @brief  This file contains the execution part of LayerPlugin example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <gtest/gtest.h>

#include <dlfcn.h>
#include <iostream>
#include <string>

#include <app_context.h>
#include <layer.h>

const char *RES_PATH = std::getenv("RES_PATH");

TEST(AppContext, DlRegisterOpen_p) {
  ASSERT_NE(RES_PATH, nullptr) << "RES_PATH environment value must be set";
  auto &ac = nntrainer::AppContext::Global();

  ac.registerLayerPlugin("libpow_layer.so", RES_PATH);

  auto layer = ml::train::createLayer("pow");

  EXPECT_EQ(layer->getType(), "pow");
}

TEST(AppContext, DlRegisterWrongPath_n) {
  ASSERT_NE(RES_PATH, nullptr) << "RES_PATH environment value must be set";
  auto &ac = nntrainer::AppContext::Global();

  EXPECT_THROW(ac.registerLayerPlugin("wrong_name.so"), std::invalid_argument);
}
