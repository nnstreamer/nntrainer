// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_plugin_pow_test.cpp
 * @date   26 January 2021
 * @brief  This file contains the execution part of pow layer in LayerPlugin
 * example
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

TEST(PowLayer, DlRegisterOpen_p) {
  ASSERT_NE(NNTRAINER_PATH, nullptr)
    << "NNTRAINER_PATH environment value must be set";
  auto ac = nntrainer::AppContext();

  ac.registerLayer("libpow_layer.so", NNTRAINER_PATH);

  auto layer = ac.createObject<nntrainer::LayerV1>("pow");

  EXPECT_EQ(layer->getType(), "pow");
}

TEST(PowLayer, DlRegisterWrongPath_n) {
  ASSERT_NE(NNTRAINER_PATH, nullptr)
    << "NNTRAINER_PATH environment value must be set";
  auto ac = nntrainer::AppContext();

  EXPECT_THROW(ac.registerLayer("wrong_name.so"), std::invalid_argument);
}

TEST(PowLayer, DlRegisterDirectory_p) {
  ASSERT_NE(NNTRAINER_PATH, nullptr)
    << "NNTRAINER_PATH environment value must be set";
  auto ac = nntrainer::AppContext();

  ac.registerPluggableFromDirectory(NNTRAINER_PATH);

  auto layer = ac.createObject<nntrainer::LayerV1>("pow");

  EXPECT_EQ(layer->getType(), "pow");
}

TEST(PowLayer, DlRegisterDirectory_n) {
  auto ac = nntrainer::AppContext();

  EXPECT_THROW(ac.registerPluggableFromDirectory("wrong path"),
               std::invalid_argument);
}

TEST(PowLayer, DefaultEnvironmentPath_p) {
  /// as NNTRAINER_PATH is fed to the test, this should success without an
  /// error
  std::shared_ptr<ml::train::Layer> l = ml::train::createLayer("pow");
  EXPECT_EQ(l->getType(), "pow");

  auto layer = nntrainer::getLayerDevel(l);

  std::ifstream input_file("does_not_exist");
  EXPECT_NO_THROW(layer->read(input_file));
  if (remove("does_not_exist")) {
    std::cerr << "failed to remove file\n";
  }

  std::ofstream output_file("does_not_exist");
  EXPECT_NO_THROW(layer->save(output_file));
  if (remove("does_not_exist")) {
    std::cerr << "failed to remove file\n";
  }

  EXPECT_NO_THROW(layer->getType());
  EXPECT_NE(layer->setProperty({"invalid_values"}), ML_ERROR_NONE);
  EXPECT_EQ(layer->checkValidation(), ML_ERROR_NONE);
  EXPECT_EQ(layer->getOutputDimension()[0], nntrainer::TensorDim());
  EXPECT_EQ(layer->getInputDimension()[0], nntrainer::TensorDim());
}

TEST(PowLayer, DefaultEnvironmentPath_n) {
  /// pow2 does not exist
  EXPECT_THROW(ml::train::createLayer("pow2"), std::invalid_argument);
}
