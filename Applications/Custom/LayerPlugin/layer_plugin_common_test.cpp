// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_plugin_common_test.cpp
 * @date   11 June 2021
 * @brief  This file contains the parameterized common test of layer plugin
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <layer_plugin_common_test.h>

#include <fstream>

TEST_P(LayerPluginCommonTest, DlRegisterOpen_p) {
  ac.registerLayer(plugin_lib_name, NNTRAINER_PATH);
  auto layer = ac.createObject<nntrainer::LayerV1>(layer_type_name);

  EXPECT_EQ(layer->getType(), layer_type_name);
}

TEST_P(LayerPluginCommonTest, DlRegisterWrongPath_n) {
  EXPECT_THROW(ac.registerLayer("wrong_name.so"), std::invalid_argument);
}

TEST_P(LayerPluginCommonTest, DlRegisterDirectory_p) {
  ac.registerPluggableFromDirectory(NNTRAINER_PATH);
  auto layer = ac.createObject<nntrainer::LayerV1>(layer_type_name);
  EXPECT_EQ(layer->getType(), layer_type_name);
}

TEST_P(LayerPluginCommonTest, DlRegisterDirectory_n) {
  EXPECT_THROW(ac.registerPluggableFromDirectory("wrong path"),
               std::invalid_argument);
}

TEST_P(LayerPluginCommonTest, DefaultEnvironmentPath_p) {
  /// as NNTRAINER_PATH is fed to the test, this should success without an
  /// error
  std::shared_ptr<ml::train::Layer> l = ml::train::createLayer(layer_type_name);
  EXPECT_EQ(l->getType(), layer_type_name);

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

  EXPECT_NE(layer->setProperty({"invalid_values"}), ML_ERROR_NONE);
  EXPECT_EQ(layer->checkValidation(), ML_ERROR_NONE);
  EXPECT_EQ(layer->getOutputDimension()[0], nntrainer::TensorDim());
  EXPECT_EQ(layer->getInputDimension()[0], nntrainer::TensorDim());
}

TEST_P(LayerPluginCommonTest, DefaultEnvironmentPathLayerNotExist_n) {
  EXPECT_THROW(ml::train::createLayer("key_does_not_exist"),
               std::invalid_argument);
}
