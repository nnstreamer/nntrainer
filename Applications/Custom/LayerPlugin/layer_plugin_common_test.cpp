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
#include <layer_node.h>

TEST_P(LayerPluginCommonTest, DlRegisterOpen_p) {
  ac.registerLayer(plugin_lib_name, NNTRAINER_PATH);
  auto layer = ac.createObject<nntrainer::Layer>(layer_type_name);

  EXPECT_EQ(layer->getType(), layer_type_name);
}

TEST_P(LayerPluginCommonTest, DlRegisterWrongPath_n) {
  EXPECT_THROW(ac.registerLayer("wrong_name.so"), std::invalid_argument);
}

TEST_P(LayerPluginCommonTest, DlRegisterDirectory_p) {
  ac.registerPluggableFromDirectory(NNTRAINER_PATH);
  auto layer = ac.createObject<nntrainer::Layer>(layer_type_name);
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

  auto lnode = std::static_pointer_cast<nntrainer::LayerNode>(l);

  EXPECT_THROW(lnode->setProperty({"invalid_values"}), std::invalid_argument);
  EXPECT_THROW(lnode->getOutputDimensions(), std::runtime_error);
  EXPECT_THROW(lnode->getInputDimensions(), std::runtime_error);
}

TEST_P(LayerPluginCommonTest, DefaultEnvironmentPathLayerNotExist_n) {
  EXPECT_THROW(ml::train::createLayer("key_does_not_exist"),
               std::invalid_argument);
}
