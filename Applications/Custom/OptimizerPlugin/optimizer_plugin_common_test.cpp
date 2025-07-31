// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   optimizer_plugin_common_test.cpp
 * @date   30 March 2023
 * @brief  This file contains the parameterized common test of optimizer plugin
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <optimizer_plugin_common_test.h>

#include <fstream>
#include <iostream>
#include <optimizer.h>

TEST_P(OptimizerPluginCommonTest, DlRegisterOpen_p) {
  ac.registerOptimizer(plugin_lib_name, NNTRAINER_PATH);
  auto optimizer = ac.createObject<nntrainer::Optimizer>(optimizer_type_name);

  EXPECT_EQ(optimizer->getType(), optimizer_type_name);
}

TEST_P(OptimizerPluginCommonTest, DlRegisterWrongPath_n) {
  EXPECT_THROW(ac.registerOptimizer("wrong_name.so"), std::invalid_argument);
}

TEST_P(OptimizerPluginCommonTest, DlRegisterDirectory_p) {
  ac.registerPluggableFromDirectory(NNTRAINER_PATH);
  auto optimizer = ac.createObject<nntrainer::Optimizer>(optimizer_type_name);
  EXPECT_EQ(optimizer->getType(), optimizer_type_name);
}

TEST_P(OptimizerPluginCommonTest, DlRegisterDirectory_n) {
  EXPECT_THROW(ac.registerPluggableFromDirectory("wrong path"),
               std::invalid_argument);
}

TEST_P(OptimizerPluginCommonTest, DefaultEnvironmentPath_p) {
  /// as NNTRAINER_PATH is fed to the test, this should success without an
  /// error
  std::shared_ptr<ml::train::Optimizer> opt =
    ml::train::createOptimizer(optimizer_type_name);
  EXPECT_EQ(opt->getType(), optimizer_type_name);
}

TEST_P(OptimizerPluginCommonTest, DefaultEnvironmentPathOptimizerNotExist_n) {
  EXPECT_THROW(ml::train::createOptimizer("key_does_not_exist"),
               std::invalid_argument);
}

TEST_P(OptimizerSemantics, DISABLED_setProperties_p) {
  /// @todo check if setProperties does not collide with layerNode designated
  /// properties
  EXPECT_EQ(1, 1); /**< no assert tc from TCM, this is disabled test */
}

TEST_P(OptimizerSemantics, setProperties_n) {
  /** must not crash */
  EXPECT_THROW(opt->setProperty({"unknown_props=2"}), std::invalid_argument);
}

TEST_P(OptimizerSemantics, DISABLED_setPropertiesValidWithInvalid_n) {
  EXPECT_EQ(1, 1); /**< no assert tc from TCM, this is disabled test */
}

TEST_P(OptimizerSemantics, gettersValidate_p) {
  std::string type;

  EXPECT_NO_THROW(type = opt->getType());
  EXPECT_GT(type.size(), size_t(0));
}

TEST_P(OptimizerPluginCommonTest, save_p) {
  std::string filepath = "optimizer_type.txt";
  std::ofstream writeFile(filepath.data());
  ac.registerOptimizer(plugin_lib_name, NNTRAINER_PATH);
  auto optimizer = ac.createObject<nntrainer::Optimizer>(optimizer_type_name);
  EXPECT_NO_THROW(optimizer->finalize());
  EXPECT_NO_THROW(optimizer->save(writeFile));
}
