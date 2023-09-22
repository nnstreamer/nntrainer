// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   optimizer_plugin_common_test.h
 * @date   31 March 2023
 * @brief  This file contains the parameterized common test of optimizer plugin
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __OPTIMIZER_PLUGIN_COMMON_TEST_H__
#define __OPTIMIZER_PLUGIN_COMMON_TEST_H__
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include <app_context.h>
#include <optimizer.h>

static const char *NNTRAINER_PATH = std::getenv("NNTRAINER_PATH");

using OptimizerFactoryType =
  std::function<std::unique_ptr<nntrainer::Optimizer>(
    const std::vector<std::string> &)>;

using OptimizerSemanticsParamType =
  std::tuple<OptimizerFactoryType /** optimizer factory */,
             std::string /** Type of Optimizer */,
             std::vector<std::string> /** Necessary Properties */,
             unsigned int /** Options */, bool /** fail or succeed */,
             unsigned int /** number of inputs */
             >;

/**
 * @brief Optimizer Semantics class
 *
 */
class OptimizerSemantics
  : public ::testing::TestWithParam<OptimizerSemanticsParamType> {
public:
  /**
   * @brief Destroy the Optimizer Semantics object
   *
   */
  virtual ~OptimizerSemantics() {}

  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp() {
    auto f = std::get<0>(GetParam());
    opt = std::move(f({}));
    std::tie(std::ignore, expected_type, valid_properties, options, must_fail,
             num_inputs) = GetParam();

    num_inputs = std::max(1u, num_inputs);
  }

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown() {}

protected:
  std::unique_ptr<nntrainer::Optimizer> opt;
  std::string expected_type;
  std::vector<std::string> valid_properties;
  unsigned int options;
  bool must_fail;
  unsigned int num_inputs;
};

/**
 * @brief optimizer plugin common test cases
 *
 */
class OptimizerPluginCommonTest
  : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {

public:
  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp() {
    ASSERT_NE(NNTRAINER_PATH, nullptr)
      << "NNTRAINER_PATH environment value must be set";

    const auto &params = GetParam();
    plugin_lib_name = std::get<0>(params);
    optimizer_type_name = std::get<1>(params);
    ac = nntrainer::AppContext();
  };

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown(){};

protected:
  nntrainer::AppContext ac;        /**< AppContext          */
  std::string plugin_lib_name;     /**< plugin library name */
  std::string optimizer_type_name; /**< optimizer type name */
};
#endif // __OPTIMIZER_PLUGIN_COMMON_TEST_H__
