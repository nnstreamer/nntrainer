// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_plugin_common_test.h
 * @date   11 June 2021
 * @brief  This file contains the parameterized common test of layer plugin
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __LAYER_PLUGIN_COMMON_TEST_H__
#define __LAYER_PLUGIN_COMMON_TEST_H__
#include <string>
#include <tuple>

#include <gtest/gtest.h>

/// @todo check if below headers are part of devel header
#include <app_context.h>
#include <layer.h>

static const char *NNTRAINER_PATH = std::getenv("NNTRAINER_PATH");

/**
 * @brief layer plugin common test cases
 * @todo expand this to typed_test_p for other types of example (eg) optimizer
 *
 */
class LayerPluginCommonTest
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
    layer_type_name = std::get<1>(params);
    ac = nntrainer::AppContext();
  };

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown(){};

protected:
  nntrainer::AppContext ac;    /**< AppContext          */
  std::string plugin_lib_name; /**< plugin library name */
  std::string layer_type_name; /**< layer type name */
};
#endif // __LAYER_PLUGIN_COMMON_TEST_H__
