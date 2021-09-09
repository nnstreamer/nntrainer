// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file layer_common_tests.h
 * @date 15 June 2021
 * @brief Common test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __LAYERS_COMMON_TESTS_H__
#define __LAYERS_COMMON_TESTS_H__

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <layer_devel.h>

typedef enum {
  AVAILABLE_FROM_APP_CONTEXT =
    1 << 0, /**< set if layer is available from app context */
} LayerCreateSetPropertyOptions;

using LayerFactoryType = std::function<std::unique_ptr<nntrainer::Layer>(
  const std::vector<std::string> &)>;

using LayerSemanticsParamType =
  std::tuple<LayerFactoryType /** layer factory */,
             std::string /** Type of Layer */,
             std::vector<std::string> /** Necessary Properties */,
             unsigned int /** Options */, bool /** fail or succeed */
             >;

/**
 * @brief LayerSemantics
 * @note  This test suite includes
 * @see   layers_common_test.cpp for details
 * 1. Layer Creation
 * 2. SetProperties
 * 3. Semantics of Layer (eg) finalize twice is prohibited)
 */
class LayerSemantics
  : public ::testing::TestWithParam<LayerSemanticsParamType> {
public:
  /**
   * @brief Destroy the Layer Semantics object
   *
   */
  virtual ~LayerSemantics() {}

  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp() {
    auto f = std::get<0>(GetParam());
    layer = std::move(f({}));
    std::tie(std::ignore, expected_type, valid_properties, options, must_fail) =
      GetParam();
  }

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown() {}

protected:
  std::unique_ptr<nntrainer::Layer> layer;
  std::string expected_type;
  std::vector<std::string> valid_properties;
  unsigned int options;
  bool must_fail;
};

using LayerGoldenTestParamType =
  std::tuple<LayerFactoryType /**< Layer creator */,
             std::vector<std::string> /**< Properties */,
             const char *, /**< Input Tensor dimensions representation */
             const char * /**< Golden file name */>;

/**
 * @brief Golden Layer Test with designated format
 */
class LayerGoldenTest
  : public ::testing::TestWithParam<LayerGoldenTestParamType> {
public:
  /**
   * @brief Destroy the Layer Semantics object
   *
   */
  virtual ~LayerGoldenTest();

  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp();

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown();
};

#endif // __LAYERS_COMMON_TESTS_H__
