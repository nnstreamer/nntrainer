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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace nntrainer {
class Layer;
}

typedef enum {
  AVAILABLE_FROM_APP_CONTEXT =
    1 << 0, /**< set if layer is available from app context */
} LayerCreateSetPropertyOptions;

using LayerFactoryType = std::function<std::unique_ptr<nntrainer::Layer>(
  const std::vector<std::string> &)>;

using LayerSemanticsParamType =
  std::tuple<LayerFactoryType /** layer factory */,
             std::string /** Type of Layer */,
             std::vector<std::string> /** Valid Properties */,
             std::vector<std::string> /** Invalid Properties */,
             unsigned int /** Options */
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
  virtual ~LayerSemantics();

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

protected:
  std::unique_ptr<nntrainer::Layer> layer;
  std::string expected_type;
  std::vector<std::string> valid_properties;
  std::vector<std::string> invalid_properties;
  unsigned int options;
};

/**
 * @brief Golden Layer Test with designated format
 * @note NYI
 *
 * 1. type
 * 2. properties for the layer
 * 3. batch size
 * 4. golden file name
 */
class LayerGoldenTest : public ::testing::TestWithParam<const char *> {
public:
  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp(){};

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown(){};
};

#endif // __LAYERS_COMMON_TESTS_H__
