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
#include <nntrainer_test_util.h>

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
             unsigned int /** Options */, bool /** fail or succeed */,
             unsigned int /** number of inputs */
             >;

/**
 * @brief LayerSemantics
 * @note  This test suite includes
 * @see   layers_[standalone|dependent]_common_test.cpp for details
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
  std::unique_ptr<nntrainer::Layer> layer;
  std::string expected_type;
  std::vector<std::string> valid_properties;
  unsigned int options;
  bool must_fail;
  unsigned int num_inputs;
};

/**
 * @brief LayerPropertySemantics
 * @details Inherit LayerSemantics to solely test negative property cases
 */
class LayerPropertySemantics : public LayerSemantics {};

typedef enum {
  SKIP_CALC_GRAD = 1 << 0,  /**< skip calculating gradient and compare */
  SKIP_CALC_DERIV = 1 << 1, /**< skip calculating derivative and compare */

  FORWARD_MODE_INFERENCE =
    1 << 2, /**< set if layer should be forwarded with inference mode */

  DROPOUT_MATCH_60_PERCENT = 1 << 3, /**< set if only 60 percentage output
                               match is sufficient for dropout */
  DEFAULT =
    0, /**< default set up, compare forward, backward in training mode */
} LayerGoldenTestParamOptions;

using LayerGoldenTestParamType =
  std::tuple<LayerFactoryType /**< Layer creator */,
             std::vector<std::string> /**< Properties */,
             const char *, /**< Input Tensor dimensions representation */
             const char * /**< Golden file name */,
             int /**< LayerGoldenTestParamOptions */,
             std::string /** < TensorFormat */,
             std::string /** < TensorTeyp */>;

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

  /**
   * @brief check if given test suite should be forwarded with inference mode
   * enabled
   *
   * @return bool layer should be forwarded with inference
   */
  bool shouldForwardWithInferenceMode();

  /**
   * @brief check if given test suite must compare results using with a percent
   * match for the tensors enabled
   *
   * @return bool layer should be match approximately
   */
  bool shouldMatchDropout60Percent();

  /**
   * @brief check if given test suite should skip calculating derivative
   *
   * @return bool true if should skip calculating derivative
   */
  bool shouldSkipCalcDeriv();

  /**
   * @brief check if given test suite should skip calculating Gradient
   *
   * @return bool true if should skip calculating Gradient
   */
  bool shouldSkipCalcGrad();
};

#endif // __LAYERS_COMMON_TESTS_H__
