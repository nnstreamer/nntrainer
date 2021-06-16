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

/**
 * @brief LayerCreateDestroyTest
 * @note NYI
 *
 * parameters required
 * 1. type
 * 2. a set of valid properties
 * 3. a set of invalid properties
 */
class LayerCreateDestroyTest : public ::testing::TestWithParam<const char *> {
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
