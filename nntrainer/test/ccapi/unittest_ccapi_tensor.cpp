// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_ccapi_tensor.cpp
 * @date        11 December 2023
 * @brief       cc API Tensor Unit tests.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <iostream>

#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <tensor_api.h>

/**
 * @brief Tensor Construct Test
 */

TEST(nntrainer_ccapi, tensor_01_p) {

  std::shared_ptr<ml::train::Layer> layer;

  ml::train::Tensor a;

  EXPECT_NO_THROW(layer =
                    ml::train::layer::Input({"name=input0", "input_shape=1:1:2",
                                             "normalization=true"}));

  EXPECT_NO_THROW(a.setSrcLayer(layer));

  std::shared_ptr<ml::train::Layer> layer_b = a.getSrcLayer();
  EXPECT_EQ(layer_b->getName(), "input0");
}
