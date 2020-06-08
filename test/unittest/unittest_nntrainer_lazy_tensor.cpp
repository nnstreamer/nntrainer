/* SPDX-License-Identifier: Apache-2.0-only
 * Copyright (C) 2020 Jihoon Lee <jihoon.it.lee@samsung.com>
 *
 * @file	unittest_nntrainer_lazy_tensor.cpp
 * @date	05 Jun 2020
 * @brief	A unittest for nntrainer_lazy_tensor
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jihoon Lee <jihoon.it.lee@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <lazy_tensor.h>
#include <fstream>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

TEST(nntrainer_LazyTensor, LazyTensor_01_p) {
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + l);

  nntrainer::LazyTensor delayed(target);

  nntrainer::Tensor result;
  result = delayed.run();

  float *expected = target.getData();
  ASSERT_NE(expected, (float *)NULL);
  float *current = result.getData();
  ASSERT_NE(current, (float *)NULL);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(current[i], expected[i]);
  }
}

TEST(nntrainer_LazyTensor, LazyTensor_02_p) {
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result;
  result = target.chain().run();

  float *expected = target.getData();
  ASSERT_NE(expected, (float *)NULL);
  float *current = result.getData();
  ASSERT_NE(current, (float *)NULL);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(current[i], expected[i]);
  }
}

TEST(nntrainer_LazyTensor, LazyTensor_03_p) {
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result;
  result = target.chain().add_i(2.1).run();

  float *expected = target.getData();
  ASSERT_NE(expected, (float *)NULL);
  float *current = result.getData();
  ASSERT_NE(current, (float *)NULL);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(current[i], expected[i] + 2.1);
  }
}

TEST(nntrainer_LazyTensor, LazyTensor_04_p) {
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result;
  result = target.chain().add_i(2.1).add_i(1.0).run();

  float *expected = target.getData();
  ASSERT_NE(expected, (float *)NULL);
  float *current = result.getData();
  ASSERT_NE(current, (float *)NULL);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(current[i], expected[i] + 3.1);
  }
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  testing::InitGoogleTest(&argc, argv);

  result = RUN_ALL_TESTS();

  return result;
}
