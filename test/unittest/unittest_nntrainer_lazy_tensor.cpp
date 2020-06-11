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
#include <fstream>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

class nntrainer_LazyTensorOpsTest : public ::testing::Test {
protected:
  nntrainer_LazyTensorOpsTest() {}

  virtual void SetUp() {
    target = nntrainer::Tensor(batch, channel, height, width);
    GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + l);

    original.copy(target);
  }

  // virtual void TearDown()
  /**
   * @brief return a tensor filled with contant value
   */
  nntrainer::Tensor constant(float value) {
    nntrainer::Tensor t(batch, channel, height, width);
    return t.apply([value](float) { return value; });
  }

  /**
   * @brief return a tensor filled with contant value with dimension
   */
  nntrainer::Tensor constant(float value, unsigned int batch, unsigned channel,
                             unsigned height, unsigned width) {
    nntrainer::Tensor t(batch, channel, height, width);
    return t.apply([value](float) { return value; });
  }

  void test_eq(nntrainer::Tensor const &A, nntrainer::Tensor const &B) {
    EXPECT_EQ(A.getBatch(), B.getBatch());
    EXPECT_EQ(A.getChannel(), B.getChannel());
    EXPECT_EQ(A.getHeight(), B.getHeight());
    EXPECT_EQ(A.getWidth(), B.getWidth());

    int len = A.getDim().getDataLen();
    const float *aData = A.getData();
    ASSERT_NE(aData, (float *)NULL);
    const float *bData = B.getData();
    ASSERT_NE(bData, (float *)NULL);

    for (int i = 0; i < len; ++i) {
      EXPECT_FLOAT_EQ(aData[i], bData[i]);
    }
  }

  nntrainer::Tensor target;
  nntrainer::Tensor original;
  nntrainer::Tensor expected;

private:
  int batch = 3;
  int height = 2;
  int width = 10;
  int channel = 1;
};

// LazyTensor init test
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

// Simple chain and run
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_01_p) {
  test_eq(target.chain().run(), original);
}

// Simple chain and add_i(float)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_02_p) {
  expected = original.add(2.1);
  test_eq(target.chain().add_i(2.1).run(), expected);
}

// chain and add_i(float) add_i(float)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_03_p) {
  expected = original.add(4.2);
  test_eq(target.chain().add_i(2.1).add_i(2.1).run(), expected);
}

// chain and add_i(float) add_i(float)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_04_p) {
  expected = original.add(4.2);
  test_eq(target.chain().add_i(2.1).add_i(2.1).run(), expected);
}

// chain and add_i(float) add_i(Tensor)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_05_p) {
  expected = original.add(4.1);
  test_eq(target.chain().add_i(2.1).add_i(constant(2.0)).run(), expected);
}

// chain and add_i(float) subtract(float)
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_06_p) {
  test_eq(target.chain().add_i(2.1).subtract_i(2.1).run(), original);
}

// other basic operations (positive)...
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_07_p) {
  target = constant(1.0);
  expected = constant(2.0);
  test_eq(target.chain().multiply_i(2.0).run(), expected);
  test_eq(target.chain().multiply_i(constant(2.0)).run(), expected);

  target = constant(1.0);
  expected = constant(0.5);
  test_eq(target.chain().divide_i(2.0).run(), expected);
  test_eq(target.chain().divide_i(constant(2.0)).run(), expected);
}

// other basic operations (negative)...
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_07_n) {
  EXPECT_THROW(target.chain().add_i(constant(2.0, 1, 1, 1, 1)).run(),
               std::runtime_error);

  EXPECT_THROW(target.chain().subtract_i(constant(2.0, 1, 1, 1, 1)).run(),
               std::runtime_error);

  EXPECT_THROW(target.chain().multiply_i(constant(2.0, 1, 1, 1, 1)).run(),
               std::runtime_error);

  EXPECT_THROW(target.chain().divide_i(constant(2.0, 1, 1, 1, 1)).run(),
               std::runtime_error);
}

// sum()
TEST_F(nntrainer_LazyTensorOpsTest, LazyTensorOps_08_p) {
  target = constant(1.0, 4, 4, 4, 4);
  expected = constant(64.0, 4, 1, 1, 1);
  test_eq(target.chain().sum_by_batch().run(), expected);

  expected = constant(4.0, 1, 4, 4, 4);
  test_eq(target.chain().sum(0).run(), expected);

  expected = constant(4.0, 4, 1, 4, 4);
  test_eq(target.chain().sum(1).run(), expected);

  expected = constant(4.0, 4, 4, 1, 4);
  test_eq(target.chain().sum(2).run(), expected);

  expected = constant(4.0, 4, 4, 4, 1);
  test_eq(target.chain().sum(3).run(), expected);
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
