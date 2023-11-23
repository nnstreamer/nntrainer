// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file        unittest_nntrainer_tensor_v2.cpp
 * @date        16 November 2023
 * @brief       Unit test utility for tensor v2.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <fstream>
#include <nntrainer_error.h>
#include <tensor_dim.h>
#include <tensor_v2.h>

TEST(nntrainer_Tensor, Tensor_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::TensorV2 tensor = nntrainer::TensorV2(1, 2, 3);
  tensor.setZero();
  ASSERT_NE(nullptr, tensor.getData());
  if (tensor.getValue<float>(0, 0, 0, 0) != 0.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_02_p) {
  int status = ML_ERROR_NONE;
  int height = 3;
  int width = 10;
  std::vector<std::vector<float>> in;
  for (int i = 0; i < height; ++i) {
    std::vector<float> tv;
    for (int j = 0; j < width; ++j) {
      tv.push_back(i * 2.0 + j);
    }
    in.push_back(tv);
  }

  nntrainer::TensorV2 tensor = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue<float>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_02_nhwc_p) {
  int status = ML_ERROR_NONE;
  int width = 10;
  int channel = 3;
  std::vector<std::vector<float>> in;
  for (int i = 0; i < width; ++i) {
    std::vector<float> tv;
    for (int j = 0; j < channel; ++j) {
      tv.push_back(i * 2.0 + j);
    }
    in.push_back(tv);
  }

  nntrainer::TensorV2 tensor = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue<float>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_03_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  std::vector<std::vector<std::vector<float>>> in;

  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<float>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<float> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::TensorV2 tensor = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue<float>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_04_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  std::vector<std::vector<std::vector<float>>> in;

  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<float>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<float> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::TensorV2 t0 = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  // copy assignment operator
  nntrainer::TensorV2 t1 = t0;

  if (t1.getValue<float>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);

  // comparison operator
  EXPECT_EQ(t0, t1);
}

TEST(nntrainer_Tensor, Tensor_05_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  std::vector<std::vector<std::vector<float>>> in;

  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<float>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<float> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::TensorV2 t0 = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  // copy assignment operator
  nntrainer::TensorV2 t1 = nntrainer::TensorV2(batch, height, width);
  t1.setRandNormal(2.3, 0.5);

  float val_t0 = t0.getValue<float>(0, 0, 0, 1);
  float val_t1 = t1.getValue<float>(0, 0, 0, 1);

  swap(t0, t1);

  if (t0.getValue<float>(0, 0, 0, 1) != val_t1)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);

  if (t1.getValue<float>(0, 0, 0, 1) != val_t0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, TensorPaddedValue_p) {
  nntrainer::TensorV2 a = rangedV2(1, 1, 3, 3);
  float default_padded = -1;

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      float expected = default_padded;
      if (1 <= i && i <= 3 && 1 <= j && j <= 3) {
        expected = (i - 1) * 3 + (j - 1);
      }
      float actual = a.getValuePaddedVirtual(0, 0, i, j, 1, 1, default_padded);
      EXPECT_FLOAT_EQ(actual, expected);
    }
  }
}

TEST(nntrainer_Tensor, empty_01) {
  nntrainer::TensorV2 t;

  EXPECT_TRUE(t.empty());
}

TEST(nntrainer_Tensor, empty_02) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, false);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, empty_03) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, true);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, allocate_01_n) {
  nntrainer::TensorV2 t;
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_FALSE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_02_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, false);
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_03_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, true);
  EXPECT_TRUE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, initialize_01_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, true, nntrainer::Initializer::ONES);

  nntrainer::TensorV2 golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_02_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, true);

  nntrainer::TensorV2 golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_NE(golden, t);

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_03_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, false, nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::TensorV2 golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_04_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, false);
  t.initialize(nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::TensorV2 golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_05_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, false);
  t.allocate();

  nntrainer::TensorV2 golden(1, 2, 3, 4);
  golden.setValue(1.f);

  /**
   * Ideally, it should be NE, but it can be equal due to no initialization
   * EXPECT_NE(golden, t);
   */

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_06_n) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, true, nntrainer::Initializer::ONES);
  nntrainer::TensorV2 golden({1, 2, 3, 4}, true, nntrainer::Initializer::ZEROS);

  EXPECT_NE(golden, t);

  golden.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_07_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, true, nntrainer::Initializer::ONES);

  nntrainer::TensorV2 golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);

  t.setValue(0, 0, 0, 0, 0);
  t.setValue(0, 0, 0, t.size() - 1, 0);
  EXPECT_NE(golden, t);

  t.initialize();
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_08_p) {
  nntrainer::TensorV2 t({1, 2, 3, 4}, true, nntrainer::Initializer::ONES);

  nntrainer::TensorV2 golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);

  t.initialize(nntrainer::Initializer::HE_NORMAL);
  EXPECT_NE(golden, t);

  t.initialize();
  EXPECT_NE(golden, t);

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);

  t.initialize();
  EXPECT_EQ(golden, t);
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
