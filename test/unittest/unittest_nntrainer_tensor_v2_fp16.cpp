// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file        unittest_nntrainer_tensor_v2_fp16.cpp
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
  nntrainer::TensorV2 tensor = nntrainer::TensorV2(
    1, 2, 3, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  tensor.setZero();
  ASSERT_NE(nullptr, tensor.getData());
  if (tensor.getValue<_FP16>(0, 0, 0, 0) != 0.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_02_p) {
  int status = ML_ERROR_NONE;
  int height = 3;
  int width = 10;
  std::vector<std::vector<_FP16>> in;
  for (int i = 0; i < height; ++i) {
    std::vector<_FP16> tv;
    for (int j = 0; j < width; ++j) {
      tv.push_back(static_cast<_FP16>(i * 2.0 + j));
    }
    in.push_back(tv);
  }

  nntrainer::TensorV2 tensor = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue<_FP16>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_02_nhwc_p) {
  int status = ML_ERROR_NONE;
  int width = 10;
  int channel = 3;
  std::vector<std::vector<_FP16>> in;
  for (int i = 0; i < width; ++i) {
    std::vector<_FP16> tv;
    for (int j = 0; j < channel; ++j) {
      tv.push_back(static_cast<_FP16>(i * 2.0 + j));
    }
    in.push_back(tv);
  }

  nntrainer::TensorV2 tensor = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue<_FP16>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_03_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  std::vector<std::vector<std::vector<_FP16>>> in;

  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<_FP16>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<_FP16> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(static_cast<_FP16>(k * height * width + i * width + j));
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::TensorV2 tensor = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue<_FP16>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
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
