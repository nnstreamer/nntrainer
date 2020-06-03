/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file        unittest_nntrainer_tensor.cpp
 * @date        03 June 2020
 * @brief       Unit test utility for tensor.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include "nntrainer_test_util.h"
#include "util_func.h"
#include <fstream>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

TEST(nntrainer_TensorDim, setTensorDim_01_p) {
  int status = ML_ERROR_NONE;

  nntrainer::TensorDim tensor_dim;
  status = tensor_dim.setTensorDim("1:2:3:4");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_TensorDim, setTensorDim_02_n) {
  int status = ML_ERROR_NONE;

  nntrainer::TensorDim tensor_dim;
  status = tensor_dim.setTensorDim("1:2:3:4:5");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, Tensor_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Tensor tensor = nntrainer::Tensor(1, 2, 3);
  ASSERT_NE(nullptr, tensor.getData());
  if (tensor.getValue(0, 0, 0) != 0.0)
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

  nntrainer::Tensor tensor = nntrainer::Tensor(in);
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue(0, 0, 1) != 1.0)
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

  nntrainer::Tensor tensor = nntrainer::Tensor(in);
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue(0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor result = input.multiply(0.0);
  if (result.getValue(0, 1, 1) != 0.0)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.multiply(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] * indata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_03_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.multiply(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, divide_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.divide(1.0);
  if (result.getValue(0, 1, 1) != input.getValue(0, 1, 1))
    status = ML_ERROR_RESULT_OUT_OF_RANGE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, divide_02_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  ASSERT_EXCEPTION({ input.divide(0.0); }, std::runtime_error,
                   "Error: Divide by zero");
}

TEST(nntrainer_Tensor, divide_03_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.divide(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, add_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.add(1.0);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] + 1.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.add(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] + indata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_03_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.add(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, subtract_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.subtract(1.0);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] - 1.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, subtract_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.subtract(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != 0.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, subtract_03_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.subtract(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, sum_01_n) {
  int batch = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  ASSERT_EXCEPTION({ input.sum(3); }, std::out_of_range,
                   "Error: Cannot exceede 2");
}

TEST(nntrainer_Tensor, sum_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 2;
  int width = 10;

  float ans0[1][2][10] = {{{21, 24, 27, 30, 33, 36, 39, 42, 45, 48},
                           {51, 54, 57, 60, 63, 66, 69, 72, 75, 78}}};

  float ans1[3][1][10] = {{{18, 20, 22, 24, 26, 28, 30, 32, 34, 36}},
                          {{24, 26, 28, 30, 32, 34, 36, 38, 40, 42}},
                          {{30, 32, 34, 36, 38, 40, 42, 44, 46, 48}}};
  float ans2[3][2][1] = {{{154}, {164}}, {{160}, {170}}, {{166}, {176}}};

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result1 = input.sum(1);
  nntrainer::Tensor result2 = input.sum(2);

  for (int i = 0; i < result0.getBatch(); ++i) {
    for (int j = 0; j < result0.getHeight(); ++j) {
      for (int k = 0; k < result0.getWidth(); ++k) {
        if (ans0[i][j][k] != result0.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_test;
        }
      }
    }
  }

  for (int i = 0; i < result1.getBatch(); ++i) {
    for (int j = 0; j < result1.getHeight(); ++j) {
      for (int k = 0; k < result1.getWidth(); ++k) {
        if (ans1[i][j][k] != result1.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_test;
        }
      }
    }
  }

  for (int i = 0; i < result2.getBatch(); ++i) {
    for (int j = 0; j < result2.getHeight(); ++j) {
      for (int k = 0; k < result2.getWidth(); ++k) {
        if (ans2[i][j][k] != result2.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_test;
        }
      }
    }
  }

end_test:
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, sum_03_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 2;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor result = input.sum();
  if (result.getValue(0, 0, 0) != 210 || result.getValue(1, 0, 0) != 330 ||
      result.getValue(2, 0, 0) != 450)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, dot_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 3;
  float ans[3][3][3] = {
    {{30, 36, 42}, {66, 81, 96}, {102, 126, 150}},
    {{435, 468, 501}, {552, 594, 636}, {669, 720, 771}},
    {{1326, 1386, 1446}, {1524, 1593, 1662}, {1722, 1800, 1878}}};

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor result = input.dot(input);

  for (int i = 0; i < result.getBatch(); ++i) {
    for (int j = 0; j < result.getHeight(); ++j) {
      for (int k = 0; k < result.getWidth(); ++k) {
        if (ans[i][j][k] != result.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_dot_01_p;
        }
      }
    }
  }
end_dot_01_p:
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, transpose_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 3;
  float ans[3][3][3] = {{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}},
                        {{10, 13, 16}, {11, 14, 17}, {12, 15, 18}},
                        {{19, 22, 25}, {20, 23, 26}, {21, 24, 27}}};
  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor result = input.transpose();

  for (int i = 0; i < result.getBatch(); ++i) {
    for (int j = 0; j < result.getHeight(); ++j) {
      for (int k = 0; k < result.getWidth(); ++k) {
        if (ans[i][j][k] != result.getValue(i, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_transpose_01_p;
        }
      }
    }
  }
end_transpose_01_p:
  EXPECT_EQ(status, ML_ERROR_NONE);
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
