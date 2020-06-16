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
  if (tensor.getValue(0, 0, 0, 0) != 0.0)
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

  if (tensor.getValue(0, 0, 0, 1) != 1.0)
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

  if (tensor.getValue(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original;
  original.copy(input);

  status = input.multiply_i(2.0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *data = original.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i] + data[i], indata[i]);
  }
}

TEST(nntrainer_Tensor, multiply_i_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original;
  original.copy(input);

  status = input.multiply_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *data = original.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i] * data[i], indata[i]);
  }
}

TEST(nntrainer_Tensor, multiply_i_01_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor target2(batch, channel, height - 2, width - 1);
  status = input.multiply_i(target2);

  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor result = input.multiply(0.0);
  if (result.getValue(0, 0, 1, 1) != 0.0)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
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
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.multiply(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, divide_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original;
  original.copy(input);

  status = input.divide_i((float)2.0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *data = original.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i], indata[i] + indata[i]);
  }
}

TEST(nntrainer_Tensor, divide_i_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  status = input.divide_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(indata[i], float(1.0));
  }
}

TEST(nntrainer_Tensor, divide_i_01_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  status = input.divide_i((float)0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_i_02_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original(batch, channel, height - 2, width - 1);

  status = input.divide_i(original);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.divide(1.0);

  float *previous = input.getData();
  ASSERT_NE(nullptr, previous);
  float *data = result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i]);
  }
}

TEST(nntrainer_Tensor, divide_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  ASSERT_EXCEPTION({ input.divide(0.0); }, std::runtime_error,
                   "Error: Divide by zero");
}

TEST(nntrainer_Tensor, divide_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.divide(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, add_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

  nntrainer::Tensor original(batch, channel, height, width);
  original.copy(target);

  status = target.add_i(2.1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *previous = original.getData();
  ASSERT_NE(nullptr, previous);
  float *data = target.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i] + (float)2.1);
  }
}

TEST(nntrainer_Tensor, add_i_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor original(batch, height, width);
  original.copy(target);

  status = target.add_i(target, 3.0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *previous = original.getData();
  ASSERT_NE(nullptr, previous);
  float *data = target.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i] * 4.0);
  }
}

/**
 * @brief operand dimension is not right
 */
TEST(nntrainer_Tensor, add_i_01_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor target2(batch, height - 2, width - 3);

  status = target.add_i(target2);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.add(1.0);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] + (float)1.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
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
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.add(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, subtract_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

  nntrainer::Tensor original(batch, height, width);
  original.copy(target);

  status = target.subtract_i(2.1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *previous = original.getData();
  ASSERT_NE(nullptr, previous);
  float *data = target.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i] - (float)2.1);
  }
}

TEST(nntrainer_Tensor, subtract_i_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

  status = target.subtract_i(target);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *data = target.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0);
  }
}

TEST(nntrainer_Tensor, subtract_i_03_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

  nntrainer::Tensor target2(batch, channel, height - 1, width - 3);

  status = target.subtract_i(target2);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, subtract_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
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
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
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
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1);

  ASSERT_EXCEPTION({ input.subtract(test); }, std::runtime_error,
                   "Error: Dimension must be equal each other");
}

TEST(nntrainer_Tensor, sum_01_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  ASSERT_EXCEPTION({ input.sum(4); }, std::out_of_range,
                   "Error: Cannot exceede 3");
}

TEST(nntrainer_Tensor, sum_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  float ans0[1][2][2][10] = {{{{39, 42, 45, 48, 51, 54, 57, 60, 63, 66},
                               {69, 72, 75, 78, 81, 84, 87, 90, 93, 96}},
                              {{57, 60, 63, 66, 69, 72, 75, 78, 81, 84},
                               {87, 90, 93, 96, 99, 102, 105, 108, 111, 114}}}};

  float ans1[3][1][2][10] = {{{{8, 10, 12, 14, 16, 18, 20, 22, 24, 26},
                               {28, 30, 32, 34, 36, 38, 40, 42, 44, 46}}},
                             {{{32, 34, 36, 38, 40, 42, 44, 46, 48, 50},
                               {52, 54, 56, 58, 60, 62, 64, 66, 68, 70}}},
                             {{{56, 58, 60, 62, 64, 66, 68, 70, 72, 74},
                               {76, 78, 80, 82, 84, 86, 88, 90, 92, 94}}}};

  float ans2[3][2][1][10] = {{{{12, 14, 16, 18, 20, 22, 24, 26, 28, 30}},
                              {{24, 26, 28, 30, 32, 34, 36, 38, 40, 42}}},
                             {{{36, 38, 40, 42, 44, 46, 48, 50, 52, 54}},
                              {{48, 50, 52, 54, 56, 58, 60, 62, 64, 66}}},
                             {{{60, 62, 64, 66, 68, 70, 72, 74, 76, 78}},
                              {{72, 74, 76, 78, 80, 82, 84, 86, 88, 90}}}};

  float ans3[3][2][2][1] = {{{{55}, {155}}, {{115}, {215}}},
                            {{{175}, {275}}, {{235}, {335}}},
                            {{{295}, {395}}, {{355}, {455}}}};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height * channel) + j * (batch * height) +
                          k * (width) + l + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result1 = input.sum(1);
  nntrainer::Tensor result2 = input.sum(2);
  nntrainer::Tensor result3 = input.sum(3);

  for (int i = 0; i < result0.getBatch(); ++i) {
    for (int l = 0; l < result0.getChannel(); ++l) {
      for (int j = 0; j < result0.getHeight(); ++j) {
        for (int k = 0; k < result0.getWidth(); ++k) {
          if (ans0[i][l][j][k] != result0.getValue(i, l, j, k)) {
            status = ML_ERROR_RESULT_OUT_OF_RANGE;
            goto end_test;
          }
        }
      }
    }
  }

  for (int i = 0; i < result1.getBatch(); ++i) {
    for (int l = 0; l < result1.getChannel(); ++l) {
      for (int j = 0; j < result1.getHeight(); ++j) {
        for (int k = 0; k < result1.getWidth(); ++k) {
          if (ans1[i][l][j][k] != result1.getValue(i, l, j, k)) {
            status = ML_ERROR_RESULT_OUT_OF_RANGE;
            goto end_test;
          }
        }
      }
    }
  }

  for (int i = 0; i < result2.getBatch(); ++i) {
    for (int l = 0; l < result2.getChannel(); ++l) {
      for (int j = 0; j < result2.getHeight(); ++j) {
        for (int k = 0; k < result2.getWidth(); ++k) {
          if (ans2[i][l][j][k] != result2.getValue(i, l, j, k)) {
            status = ML_ERROR_RESULT_OUT_OF_RANGE;
            goto end_test;
          }
        }
      }
    }
  }

  for (int i = 0; i < result3.getBatch(); ++i) {
    for (int l = 0; l < result3.getChannel(); ++l) {
      for (int j = 0; j < result3.getHeight(); ++j) {
        for (int k = 0; k < result3.getWidth(); ++k) {
          if (ans3[i][l][j][k] != result3.getValue(i, l, j, k)) {
            status = ML_ERROR_RESULT_OUT_OF_RANGE;
            goto end_test;
          }
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
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height * channel) + j * (height * width) +
                          k * width + l + 1);

  nntrainer::Tensor result = input.sum_by_batch();
  if (result.getValue(0, 0, 0, 0) != 820 ||
      result.getValue(1, 0, 0, 0) != 1300 ||
      result.getValue(2, 0, 0, 0) != 1780)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, dot_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 3;
  float ans[3][1][3][3] = {
    {{{30, 36, 42}, {66, 81, 96}, {102, 126, 150}}},
    {{{435, 468, 501}, {552, 594, 636}, {669, 720, 771}}},
    {{{1326, 1386, 1446}, {1524, 1593, 1662}, {1722, 1800, 1878}}}};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (channel * width * height) + j * (height * width) +
                          k * (width) + l + 1);

  nntrainer::Tensor result = input.dot(input);

  for (int i = 0; i < result.getBatch(); ++i) {
    for (int j = 0; j < result.getHeight(); ++j) {
      for (int k = 0; k < result.getWidth(); ++k) {
        if (ans[i][0][j][k] != result.getValue(i, 0, j, k)) {
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
  int channel = 1;
  int height = 3;
  int width = 3;
  float ans[3][1][3][3] = {{{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}},
                           {{{10, 13, 16}, {11, 14, 17}, {12, 15, 18}}},
                           {{{19, 22, 25}, {20, 23, 26}, {21, 24, 27}}}};
  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (channel * width * height) + j * (height * width) +
                          k * (width) + l + 1);
  nntrainer::Tensor result = input.transpose("0:2:1");

  for (int i = 0; i < result.getBatch(); ++i) {
    for (int j = 0; j < result.getHeight(); ++j) {
      for (int k = 0; k < result.getWidth(); ++k) {
        if (ans[i][0][j][k] != result.getValue(i, 0, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_transpose_01_p;
        }
      }
    }
  }
end_transpose_01_p:
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, set_01_p) {
  nntrainer::Tensor tensor = nntrainer::Tensor(1, 1, 1, 1);

  tensor.setZero();
  EXPECT_EQ(tensor.getValue(0, 0, 0, 0), 0.0);

  tensor.setRandUniform(-0.5, 0);
  float val = tensor.getValue(0, 0, 0, 0);
  EXPECT_TRUE(val >= -0.5 && val < 0);
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
