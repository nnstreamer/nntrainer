// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
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

TEST(nntrainer_TensorDim, setTensorDim_03_n) {
  nntrainer::TensorDim d;

  EXPECT_THROW(d.setTensorDim(0, 0), std::invalid_argument);
  EXPECT_THROW(d.setTensorDim(1, 0), std::invalid_argument);
  EXPECT_THROW(d.setTensorDim(2, 0), std::invalid_argument);
  EXPECT_THROW(d.setTensorDim(3, 0), std::invalid_argument);
}

TEST(nntrainer_TensorDim, setTensorDim_04_p) {
  nntrainer::TensorDim d;

  d.setTensorDim(0, 4);
  d.setTensorDim(1, 5);
  d.setTensorDim(2, 6);
  d.setTensorDim(3, 7);

  EXPECT_EQ(d.batch(), 4);
  EXPECT_EQ(d.channel(), 5);
  EXPECT_EQ(d.height(), 6);
  EXPECT_EQ(d.width(), 7);
}

TEST(nntrainer_Tensor, Tensor_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Tensor tensor = nntrainer::Tensor(1, 2, 3);
  tensor.setZero();
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

  EXPECT_THROW({ input.multiply(test); }, std::runtime_error);
}

TEST(nntrainer_Tensor, multiply_float_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor expected(batch, channel, height, width);
  GEN_TEST_INPUT(expected, (i * (batch * height) + j * (width) + k + 1) * 2);

  nntrainer::Tensor result = input.multiply(2.0);

  EXPECT_EQ(result, expected);
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

  EXPECT_THROW({ input.divide(0.0); }, std::runtime_error);
}

TEST(nntrainer_Tensor, divide_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1);

  EXPECT_THROW({ input.divide(test); }, std::runtime_error);
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

TEST(nntrainer_Tensor, add_i_broadcast_01_p) {
  nntrainer::TensorDim ref_dim{3, 2, 4, 5};
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 2, 4, 5);
    float answer_data[] = {
      0,   2,   4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,  26,
      28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,  54,
      56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,  40,  42,
      44,  46,  48,  50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,
      72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,
      100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 80,  82,  84,  86,
      88,  90,  92,  94,  96,  98,  100, 102, 104, 106, 108, 110, 112, 114,
      116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142,
      144, 146, 148, 150, 152, 154, 156, 158};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 1, 4, 5);
    float answer_data[] = {
      0,   2,   4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,  26,
      28,  30,  32,  34,  36,  38,  20,  22,  24,  26,  28,  30,  32,  34,
      36,  38,  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62,
      64,  66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,
      92,  94,  96,  98,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,
      100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
      128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154,
      156, 158, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162,
      164, 166, 168, 170, 172, 174, 176, 178};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 2, 4, 1);
    float answer_data[] = {
      0,   1,   2,   3,   4,   6,   7,   8,   9,   10,  12,  13,  14,  15,
      16,  18,  19,  20,  21,  22,  24,  25,  26,  27,  28,  30,  31,  32,
      33,  34,  36,  37,  38,  39,  40,  42,  43,  44,  45,  46,  48,  49,
      50,  51,  52,  54,  55,  56,  57,  58,  60,  61,  62,  63,  64,  66,
      67,  68,  69,  70,  72,  73,  74,  75,  76,  78,  79,  80,  81,  82,
      84,  85,  86,  87,  88,  90,  91,  92,  93,  94,  96,  97,  98,  99,
      100, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116,
      117, 118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133,
      134, 135, 136, 138, 139, 140, 141, 142};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 1, 1, 5);
    float answer_data[] = {
      0,   2,   4,   6,   8,   5,   7,   9,   11,  13,  10,  12,  14,  16,
      18,  15,  17,  19,  21,  23,  20,  22,  24,  26,  28,  25,  27,  29,
      31,  33,  30,  32,  34,  36,  38,  35,  37,  39,  41,  43,  45,  47,
      49,  51,  53,  50,  52,  54,  56,  58,  55,  57,  59,  61,  63,  60,
      62,  64,  66,  68,  65,  67,  69,  71,  73,  70,  72,  74,  76,  78,
      75,  77,  79,  81,  83,  80,  82,  84,  86,  88,  90,  92,  94,  96,
      98,  95,  97,  99,  101, 103, 100, 102, 104, 106, 108, 105, 107, 109,
      111, 113, 110, 112, 114, 116, 118, 115, 117, 119, 121, 123, 120, 122,
      124, 126, 128, 125, 127, 129, 131, 133};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 2, 1, 5);
    float answer_data[] = {
      0,   2,   4,   6,   8,   5,   7,   9,   11,  13,  10,  12,  14,  16,
      18,  15,  17,  19,  21,  23,  25,  27,  29,  31,  33,  30,  32,  34,
      36,  38,  35,  37,  39,  41,  43,  40,  42,  44,  46,  48,  40,  42,
      44,  46,  48,  45,  47,  49,  51,  53,  50,  52,  54,  56,  58,  55,
      57,  59,  61,  63,  65,  67,  69,  71,  73,  70,  72,  74,  76,  78,
      75,  77,  79,  81,  83,  80,  82,  84,  86,  88,  80,  82,  84,  86,
      88,  85,  87,  89,  91,  93,  90,  92,  94,  96,  98,  95,  97,  99,
      101, 103, 105, 107, 109, 111, 113, 110, 112, 114, 116, 118, 115, 117,
      119, 121, 123, 120, 122, 124, 126, 128};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 1, 4, 1);
    float answer_data[] = {
      0,   1,   2,   3,   4,   6,   7,   8,   9,   10,  12,  13,  14,  15,
      16,  18,  19,  20,  21,  22,  20,  21,  22,  23,  24,  26,  27,  28,
      29,  30,  32,  33,  34,  35,  36,  38,  39,  40,  41,  42,  44,  45,
      46,  47,  48,  50,  51,  52,  53,  54,  56,  57,  58,  59,  60,  62,
      63,  64,  65,  66,  64,  65,  66,  67,  68,  70,  71,  72,  73,  74,
      76,  77,  78,  79,  80,  82,  83,  84,  85,  86,  88,  89,  90,  91,
      92,  94,  95,  96,  97,  98,  100, 101, 102, 103, 104, 106, 107, 108,
      109, 110, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 120, 121,
      122, 123, 124, 126, 127, 128, 129, 130};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 1, 1, 5);
    float answer_data[] = {
      0,   2,   4,   6,   8,   5,   7,   9,   11,  13,  10,  12,  14,  16,
      18,  15,  17,  19,  21,  23,  20,  22,  24,  26,  28,  25,  27,  29,
      31,  33,  30,  32,  34,  36,  38,  35,  37,  39,  41,  43,  40,  42,
      44,  46,  48,  45,  47,  49,  51,  53,  50,  52,  54,  56,  58,  55,
      57,  59,  61,  63,  60,  62,  64,  66,  68,  65,  67,  69,  71,  73,
      70,  72,  74,  76,  78,  75,  77,  79,  81,  83,  80,  82,  84,  86,
      88,  85,  87,  89,  91,  93,  90,  92,  94,  96,  98,  95,  97,  99,
      101, 103, 100, 102, 104, 106, 108, 105, 107, 109, 111, 113, 110, 112,
      114, 116, 118, 115, 117, 119, 121, 123};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 2, 1, 1);
    float answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  21,  22,  23,  24,  25,  26,  27,  28,
      29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
      71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  80,  81,  82,  83,
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
      98,  99,  101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
      113, 114, 115, 116, 117, 118, 119, 120};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 1, 1, 1);
    float answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  41,  42,
      43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
      57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
      71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  82,  83,  84,  85,
      86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
      114, 115, 116, 117, 118, 119, 120, 121};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 1, 1, 1);
    m.add_i(1.0);
    float answer_data[] = {
      1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,
      29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
      43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
      57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
      71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
      85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,
      99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
      113, 114, 115, 116, 117, 118, 119, 120};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4);
    nntrainer::Tensor t = ranged(3, 5, 1, 4);
    nntrainer::Tensor m = ranged(3, 1, 1, 4);
    float answer_data[] = {0,  2,  4,  6,  4,  6,  8,  10, 8,  10, 12, 14,
                           12, 14, 16, 18, 16, 18, 20, 22, 24, 26, 28, 30,
                           28, 30, 32, 34, 32, 34, 36, 38, 36, 38, 40, 42,
                           40, 42, 44, 46, 48, 50, 52, 54, 52, 54, 56, 58,
                           56, 58, 60, 62, 60, 62, 64, 66, 64, 66, 68, 70};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, add_i_broadcast_not_supported_01_n) {
  nntrainer::Tensor target(3, 1, 3, 1);
  nntrainer::Tensor target2(3, 1, 3, 3);

  EXPECT_EQ(target.add_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_i_broadcast_not_broadcastable_02_n) {
  nntrainer::Tensor target(3, 2, 4, 5);
  nntrainer::Tensor target2(3, 2, 3, 1);

  EXPECT_EQ(target.add_i(target2), ML_ERROR_INVALID_PARAMETER);
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

  EXPECT_THROW({ input.add(test); }, std::runtime_error);
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

  EXPECT_THROW({ input.subtract(test); }, std::runtime_error);
}

TEST(nntrainer_Tensor, subtract_float_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor expected(batch, channel, height, width);
  GEN_TEST_INPUT(expected, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor result = input.subtract(1.0);

  EXPECT_EQ(result, expected);
}

TEST(nntrainer_Tensor, sum_01_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  EXPECT_THROW({ input.sum(4); }, std::out_of_range);
}

TEST(nntrainer_Tensor, sum_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  EXPECT_THROW({ input.sum(-1); }, std::out_of_range);
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

  for (unsigned int i = 0; i < result0.batch(); ++i) {
    for (unsigned int l = 0; l < result0.channel(); ++l) {
      for (unsigned int j = 0; j < result0.height(); ++j) {
        for (unsigned int k = 0; k < result0.width(); ++k) {
          if (ans0[i][l][j][k] != result0.getValue(i, l, j, k)) {
            status = ML_ERROR_RESULT_OUT_OF_RANGE;
            goto end_test;
          }
        }
      }
    }
  }

  for (unsigned int i = 0; i < result1.batch(); ++i) {
    for (unsigned int l = 0; l < result1.channel(); ++l) {
      for (unsigned int j = 0; j < result1.height(); ++j) {
        for (unsigned int k = 0; k < result1.width(); ++k) {
          if (ans1[i][l][j][k] != result1.getValue(i, l, j, k)) {
            status = ML_ERROR_RESULT_OUT_OF_RANGE;
            goto end_test;
          }
        }
      }
    }
  }

  for (unsigned int i = 0; i < result2.batch(); ++i) {
    for (unsigned int l = 0; l < result2.channel(); ++l) {
      for (unsigned int j = 0; j < result2.height(); ++j) {
        for (unsigned int k = 0; k < result2.width(); ++k) {
          if (ans2[i][l][j][k] != result2.getValue(i, l, j, k)) {
            status = ML_ERROR_RESULT_OUT_OF_RANGE;
            goto end_test;
          }
        }
      }
    }
  }

  for (unsigned int i = 0; i < result3.batch(); ++i) {
    for (unsigned int l = 0; l < result3.channel(); ++l) {
      for (unsigned int j = 0; j < result3.height(); ++j) {
        for (unsigned int k = 0; k < result3.width(); ++k) {
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

TEST(nntrainer_Tensor, multiple_sum_invalid_args_01_n) {
  nntrainer::Tensor t = constant(1.0, 1, 1, 1, 1);
  EXPECT_THROW(t.sum(std::vector<unsigned int>()), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiple_sum_out_of_range_n) {
  nntrainer::Tensor t = constant(1.0, 1, 1, 1, 1);
  EXPECT_THROW(t.sum({7}), std::out_of_range);
}

TEST(nntrainer_Tensor, multiple_sum_p) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7);
  nntrainer::Tensor actual, expected;

  actual = t.sum({0, 1});
  expected = constant(2 * 3, 1, 1, 5, 7);
  EXPECT_EQ(actual, expected);

  actual = t.sum({1, 2, 3});
  expected = constant(3 * 5 * 7, 2, 1, 1, 1);
  EXPECT_EQ(actual, expected);

  actual = t.sum({3, 1});
  expected = constant(7 * 3, 2, 1, 5, 1);
  EXPECT_EQ(actual, expected);

  actual = t.sum({3, 1}, 0.5);
  expected = constant(7 * 3 * 0.5, 2, 1, 5, 1);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_p) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7);

  nntrainer::Tensor actual, expected;

  actual = t.average();
  expected = constant(1.0, 1, 1, 1, 1);
  EXPECT_EQ(actual, expected);

  int idx = 0;
  t = t.apply([&](float in) { return idx++ % 2; });

  actual = t.average();
  expected = constant(0.5, 1, 1, 1, 1);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_axis_p) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2);
  int idx = 0;
  std::function<float(float)> f = [&](float in) { return idx++ % 2; };
  t = t.apply(f);

  nntrainer::Tensor actual, expected;

  actual = t.average(0);
  expected = constant(0, 1, 2, 2, 2).apply(f);
  EXPECT_EQ(actual, expected);

  actual = t.average(1);
  expected = constant(0, 2, 1, 2, 2).apply(f);
  EXPECT_EQ(actual, expected);

  actual = t.average(2);
  expected = constant(0, 2, 2, 1, 2).apply(f);
  EXPECT_EQ(actual, expected);

  actual = t.average(3);
  expected = constant(0.5, 2, 2, 2, 1);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_axis_out_of_range_01_n) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2);
  EXPECT_THROW(t.average(-1), std::out_of_range);
}

TEST(nntrainer_Tensor, average_axis_out_of_range_02_n) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2);
  EXPECT_THROW(t.average(7), std::out_of_range);
}

TEST(nntrainer_Tensor, average_multiple_axes_p) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7);
  nntrainer::Tensor actual, expected;

  actual = t.average({0, 1, 2});
  expected = constant(1.0, 1, 1, 1, 7);
  EXPECT_EQ(actual, expected);

  actual = t.average({0, 1, 2, 3});
  expected = constant(1.0, 1, 1, 1, 1);
  EXPECT_EQ(actual, expected);

  actual = t.average({3, 1});
  expected = constant(1.0, 2, 1, 5, 1);
  EXPECT_EQ(actual, expected);

  actual = t.average({3, 1, 1, 1, 3});
  expected = constant(1.0, 2, 1, 5, 1);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_multiple_axes_01_n) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7);
  EXPECT_THROW(t.average({5, 7}), std::out_of_range);
}

TEST(nntrainer_Tensor, dot_01_n) {
  nntrainer::Tensor input(2, 3, 4, 5);
  nntrainer::Tensor m(1, 3, 4, 5);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m),
               nntrainer::exception::not_supported);
}

TEST(nntrainer_Tensor, dot_02_n) {
  nntrainer::Tensor input(2, 3, 4, 5);
  nntrainer::Tensor m(1, 3, 4, 5);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m, true),
               nntrainer::exception::not_supported);
}

TEST(nntrainer_Tensor, dot_03_n) {
  nntrainer::Tensor input(1, 3, 4, 5);
  nntrainer::Tensor m(1, 3, 4, 5);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m, true),
               nntrainer::exception::not_supported);
}

TEST(nntrainer_Tensor, dot_04_n) {
  nntrainer::Tensor input(2, 3, 4, 5);
  nntrainer::Tensor m(1, 1, 4, 5);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m), std::runtime_error);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, false, true));
}

TEST(nntrainer_Tensor, dot_05_p) {
  int status = ML_ERROR_NONE;
  int batch = 2;
  int channel = 3;
  int height = 4;
  int width = 5;
  float ans[2][3][4][24] = {0};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (channel * width * height) + j * (height * width) +
                          k * (width) + l + 1);
  nntrainer::Tensor weight(batch, channel, height, width);
  GEN_TEST_INPUT(weight, i * (channel * width * height) + j * (height * width) +
                           k * (width) + l + 1);
  weight.reshape({1, 1, 24, 5});

  nntrainer::Tensor result = input.dot(weight, false, true);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int k = 0; k < batch * channel * height; k++) {
          ans[b][c][h][k] = 0;
          for (int w = 0; w < width; w++) {
            float val1 = input.getValue(b, c, h, w);
            float val2 = weight.getValue(0, 0, k, w);
            ans[b][c][h][k] += val1 * val2;
          }
        }
      }
    }
  }

  for (unsigned int i = 0; i < result.batch(); ++i) {
    for (unsigned int c = 0; c < result.channel(); ++c) {
      for (unsigned int j = 0; j < result.height(); ++j) {
        for (unsigned int k = 0; k < result.width(); ++k) {
          float val1 = ans[i][c][j][k];
          float val2 = result.getValue(i, c, j, k);
          if (val1 != val2) {
            status = ML_ERROR_RESULT_OUT_OF_RANGE;
            goto end_dot_01_p;
          }
        }
      }
    }
  }
end_dot_01_p:
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, dot_06_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 3;
  float ans[3][1][1][3] = {
    {{{30, 36, 42}}}, {{{66, 81, 96}}}, {{{102, 126, 150}}}};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (channel * width * height) + j * (height * width) +
                          k * (width) + l + 1);

  nntrainer::Tensor result = input.dot(input);

  for (unsigned int i = 0; i < result.batch(); ++i) {
    for (unsigned int j = 0; j < result.height(); ++j) {
      for (unsigned int k = 0; k < result.width(); ++k) {
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

  for (unsigned int i = 0; i < result.batch(); ++i) {
    for (unsigned int j = 0; j < result.height(); ++j) {
      for (unsigned int k = 0; k < result.width(); ++k) {
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

TEST(nntrainer_Tensor, save_read_01_p) {
  int batch = 3;
  int channel = 4;
  int height = 5;
  int width = 6;
  nntrainer::Tensor target(3, 4, 5, 6);
  nntrainer::Tensor readed(3, 4, 5, 6);

  GEN_TEST_INPUT(target, i * (channel * width * height) + j * (height * width) +
                           k * (width) + l + 1);

  std::ofstream save_file("save.bin", std::ios::out | std::ios::binary);
  target.save(save_file);
  save_file.close();

  std::ifstream read_file("save.bin");
  readed.read(read_file);
  read_file.close();

  EXPECT_EQ(target, readed);

  int status = std::remove("save.bin");

  ASSERT_EQ(status, 0);
}

TEST(nntrainer_Tensor, save_read_01_n) {
  int batch = 3;
  int channel = 4;
  int height = 5;
  int width = 6;
  nntrainer::Tensor target(3, 4, 5, 6);
  nntrainer::Tensor readed(3, 4, 1, 1);

  GEN_TEST_INPUT(target, i * (channel * width * height) + j * (height * width) +
                           k * (width) + l + 1);

  std::ofstream save_file("save.bin", std::ios::out | std::ios::binary);
  target.save(save_file);
  save_file.close();

  std::ifstream read_file("save.bin");
  readed.read(read_file);
  read_file.close();

  EXPECT_NE(target, readed);

  int status = std::remove("save.bin");

  ASSERT_EQ(status, 0);
}

TEST(nntrainer_Tensor, copy_and_shares_variable_p) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6);
  nntrainer::Tensor B = A.clone();
  nntrainer::Tensor C = A;

  C.setValue(1, 1, 1, 1, 2.0f);

  EXPECT_EQ(A, C);
  EXPECT_NE(B, C);

  C.reshape(nntrainer::TensorDim(3, 4, 6, 5));
  EXPECT_EQ(A.getDim(), B.getDim());
  EXPECT_NE(A.getDim(), C.getDim());
}

TEST(nntrainer_Tensor, reshape_n_01) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6);

  EXPECT_THROW(A.reshape(nntrainer::TensorDim(9, 9, 9, 9)),
               std::invalid_argument);
}

TEST(nntrainer_Tensor, reshape_n_02) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6);
  nntrainer::TensorDim A_dim = A.getDim();

  /** Changing the dim of a tensor only affects local copy of the dim */
  A_dim.setTensorDim(1, 100);
  EXPECT_EQ(A_dim.getTensorDim(1), 100);

  nntrainer::TensorDim A_dim_2 = A.getDim();
  EXPECT_EQ(A_dim_2.getTensorDim(1), 4);
}

TEST(nntrainer_Tensor, copy_and_reshape_n) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6);
  nntrainer::Tensor B = A;
  nntrainer::Tensor C = A.clone();

  EXPECT_THROW(B.reshape(nntrainer::TensorDim(9, 9, 9, 9)),
               std::invalid_argument);
}

/// @note this test case demonstrates it is dangerous to use sharedConstTensor
/// to const correct the inner data.
TEST(nntrainer_Tensor, constructor_from_shared_const_ptr_shares_variable_n) {
  nntrainer::sharedConstTensor A =
    MAKE_SHARED_TENSOR(constant(1.0f, 3, 4, 5, 6));

  nntrainer::Tensor B = *A;
  nntrainer::Tensor C = A->clone();

  B.setValue(2, 3, 4, 5, 2.0f);
  EXPECT_EQ(*A, B);
  EXPECT_NE(*A, C);

  C.reshape(nntrainer::TensorDim(3, 4, 6, 5));
  EXPECT_EQ(A->getDim(), B.getDim());
  EXPECT_NE(A->getDim(), C.getDim());
}

TEST(nntrainer_Tensor, print_small_size) {
  nntrainer::Tensor target = constant(1.0, 3, 1, 2, 3);

  std::cerr << target;
  std::stringstream ss, expected;
  ss << target;

  expected << '<' << typeid(target).name() << " at " << &target << ">\n"
           << "Shape: 3:1:2:3\n"
           << "         1          1          1 \n"
           << "         1          1          1 \n"
           << "\n"
           << "-------\n"
           << "         1          1          1 \n"
           << "         1          1          1 \n"
           << "\n"
           << "-------\n"
           << "         1          1          1 \n"
           << "         1          1          1 \n"
           << "\n"
           << "-------\n";

  EXPECT_EQ(ss.str(), expected.str());
}

TEST(nntrainer_Tensor, print_large_size) {
  nntrainer::Tensor target = constant(1.2, 3, 10, 10, 10);

  std::stringstream ss, expected;

  expected << '<' << typeid(target).name() << " at " << &target << ">\n"
           << "Shape: 3:10:10:10\n"
           << "[1.2 1.2 1.2 ... 1.2 1.2 1.2]\n";
  ss << target;

  EXPECT_EQ(ss.str(), expected.str());
}

TEST(nntrainer_Tensor, private_external_loop_n) {
  nntrainer::Tensor t = ranged(3, 5, 1, 4);
  nntrainer::Tensor b = ranged(1, 5, 1, 4);

  auto vector_func = [](float *buf, int stride, int size) {
    float *cur = buf;
    std::cerr << "[ ";
    for (int i = 0; i < size; ++i) {
      std::cerr << *cur << ' ';
      cur += stride;
    }
    std::cerr << "]\n";
  };

  nntrainer::ExternalLoopInfo e;
  EXPECT_NO_THROW(e = t.computeExternalLoop(b));

  float *buf = t.getData();
  float *mbuf = b.getData();

  auto &t_strides = t.getStrides();
  unsigned int offset;
  unsigned int m_offset;

  if (e.buffer_axis == -1) {
    vector_func(buf, t_strides[3], e.buffer_size);
    vector_func(mbuf, e.strides[3], e.buffer_size);
  } else {
    for (unsigned int b = 0; b < t.batch(); ++b) {
      if (e.buffer_axis == 0) {
        offset = b * t_strides[0];
        m_offset = b * e.strides[0];
        std::cerr << "=====================\n";
        vector_func(buf + offset, t_strides[3], e.buffer_size);
        vector_func(mbuf + m_offset, e.strides[3], e.buffer_size);
        continue;
      }
      for (unsigned int c = 0; c < t.channel(); ++c) {
        if (e.buffer_axis == 1) {
          offset = b * t_strides[0] + c * t_strides[1];
          m_offset = b * e.strides[0] + c * e.strides[1];
          std::cerr << "=====================\n";
          vector_func(buf + offset, t_strides[3], e.buffer_size);
          vector_func(mbuf + m_offset, e.strides[3], e.buffer_size);
          continue;
        }
        for (unsigned int h = 0; h < t.height(); ++h) {
          if (e.buffer_axis == 2) {
            offset = b * t_strides[0] + c * t_strides[1] + h * t_strides[2];
            m_offset = b * e.strides[0] + c * e.strides[1] + h * e.strides[2];
            std::cerr << "=====================\n";
            vector_func(buf + offset, t_strides[3], e.buffer_size);
            vector_func(mbuf + m_offset, e.strides[3], e.buffer_size);
            continue;
          }
          for (unsigned int w = 0; w < t.width(); ++w) {
            offset = b * t_strides[0] + c * t_strides[1] + h * t_strides[2] +
                     w * t_strides[3];
            m_offset = b * e.strides[0] + c * e.strides[1] + h * e.strides[2] +
                       w * e.strides[3];
            std::cerr << "after width: " << offset << std::endl;
            std::cerr << "=====================\n";
            vector_func(buf + offset, t_strides[3], e.buffer_size);
            vector_func(mbuf + m_offset, e.strides[3], e.buffer_size);
          }
        }
      }
    }
  }
  std::cerr << "buffer_axis: " << e.buffer_axis << std::endl;
  std::cerr << "t strides: ";
  for (auto i : t_strides)
    std::cerr << i << ' ';
  std::cerr << std::endl;
  std::cerr << "strides: ";
  for (auto i : e.strides)
    std::cerr << i << ' ';
  std::cerr << std::endl;
  std::cerr << "buffer_size: " << e.buffer_size << std::endl;
  std::cerr << "buffer_cnt: " << e.buffer_cnt << std::endl;
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  return result;
}
