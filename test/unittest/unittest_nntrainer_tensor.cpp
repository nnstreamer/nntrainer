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

TEST(nntrainer_Tensor, multiply_i_broadcast_p_01) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 2, 4, 5);
    float answer_data[] = {
      0,    1,    4,    9,    16,   25,   36,   49,   64,   81,   100,  121,
      144,  169,  196,  225,  256,  289,  324,  361,  400,  441,  484,  529,
      576,  625,  676,  729,  784,  841,  900,  961,  1024, 1089, 1156, 1225,
      1296, 1369, 1444, 1521, 0,    41,   84,   129,  176,  225,  276,  329,
      384,  441,  500,  561,  624,  689,  756,  825,  896,  969,  1044, 1121,
      1200, 1281, 1364, 1449, 1536, 1625, 1716, 1809, 1904, 2001, 2100, 2201,
      2304, 2409, 2516, 2625, 2736, 2849, 2964, 3081, 0,    81,   164,  249,
      336,  425,  516,  609,  704,  801,  900,  1001, 1104, 1209, 1316, 1425,
      1536, 1649, 1764, 1881, 2000, 2121, 2244, 2369, 2496, 2625, 2756, 2889,
      3024, 3161, 3300, 3441, 3584, 3729, 3876, 4025, 4176, 4329, 4484, 4641};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 1, 4, 5);
    float answer_data[] = {
      0,    1,    4,    9,    16,   25,   36,   49,   64,   81,   100,  121,
      144,  169,  196,  225,  256,  289,  324,  361,  0,    21,   44,   69,
      96,   125,  156,  189,  224,  261,  300,  341,  384,  429,  476,  525,
      576,  629,  684,  741,  800,  861,  924,  989,  1056, 1125, 1196, 1269,
      1344, 1421, 1500, 1581, 1664, 1749, 1836, 1925, 2016, 2109, 2204, 2301,
      1200, 1281, 1364, 1449, 1536, 1625, 1716, 1809, 1904, 2001, 2100, 2201,
      2304, 2409, 2516, 2625, 2736, 2849, 2964, 3081, 3200, 3321, 3444, 3569,
      3696, 3825, 3956, 4089, 4224, 4361, 4500, 4641, 4784, 4929, 5076, 5225,
      5376, 5529, 5684, 5841, 4000, 4141, 4284, 4429, 4576, 4725, 4876, 5029,
      5184, 5341, 5500, 5661, 5824, 5989, 6156, 6325, 6496, 6669, 6844, 7021};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 2, 4, 1);
    float answer_data[] = {
      0,    0,    0,    0,    0,    5,    6,    7,    8,    9,    20,   22,
      24,   26,   28,   45,   48,   51,   54,   57,   80,   84,   88,   92,
      96,   125,  130,  135,  140,  145,  180,  186,  192,  198,  204,  245,
      252,  259,  266,  273,  320,  328,  336,  344,  352,  405,  414,  423,
      432,  441,  500,  510,  520,  530,  540,  605,  616,  627,  638,  649,
      720,  732,  744,  756,  768,  845,  858,  871,  884,  897,  980,  994,
      1008, 1022, 1036, 1125, 1140, 1155, 1170, 1185, 1280, 1296, 1312, 1328,
      1344, 1445, 1462, 1479, 1496, 1513, 1620, 1638, 1656, 1674, 1692, 1805,
      1824, 1843, 1862, 1881, 2000, 2020, 2040, 2060, 2080, 2205, 2226, 2247,
      2268, 2289, 2420, 2442, 2464, 2486, 2508, 2645, 2668, 2691, 2714, 2737};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 1, 1, 5);
    float answer_data[] = {
      0,    1,    4,    9,    16,   0,    6,    14,   24,   36,   0,    11,
      24,   39,   56,   0,    16,   34,   54,   76,   0,    21,   44,   69,
      96,   0,    26,   54,   84,   116,  0,    31,   64,   99,   136,  0,
      36,   74,   114,  156,  200,  246,  294,  344,  396,  225,  276,  329,
      384,  441,  250,  306,  364,  424,  486,  275,  336,  399,  464,  531,
      300,  366,  434,  504,  576,  325,  396,  469,  544,  621,  350,  426,
      504,  584,  666,  375,  456,  539,  624,  711,  800,  891,  984,  1079,
      1176, 850,  946,  1044, 1144, 1246, 900,  1001, 1104, 1209, 1316, 950,
      1056, 1164, 1274, 1386, 1000, 1111, 1224, 1339, 1456, 1050, 1166, 1284,
      1404, 1526, 1100, 1221, 1344, 1469, 1596, 1150, 1276, 1404, 1534, 1666};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 2, 1, 5);
    float answer_data[] = {
      0,   1,   4,    9,   16,  0,   6,   14,  24,  36,  0,   11,  24,  39,
      56,  0,   16,   34,  54,  76,  100, 126, 154, 184, 216, 125, 156, 189,
      224, 261, 150,  186, 224, 264, 306, 175, 216, 259, 304, 351, 0,   41,
      84,  129, 176,  0,   46,  94,  144, 196, 0,   51,  104, 159, 216, 0,
      56,  114, 174,  236, 300, 366, 434, 504, 576, 325, 396, 469, 544, 621,
      350, 426, 504,  584, 666, 375, 456, 539, 624, 711, 0,   81,  164, 249,
      336, 0,   86,   174, 264, 356, 0,   91,  184, 279, 376, 0,   96,  194,
      294, 396, 500,  606, 714, 824, 936, 525, 636, 749, 864, 981, 550, 666,
      784, 904, 1026, 575, 696, 819, 944, 1071};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 1, 4, 1);
    float answer_data[] = {
      0,    0,    0,    0,    0,    5,    6,    7,    8,    9,    20,   22,
      24,   26,   28,   45,   48,   51,   54,   57,   0,    0,    0,    0,
      0,    25,   26,   27,   28,   29,   60,   62,   64,   66,   68,   105,
      108,  111,  114,  117,  160,  164,  168,  172,  176,  225,  230,  235,
      240,  245,  300,  306,  312,  318,  324,  385,  392,  399,  406,  413,
      240,  244,  248,  252,  256,  325,  330,  335,  340,  345,  420,  426,
      432,  438,  444,  525,  532,  539,  546,  553,  640,  648,  656,  664,
      672,  765,  774,  783,  792,  801,  900,  910,  920,  930,  940,  1045,
      1056, 1067, 1078, 1089, 800,  808,  816,  824,  832,  945,  954,  963,
      972,  981,  1100, 1110, 1120, 1130, 1140, 1265, 1276, 1287, 1298, 1309};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 1, 1, 5);
    float answer_data[] = {
      0, 1,   4,   9,   16,  0, 6,   14,  24,  36,  0, 11,  24,  39,  56,
      0, 16,  34,  54,  76,  0, 21,  44,  69,  96,  0, 26,  54,  84,  116,
      0, 31,  64,  99,  136, 0, 36,  74,  114, 156, 0, 41,  84,  129, 176,
      0, 46,  94,  144, 196, 0, 51,  104, 159, 216, 0, 56,  114, 174, 236,
      0, 61,  124, 189, 256, 0, 66,  134, 204, 276, 0, 71,  144, 219, 296,
      0, 76,  154, 234, 316, 0, 81,  164, 249, 336, 0, 86,  174, 264, 356,
      0, 91,  184, 279, 376, 0, 96,  194, 294, 396, 0, 101, 204, 309, 416,
      0, 106, 214, 324, 436, 0, 111, 224, 339, 456, 0, 116, 234, 354, 476};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(1, 2, 1, 1);
    float answer_data[] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 1, 1, 1);
    float answer_data[] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  160, 162, 164, 166,
      168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194,
      196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222,
      224, 226, 228, 230, 232, 234, 236, 238};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4);
    nntrainer::Tensor t = ranged(3, 5, 1, 4);
    nntrainer::Tensor m = ranged(3, 1, 1, 4);
    float answer_data[] = {0,   1,   4,   9,   0,   5,   12,  21,  0,   9,
                           20,  33,  0,   13,  28,  45,  0,   17,  36,  57,
                           80,  105, 132, 161, 96,  125, 156, 189, 112, 145,
                           180, 217, 128, 165, 204, 245, 144, 185, 228, 273,
                           320, 369, 420, 473, 352, 405, 460, 517, 384, 441,
                           500, 561, 416, 477, 540, 605, 448, 513, 580, 649};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_supported_01_n) {
  nntrainer::Tensor target(3, 1, 3, 1);
  nntrainer::Tensor target2(3, 1, 3, 3);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_broadcastable_02_n) {
  nntrainer::Tensor target(3, 2, 4, 5);
  nntrainer::Tensor target2(3, 2, 3, 1);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
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

TEST(nntrainer_Tensor, divide_i_broadcast_01_p) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(1, 2, 4, 5);
    m.add_i(1);
    float answer_data[] = {
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       1.0,       1.0,       41.0,      21.0,
      14.333333, 11.0,      9.0,       7.6666665, 6.714286,  6.0,
      5.4444447, 5.0,       4.6363635, 4.3333335, 4.076923,  3.857143,
      3.6666667, 3.5,       3.3529413, 3.2222223, 3.1052632, 3.0,
      2.9047618, 2.8181818, 2.7391305, 2.6666667, 2.6,       2.5384614,
      2.4814816, 2.4285715, 2.3793104, 2.3333333, 2.2903225, 2.25,
      2.2121212, 2.1764705, 2.142857,  2.1111112, 2.0810812, 2.0526316,
      2.025641,  2.0,       81.0,      41.0,      27.666666, 21.0,
      17.0,      14.333333, 12.428572, 11.0,      9.888889,  9.0,
      8.272727,  7.6666665, 7.1538463, 6.714286,  6.3333335, 6.0,
      5.7058825, 5.4444447, 5.2105265, 5.0,       4.8095236, 4.6363635,
      4.478261,  4.3333335, 4.2,       4.076923,  3.9629629, 3.857143,
      3.7586207, 3.6666667, 3.580645,  3.5,       3.4242425, 3.3529413,
      3.2857144, 3.2222223, 3.162162,  3.1052632, 3.0512822, 3.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 1, 4, 5);
    m.add_i(1);
    float answer_data[] = {
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
      1.0,       1.0,       21.0,      11.0,      7.6666665, 6.0,
      5.0,       4.3333335, 3.857143,  3.5,       3.2222223, 3.0,
      2.8181818, 2.6666667, 2.5384614, 2.4285715, 2.3333333, 2.25,
      2.1764705, 2.1111112, 2.0526316, 2.0,       1.9523809, 1.9090909,
      1.8695652, 1.8333334, 1.8,       1.7692307, 1.7407408, 1.7142857,
      1.6896552, 1.6666666, 1.6451613, 1.625,     1.6060606, 1.5882353,
      1.5714285, 1.5555556, 1.5405406, 1.5263158, 1.5128205, 1.5,
      2.9047618, 2.8181818, 2.7391305, 2.6666667, 2.6,       2.5384614,
      2.4814816, 2.4285715, 2.3793104, 2.3333333, 2.2903225, 2.25,
      2.2121212, 2.1764705, 2.142857,  2.1111112, 2.0810812, 2.0526316,
      2.025641,  2.0,       1.9756098, 1.9523809, 1.9302325, 1.9090909,
      1.8888888, 1.8695652, 1.8510638, 1.8333334, 1.8163265, 1.8,
      1.7843137, 1.7692307, 1.754717,  1.7407408, 1.7272727, 1.7142857,
      1.7017543, 1.6896552, 1.6779661, 1.6666666, 2.4634147, 2.4285715,
      2.3953488, 2.3636363, 2.3333333, 2.3043478, 2.2765958, 2.25,
      2.2244897, 2.2,       2.1764705, 2.1538463, 2.1320755, 2.1111112,
      2.090909,  2.0714285, 2.0526316, 2.0344827, 2.0169492, 2.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 2, 4, 1);
    m.add_i(1);
    float answer_data[] = {
      1.0,       2.0,       3.0,       4.0,       5.0,       3.0,
      3.5,       4.0,       4.5,       5.0,       3.6666667, 4.0,
      4.3333335, 4.6666665, 5.0,       4.0,       4.25,      4.5,
      4.75,      5.0,       4.2,       4.4,       4.6,       4.8,
      5.0,       4.3333335, 4.5,       4.6666665, 4.8333335, 5.0,
      4.428571,  4.571429,  4.714286,  4.857143,  5.0,       4.5,
      4.625,     4.75,      4.875,     5.0,       4.5555553, 4.6666665,
      4.7777777, 4.888889,  5.0,       4.6,       4.7,       4.8,
      4.9,       5.0,       4.6363635, 4.7272725, 4.818182,  4.909091,
      5.0,       4.6666665, 4.75,      4.8333335, 4.9166665, 5.0,
      4.6923075, 4.769231,  4.8461537, 4.923077,  5.0,       4.714286,
      4.785714,  4.857143,  4.928571,  5.0,       4.733333,  4.8,
      4.866667,  4.9333334, 5.0,       4.75,      4.8125,    4.875,
      4.9375,    5.0,       4.7647057, 4.8235292, 4.882353,  4.9411764,
      5.0,       4.7777777, 4.8333335, 4.888889,  4.9444447, 5.0,
      4.7894735, 4.8421054, 4.894737,  4.9473686, 5.0,       4.8,
      4.85,      4.9,       4.95,      5.0,       4.8095236, 4.857143,
      4.904762,  4.952381,  5.0,       4.818182,  4.8636365, 4.909091,
      4.9545455, 5.0,       4.826087,  4.869565,  4.9130435, 4.9565215,
      5.0,       4.8333335, 4.875,     4.9166665, 4.9583335, 5.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 1, 1, 5);
    m.add_i(1);
    float answer_data[] = {
      1.0,       1.0,       1.0,       1.0,       1.0,       6.0,
      3.5,       2.6666667, 2.25,      2.0,       11.0,      6.0,
      4.3333335, 3.5,       3.0,       16.0,      8.5,       6.0,
      4.75,      4.0,       21.0,      11.0,      7.6666665, 6.0,
      5.0,       26.0,      13.5,      9.333333,  7.25,      6.0,
      31.0,      16.0,      11.0,      8.5,       7.0,       36.0,
      18.5,      12.666667, 9.75,      8.0,       6.8333335, 6.0,
      5.375,     4.888889,  4.5,       7.6666665, 6.714286,  6.0,
      5.4444447, 5.0,       8.5,       7.428571,  6.625,     6.0,
      5.5,       9.333333,  8.142858,  7.25,      6.5555553, 6.0,
      10.166667, 8.857142,  7.875,     7.111111,  6.5,       11.0,
      9.571428,  8.5,       7.6666665, 7.0,       11.833333, 10.285714,
      9.125,     8.222222,  7.5,       12.666667, 11.0,      9.75,
      8.777778,  8.0,       7.3636365, 6.8333335, 6.3846154, 6.0,
      5.6666665, 7.818182,  7.25,      6.769231,  6.357143,  6.0,
      8.272727,  7.6666665, 7.1538463, 6.714286,  6.3333335, 8.727273,
      8.083333,  7.5384617, 7.071429,  6.6666665, 9.181818,  8.5,
      7.923077,  7.428571,  7.0,       9.636364,  8.916667,  8.307693,
      7.785714,  7.3333335, 10.090909, 9.333333,  8.692307,  8.142858,
      7.6666665, 10.545455, 9.75,      9.076923,  8.5,       8.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(1, 2, 1, 5);
    m.add_i(1);
    float answer_data[] = {
      1.0,       1.0,       1.0,       1.0,       1.0,       6.0,
      3.5,       2.6666667, 2.25,      2.0,       11.0,      6.0,
      4.3333335, 3.5,       3.0,       16.0,      8.5,       6.0,
      4.75,      4.0,       3.5,       3.142857,  2.875,     2.6666667,
      2.5,       4.3333335, 3.857143,  3.5,       3.2222223, 3.0,
      5.1666665, 4.571429,  4.125,     3.7777777, 3.5,       6.0,
      5.285714,  4.75,      4.3333335, 4.0,       41.0,      21.0,
      14.333333, 11.0,      9.0,       46.0,      23.5,      16.0,
      12.25,     10.0,      51.0,      26.0,      17.666666, 13.5,
      11.0,      56.0,      28.5,      19.333334, 14.75,     12.0,
      10.166667, 8.857142,  7.875,     7.111111,  6.5,       11.0,
      9.571428,  8.5,       7.6666665, 7.0,       11.833333, 10.285714,
      9.125,     8.222222,  7.5,       12.666667, 11.0,      9.75,
      8.777778,  8.0,       81.0,      41.0,      27.666666, 21.0,
      17.0,      86.0,      43.5,      29.333334, 22.25,     18.0,
      91.0,      46.0,      31.0,      23.5,      19.0,      96.0,
      48.5,      32.666668, 24.75,     20.0,      16.833334, 14.571428,
      12.875,    11.555555, 10.5,      17.666666, 15.285714, 13.5,
      12.111111, 11.0,      18.5,      16.0,      14.125,    12.666667,
      11.5,      19.333334, 16.714285, 14.75,     13.222222, 12.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 1, 4, 1);
    m.add_i(1);
    float answer_data[] = {
      1.0,       2.0,       3.0,       4.0,       5.0,       3.0,
      3.5,       4.0,       4.5,       5.0,       3.6666667, 4.0,
      4.3333335, 4.6666665, 5.0,       4.0,       4.25,      4.5,
      4.75,      5.0,       21.0,      22.0,      23.0,      24.0,
      25.0,      13.0,      13.5,      14.0,      14.5,      15.0,
      10.333333, 10.666667, 11.0,      11.333333, 11.666667, 9.0,
      9.25,      9.5,       9.75,      10.0,      8.2,       8.4,
      8.6,       8.8,       9.0,       7.6666665, 7.8333335, 8.0,
      8.166667,  8.333333,  7.285714,  7.428571,  7.571429,  7.714286,
      7.857143,  7.0,       7.125,     7.25,      7.375,     7.5,
      12.2,      12.4,      12.6,      12.8,      13.0,      11.0,
      11.166667, 11.333333, 11.5,      11.666667, 10.142858, 10.285714,
      10.428572, 10.571428, 10.714286, 9.5,       9.625,     9.75,
      9.875,     10.0,      9.0,       9.111111,  9.222222,  9.333333,
      9.444445,  8.6,       8.7,       8.8,       8.9,       9.0,
      8.272727,  8.363636,  8.454545,  8.545455,  8.636364,  8.0,
      8.083333,  8.166667,  8.25,      8.333333,  11.222222, 11.333333,
      11.444445, 11.555555, 11.666667, 10.6,      10.7,      10.8,
      10.9,      11.0,      10.090909, 10.181818, 10.272727, 10.363636,
      10.454545, 9.666667,  9.75,      9.833333,  9.916667,  10.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(1, 1, 1, 5);
    m.add_i(1);
    float answer_data[] = {
      1.0,   1.0,  1.0,       1.0,  1.0,  6.0,   3.5,  2.6666667, 2.25,  2.0,
      11.0,  6.0,  4.3333335, 3.5,  3.0,  16.0,  8.5,  6.0,       4.75,  4.0,
      21.0,  11.0, 7.6666665, 6.0,  5.0,  26.0,  13.5, 9.333333,  7.25,  6.0,
      31.0,  16.0, 11.0,      8.5,  7.0,  36.0,  18.5, 12.666667, 9.75,  8.0,
      41.0,  21.0, 14.333333, 11.0, 9.0,  46.0,  23.5, 16.0,      12.25, 10.0,
      51.0,  26.0, 17.666666, 13.5, 11.0, 56.0,  28.5, 19.333334, 14.75, 12.0,
      61.0,  31.0, 21.0,      16.0, 13.0, 66.0,  33.5, 22.666666, 17.25, 14.0,
      71.0,  36.0, 24.333334, 18.5, 15.0, 76.0,  38.5, 26.0,      19.75, 16.0,
      81.0,  41.0, 27.666666, 21.0, 17.0, 86.0,  43.5, 29.333334, 22.25, 18.0,
      91.0,  46.0, 31.0,      23.5, 19.0, 96.0,  48.5, 32.666668, 24.75, 20.0,
      101.0, 51.0, 34.333332, 26.0, 21.0, 106.0, 53.5, 36.0,      27.25, 22.0,
      111.0, 56.0, 37.666668, 28.5, 23.0, 116.0, 58.5, 39.333332, 29.75, 24.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(1, 2, 1, 1);
    m.add_i(1);
    float answer_data[] = {
      1.0,  2.0,  3.0,  4.0,   5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0,  17.0, 18.0, 19.0, 20.0, 10.5, 11.0, 11.5, 12.0,
      12.5, 13.0, 13.5, 14.0,  14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0,
      18.5, 19.0, 19.5, 20.0,  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
      49.0, 50.0, 51.0, 52.0,  53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
      30.5, 31.0, 31.5, 32.0,  32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0,
      36.5, 37.0, 37.5, 38.0,  38.5, 39.0, 39.5, 40.0, 81.0, 82.0, 83.0, 84.0,
      85.0, 86.0, 87.0, 88.0,  89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0,
      97.0, 98.0, 99.0, 100.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0,
      54.5, 55.0, 55.5, 56.0,  56.5, 57.0, 57.5, 58.0, 58.5, 59.0, 59.5, 60.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 1, 1, 1);
    m.add_i(1);
    float answer_data[] = {
      1.0,       2.0,       3.0,  4.0,       5.0,       6.0,
      7.0,       8.0,       9.0,  10.0,      11.0,      12.0,
      13.0,      14.0,      15.0, 16.0,      17.0,      18.0,
      19.0,      20.0,      21.0, 22.0,      23.0,      24.0,
      25.0,      26.0,      27.0, 28.0,      29.0,      30.0,
      31.0,      32.0,      33.0, 34.0,      35.0,      36.0,
      37.0,      38.0,      39.0, 40.0,      20.5,      21.0,
      21.5,      22.0,      22.5, 23.0,      23.5,      24.0,
      24.5,      25.0,      25.5, 26.0,      26.5,      27.0,
      27.5,      28.0,      28.5, 29.0,      29.5,      30.0,
      30.5,      31.0,      31.5, 32.0,      32.5,      33.0,
      33.5,      34.0,      34.5, 35.0,      35.5,      36.0,
      36.5,      37.0,      37.5, 38.0,      38.5,      39.0,
      39.5,      40.0,      27.0, 27.333334, 27.666666, 28.0,
      28.333334, 28.666666, 29.0, 29.333334, 29.666666, 30.0,
      30.333334, 30.666666, 31.0, 31.333334, 31.666666, 32.0,
      32.333332, 32.666668, 33.0, 33.333332, 33.666668, 34.0,
      34.333332, 34.666668, 35.0, 35.333332, 35.666668, 36.0,
      36.333332, 36.666668, 37.0, 37.333332, 37.666668, 38.0,
      38.333332, 38.666668, 39.0, 39.333332, 39.666668, 40.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4);
    nntrainer::Tensor t = ranged(3, 5, 1, 4);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 1, 1, 4);
    m.add_i(1);
    float answer_data[] = {
      1.0,       1.0,       1.0,       1.0,       5.0,       3.0,
      2.3333333, 2.0,       9.0,       5.0,       3.6666667, 3.0,
      13.0,      7.0,       5.0,       4.0,       17.0,      9.0,
      6.3333335, 5.0,       4.2,       3.6666667, 3.2857144, 3.0,
      5.0,       4.3333335, 3.857143,  3.5,       5.8,       5.0,
      4.428571,  4.0,       6.6,       5.6666665, 5.0,       4.5,
      7.4,       6.3333335, 5.571429,  5.0,       4.5555553, 4.2,
      3.909091,  3.6666667, 5.0,       4.6,       4.2727275, 4.0,
      5.4444447, 5.0,       4.6363635, 4.3333335, 5.888889,  5.4,
      5.0,       4.6666665, 6.3333335, 5.8,       5.3636365, 5.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, divide_i_broadcast_not_supported_01_n) {
  nntrainer::Tensor target(3, 1, 3, 1);
  nntrainer::Tensor target2(3, 1, 3, 3);

  EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_i_broadcast_not_broadcastable_02_n) {
  nntrainer::Tensor target(3, 2, 4, 5);
  nntrainer::Tensor target2(3, 2, 3, 1);

  EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
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
  {
    nntrainer::TensorDim ref_dim(1, 1, 2, 1);
    nntrainer::Tensor t = ranged(1, 1, 2, 1);
    nntrainer::Tensor m = ranged(1, 1, 2, 1);
    float answer_data[] = {0.0, 2.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(16, 1, 1, 1);
    nntrainer::Tensor t = ranged(16, 1, 1, 1);
    nntrainer::Tensor m = ranged(1, 1, 1, 1);
    float answer_data[] = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                           8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
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

TEST(nntrainer_Tensor, DISABLED_broadcast_info_n) {
  nntrainer::Tensor t = ranged(1, 1, 2, 1);
  nntrainer::Tensor b = ranged(1, 1, 2, 1);

  auto vector_func = [](float *buf, int stride, int size) {
    float *cur = buf;
    std::cerr << "[ ";
    for (int i = 0; i < size; ++i) {
      std::cerr << *cur << ' ';
      cur += stride;
    }
    std::cerr << "]\n";
  };

  nntrainer::BroadcastInfo e;
  EXPECT_NO_THROW(e = t.computeBroadcastInfo(b));

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
}

TEST(nntrainer_Tensor, DISABLED_equation_test_01_p) {
  nntrainer::Tensor a, b, c;
  nntrainer::Tensor ret1, ret2;

  a = randUniform(4, 6, 7, 3, -100, 100);
  b = randUniform(4, 6, 7, 3, -100, 100);
  c = randUniform(4, 6, 7, 3, -100, 100);

  ret1 = a.subtract(b).multiply(c);
  ret2 = a.multiply(c).subtract(b.multiply(c));

  float *data1 = ret1.getData();
  float *data2 = ret2.getData();
  EXPECT_EQ(ret1, ret2);

  for (unsigned int i = 0; i < ret1.length(); ++i) {
    EXPECT_FLOAT_EQ(data1[i], data2[i]);
  }
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
