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
  ASSERT_NE(nullptr, tensor.getData<float>());

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
  ASSERT_NE(nullptr, tensor.getData<float>());

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
  ASSERT_NE(nullptr, tensor.getData<float>());

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
  ASSERT_NE(nullptr, tensor.getData<float>());

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

TEST(nntrainer_Tensor, multiply_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 original;
  original.copy(input);

  status = input.multiply_i(2.0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *data = original.getData<float>();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData<float>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * channel * width * height; ++i) {
    EXPECT_FLOAT_EQ(data[i] + data[i], indata[i]);
  }
}

TEST(nntrainer_Tensor, multiply_i_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 original;
  original.copy(input);

  status = input.multiply_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *data = original.getData<float>();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData<float>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * channel * width * height; ++i) {
    EXPECT_FLOAT_EQ(data[i] * data[i], indata[i]);
  }
}

TEST(nntrainer_Tensor, multiply_i_03_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 target2(batch, channel, height - 2, width - 1);
  status = input.multiply_i(target2);

  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_01_p) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(1, 2, 4, 5);
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
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(3, 1, 4, 5);
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
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(3, 2, 4, 1);
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
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 5);
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
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(1, 2, 1, 5);
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
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(3, 1, 4, 1);
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
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(1, 1, 1, 5);
    float answer_data[] = {
      0, 1,   4,   9,   16,  0, 6,   14,  24,  36,  0, 11,  24,  39,  56,
      0, 16,  34,  54,  76,  0, 21,  44,  69,  96,  0, 26,  54,  84,  116,
      0, 31,  64,  99,  136, 0, 36,  74,  114, 156, 0, 41,  84,  129, 176,
      0, 46,  94,  144, 196, 0, 51,  104, 159, 216, 0, 56,  114, 174, 236,
      0, 61,  124, 189, 256, 0, 66,  134, 204, 276, 0, 71,  144, 219, 296,
      0, 76,  154, 234, 316, 0, 81,  164, 249, 336, 0, 86,  174, 264, 356,
      0, 91,  184, 279, 376, 0, 96,  194, 294, 396, 0, 101, 204, 309, 416,
      0, 106, 214, 324, 436, 0, 111, 224, 339, 456, 0, 116, 234, 354, 476};
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(1, 2, 1, 1);
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
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 1);
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
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4);
    nntrainer::TensorV2 t = rangedV2(3, 5, 1, 4);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 4);
    float answer_data[] = {0,   1,   4,   9,   0,   5,   12,  21,  0,   9,
                           20,  33,  0,   13,  28,  45,  0,   17,  36,  57,
                           80,  105, 132, 161, 96,  125, 156, 189, 112, 145,
                           180, 217, 128, 165, 204, 245, 144, 185, 228, 273,
                           320, 369, 420, 473, 352, 405, 460, 517, 384, 441,
                           500, 561, 416, 477, 540, 605, 448, 513, 580, 649};
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_supported_01_n) {
  nntrainer::TensorV2 target(3, 1, 3, 1);
  nntrainer::TensorV2 target2(3, 1, 3, 3);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_broadcastable_02_n) {
  nntrainer::TensorV2 target(3, 2, 4, 5);
  nntrainer::TensorV2 target2(3, 2, 3, 1);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 result = input.multiply(0.0);
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

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 result = input.multiply(input);

  float *data = result.getData<float>();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData<float>();
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

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 test(batch - 1, height - 1, width - 1);

  EXPECT_THROW({ input.multiply(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(batch, channel, height, 2 * width);
  nntrainer::TensorV2 shared_input =
    input.getSharedDataTensor(dim, 0, false, "");
  nntrainer::TensorV2 test(dim);

  EXPECT_THROW(shared_input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim);
  nntrainer::TensorV2 test(batch, channel, height, 2 * width);
  nntrainer::TensorV2 shared_test = test.getSharedDataTensor(dim, 0, false, "");

  EXPECT_THROW(input.multiply(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim, false);
  nntrainer::TensorV2 test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::TensorV2 test(dim, false);

  EXPECT_THROW(input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::TensorV2 test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::TensorV2 output(dim, false);

  EXPECT_THROW(input.multiply(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_float_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 expected(batch, channel, height, width);
  GEN_TEST_INPUT(expected, (i * (batch * height) + j * (width) + k + 1) * 2);

  nntrainer::TensorV2 result = input.multiply(2.0);

  EXPECT_EQ(result, expected);
}

TEST(nntrainer_Tensor, multiply_strided_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 result = input.multiply_strided(input);

  float *data = result.getData<float>();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData<float>();
  ASSERT_NE(nullptr, indata);

  float *outdata = new float[(input.size())];

  std::transform(indata, indata + batch * channel * height * width, indata,
                 outdata, std::multiplies<float>());

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != outdata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  delete[] outdata;

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_strided_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 test(batch - 1, height - 1, width - 1);

  EXPECT_THROW({ input.multiply_strided(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);
  // input is not allocated now : alloc_now == false
  nntrainer::TensorV2 input(dim, false);
  nntrainer::TensorV2 test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.multiply_strided(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  // test is not allocated.
  nntrainer::TensorV2 test(dim, false);

  EXPECT_THROW(input.multiply_strided(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::TensorV2 test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);
  // output is not allocated
  nntrainer::TensorV2 output(dim, false);

  EXPECT_THROW(input.multiply_strided(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_06_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 output(batch, channel, height, width);
  GEN_TEST_INPUT(output, i * (batch * height) + j * (width) + k + 1);

  float *indata = input.getData<float>();
  ASSERT_NE(nullptr, indata);

  float *outdata_beta = new float[(input.size())];
  float *indata_mul = new float[(input.size())];
  float *outdata = new float[(input.size())];

  std::transform(
    indata, indata + batch * channel * height * width, outdata_beta,
    std::bind(std::multiplies<float>(), std::placeholders::_1, 10.0));

  std::transform(indata, indata + batch * channel * height * width, indata,
                 indata_mul, std::multiplies<float>());
  std::transform(indata_mul, indata_mul + batch * channel * height * width,
                 outdata_beta, outdata, std::plus<float>());

  input.multiply_strided(input, output, 10.0);

  float *data = output.getData<float>();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != outdata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  delete[] outdata_beta;
  delete[] indata_mul;
  delete[] outdata;

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
