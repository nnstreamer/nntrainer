// SPDX-License-Identifier: Apache-2.0
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
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <fstream>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

TEST(nntrainer_TensorDim, ctor_initializer_p) {
  unsigned int b = 3;
  unsigned int c = 2;
  unsigned int h = 4;
  unsigned int w = 5;

  nntrainer::TensorDim t = {w};
  EXPECT_EQ(nntrainer::TensorDim(1, 1, 1, w), t);

  t = {h, w};
  EXPECT_EQ(nntrainer::TensorDim(1, 1, h, w), t);

  t = {c, h, w};
  EXPECT_EQ(nntrainer::TensorDim(1, c, h, w), t);

  t = {b, c, h, w};
  EXPECT_EQ(nntrainer::TensorDim(b, c, h, w), t);
}

TEST(nntrainer_TensorDim, default_constructor_1_sized_dimShapes_p) {
  unsigned int b = 3;
  unsigned int c = 2;
  unsigned int h = 4;
  unsigned int w = 5;

  EXPECT_EQ(nntrainer::TensorDim(c), nntrainer::TensorDim(1, 1, 1, c));
  EXPECT_EQ(nntrainer::TensorDim(1, 1, w, c), nntrainer::TensorDim(w, c));
  EXPECT_EQ(nntrainer::TensorDim(1, h, w, c), nntrainer::TensorDim(h, w, c));
}

TEST(nntrianer_TensorDim, effective_dimension_p) {
  nntrainer::TensorDim t(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                         nntrainer::Tdatatype::FP32);
  EXPECT_EQ(t.getEffectiveDimension(), std::vector<int>({3, 2, 4, 5}));

  t.setEffDimFlag(0b1101);
  EXPECT_EQ(t.getEffectiveDimension(), std::vector<int>({3, 2, 5}));

  t.setEffDimFlag(0b0011);
  EXPECT_EQ(t.getEffectiveDimension(), std::vector<int>({4, 5}));

  t.setEffDimFlag(0b1111);
  EXPECT_EQ(t.getEffectiveDimension(), std::vector<int>({3, 2, 4, 5}));

  t.setEffDimFlag(0b1100);
  EXPECT_EQ(t.getEffectiveDimension(), std::vector<int>({3, 2}));

  t.setDynDimFlag(0b1100);
  EXPECT_EQ(t.getEffectiveDimension(true), std::vector<int>({-1, -1}));

  auto copied_t = t;
  EXPECT_EQ(copied_t.getEffectiveDimension(), std::vector<int>({3, 2}));
  EXPECT_EQ(copied_t.getEffectiveDimension(true), std::vector<int>({-1, -1}));

  auto moved_t = std::move(copied_t);
  EXPECT_EQ(moved_t.getEffectiveDimension(), std::vector<int>({3, 2}));
  EXPECT_EQ(moved_t.getEffectiveDimension(true), std::vector<int>({-1, -1}));
}

TEST(nntrainer_TensorDim, ctor_initializer_n) {
  EXPECT_THROW(nntrainer::TensorDim t({1, 2, 3, 4, 5}), std::invalid_argument);
}

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

  EXPECT_EQ(d.batch(), 4u);
  EXPECT_EQ(d.channel(), 5u);
  EXPECT_EQ(d.height(), 6u);
  EXPECT_EQ(d.width(), 7u);
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

// TEST(nntrainer_Tensor, Tensor_02_p) {
//   int status = ML_ERROR_NONE;
//   int height = 3;
//   int width = 10;
//   std::vector<std::vector<float>> in;
//   for (int i = 0; i < height; ++i) {
//     std::vector<float> tv;
//     for (int j = 0; j < width; ++j) {
//       tv.push_back(i * 2.0 + j);
//     }
//     in.push_back(tv);
//   }

//   nntrainer::Tensor tensor = nntrainer::Tensor(in);
//   ASSERT_NE(nullptr, tensor.getData());

//   if (tensor.getValue(0, 0, 0, 1) != 1.0)
//     status = ML_ERROR_INVALID_PARAMETER;
//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, Tensor_02_nhwc_p) {
//   int status = ML_ERROR_NONE;
//   int width = 10;
//   int channel = 3;
//   std::vector<std::vector<float>> in;
//   for (int i = 0; i < width; ++i) {
//     std::vector<float> tv;
//     for (int j = 0; j < channel; ++j) {
//       tv.push_back(i * 2.0 + j);
//     }
//     in.push_back(tv);
//   }

//   nntrainer::Tensor tensor = nntrainer::Tensor(in, NHWC_);
//   ASSERT_NE(nullptr, tensor.getData());

//   if (tensor.getValue(0, 0, 0, 1) != 1.0)
//     status = ML_ERROR_INVALID_PARAMETER;
//   EXPECT_EQ(status, ML_ERROR_NONE);

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

  nntrainer::Tensor tensor = nntrainer::Tensor(
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
  std::vector<std::vector<std::vector<int8_t>>> in;

  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<int8_t>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<int8_t> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::Tensor tensor = nntrainer::Tensor(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
  ASSERT_NE(nullptr, tensor.getData<int8_t>());

  if (tensor.getValue<int8_t>(0, 0, 0, 1) != 1)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_05_p) {
  int status = ML_ERROR_NONE;
  std::vector<std::vector<std::vector<uint16_t>>> in = {{{0, 1}, {2, 3}},
                                                        {{4, 5}, {6, 7}},
                                                        {{8, 9}, {10, 11}},
                                                        {{12, 13}, {14, 15}}};

  nntrainer::Tensor tensor = nntrainer::Tensor(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16});
  ASSERT_NE(nullptr, tensor.getData<uint16_t>());

  for (size_t b = 0; b < tensor.batch(); ++b) {
    for (size_t c = 0; c < tensor.channel(); ++c) {
      for (size_t h = 0; h < tensor.height(); ++h) {
        for (size_t w = 0; w < tensor.width(); ++w) {
          size_t idx = tensor.getIndex(b, c, h, w);
          ASSERT_EQ(idx, tensor.getValue<uint16_t>(idx));
        }
      }
    }
  }
}

// TEST(nntrainer_Tensor, Tensor_06_p) {
//   int status = ML_ERROR_NONE;
//   nntrainer::Tensor tensor = nntrainer::Tensor(
//     1, 4, 2, 2, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4});
//   ASSERT_NE(nullptr, tensor.getData<uint8_t>());

//   tensor.setValue(2);

//   for (size_t b = 0; b < tensor.batch(); ++b) {
//     for (size_t c = 0; c < tensor.channel(); ++c) {
//       for (size_t h = 0; h < tensor.height(); ++h) {
//         for (size_t w = 0; w < tensor.width(); ++w) {
//           size_t idx = tensor.getIndex(b, c, h, w);
//           ASSERT_EQ(2, tensor.getValueQint4(idx));
//         }
//       }
//     }
//   }
// }

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

TEST(nntrainer_Tensor, multiply_i_03_n) {
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

TEST(nntrainer_Tensor, multiply_i_broadcast_01_p) {
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

  EXPECT_THROW({ input.multiply(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.multiply(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.multiply(test, output), std::invalid_argument);
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

  EXPECT_THROW({ input.divide(0.0); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.divide(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.divide(test, output), std::invalid_argument);
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

  EXPECT_THROW({ input.add(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, add_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.add(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.add(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.add(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.add(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.add(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, pow_01_p) {

  nntrainer::Tensor input = constant(4.0, 3, 2, 4, 5);

  nntrainer::Tensor actual, expected;

  actual = input.pow(0.5f);
  expected = constant(2.0, 3, 2, 4, 5);
  EXPECT_EQ(actual, expected);

  actual = input.pow(2.0f);
  expected = constant(16.0, 3, 2, 4, 5);
  EXPECT_EQ(actual, expected);

  actual = input.pow(-0.5f);
  expected = constant(0.5, 3, 2, 4, 5);
  EXPECT_EQ(actual, expected);
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
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.subtract(input);

  EXPECT_EQ(constant(0.0, batch, channel, height, width), result);
}

TEST(nntrainer_Tensor, subtract_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1);

  EXPECT_THROW({ input.subtract(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.subtract(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.subtract(test, output), std::invalid_argument);
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
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::Tensor ans0(
    std::vector<std::vector<std::vector<std::vector<float>>>>(
      {{{{39, 42, 45, 48, 51, 54, 57, 60, 63, 66},
         {69, 72, 75, 78, 81, 84, 87, 90, 93, 96}},
        {{57, 60, 63, 66, 69, 72, 75, 78, 81, 84},
         {87, 90, 93, 96, 99, 102, 105, 108, 111, 114}}}}),
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP32});

  nntrainer::Tensor ans1(
    std::vector<std::vector<std::vector<std::vector<float>>>>(
      {{{{8, 10, 12, 14, 16, 18, 20, 22, 24, 26},
         {28, 30, 32, 34, 36, 38, 40, 42, 44, 46}}},
       {{{32, 34, 36, 38, 40, 42, 44, 46, 48, 50},
         {52, 54, 56, 58, 60, 62, 64, 66, 68, 70}}},
       {{{56, 58, 60, 62, 64, 66, 68, 70, 72, 74},
         {76, 78, 80, 82, 84, 86, 88, 90, 92, 94}}}}),
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP32});

  nntrainer::Tensor ans2(
    std::vector<std::vector<std::vector<std::vector<float>>>>(
      {{{{12, 14, 16, 18, 20, 22, 24, 26, 28, 30}},
        {{24, 26, 28, 30, 32, 34, 36, 38, 40, 42}}},
       {{{36, 38, 40, 42, 44, 46, 48, 50, 52, 54}},
        {{48, 50, 52, 54, 56, 58, 60, 62, 64, 66}}},
       {{{60, 62, 64, 66, 68, 70, 72, 74, 76, 78}},
        {{72, 74, 76, 78, 80, 82, 84, 86, 88, 90}}}}),
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP32});

  nntrainer::Tensor ans3(
    std::vector<std::vector<std::vector<std::vector<float>>>>(
      {{{{55}, {155}}, {{115}, {215}}},
       {{{175}, {275}}, {{235}, {335}}},
       {{{295}, {395}}, {{355}, {455}}}}),
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP32});

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height * channel) + j * (batch * height) +
                          k * (width) + l + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result1 = input.sum(1);
  nntrainer::Tensor result2 = input.sum(2);
  nntrainer::Tensor result3 = input.sum(3);

  EXPECT_EQ(ans0, result0);
  EXPECT_EQ(ans1, result1);
  EXPECT_EQ(ans2, result2);
  EXPECT_EQ(ans3, result3);
}

TEST(nntrainer_Tensor, sum_03_p) {
  const int batch = 3;
  const int channel = 2;
  const int height = 1;
  const int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (height * channel * width) + j * (height * width) +
                          k * (width) + l + 1);
  // Test for alpha == 1 and beta == 0 and dimension of reduced axis == 1
  {
    nntrainer::Tensor ans_0_1_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{63, 66, 69, 72, 75, 78, 81, 84, 87, 90}},
          {{93, 96, 99, 102, 105, 108, 111, 114, 117, 120}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_1_1_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{12, 14, 16, 18, 20, 22, 24, 26, 28, 30}}},
         {{{52, 54, 56, 58, 60, 62, 64, 66, 68, 70}}},
         {{{92, 94, 96, 98, 100, 102, 104, 106, 108, 110}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_2_1_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
          {{11, 12, 13, 14, 15, 16, 17, 18, 19, 20}}},
         {{{21, 22, 23, 24, 25, 26, 27, 28, 29, 30}},
          {{31, 32, 33, 34, 35, 36, 37, 38, 39, 40}}},
         {{{41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
          {{51, 52, 53, 54, 55, 56, 57, 58, 59, 60}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_3_1_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{55}}, {{155}}}, {{{255}}, {{355}}}, {{{455}}, {{555}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor result_0_1_0 = input.sum(0, 1);
    nntrainer::Tensor result_1_1_0 = input.sum(1, 1);
    nntrainer::Tensor result_2_1_0 = input.sum(2, 1);
    nntrainer::Tensor result_3_1_0 = input.sum(3, 1);

    EXPECT_EQ(ans_0_1_0, result_0_1_0);
    EXPECT_EQ(ans_1_1_0, result_1_1_0);
    EXPECT_EQ(ans_2_1_0, result_2_1_0);
    EXPECT_EQ(ans_3_1_0, result_3_1_0);
  }

  // Test for alpha == 1 and beta == 2 and dimension of reduced axis == 1
  {
    nntrainer::Tensor ans_0_1_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{65, 70, 75, 80, 85, 90, 95, 100, 105, 110}},
          {{115, 120, 125, 130, 135, 140, 145, 150, 155, 160}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_1_1_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{14, 18, 22, 26, 30, 34, 38, 42, 46, 50}}},
         {{{74, 78, 82, 86, 90, 94, 98, 102, 106, 110}}},
         {{{134, 138, 142, 146, 150, 154, 158, 162, 166, 170}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_2_1_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{3, 6, 9, 12, 15, 18, 21, 24, 27, 30}},
          {{33, 36, 39, 42, 45, 48, 51, 54, 57, 60}}},
         {{{63, 66, 69, 72, 75, 78, 81, 84, 87, 90}},
          {{93, 96, 99, 102, 105, 108, 111, 114, 117, 120}}},
         {{{123, 126, 129, 132, 135, 138, 141, 144, 147, 150}},
          {{153, 156, 159, 162, 165, 168, 171, 174, 177, 180}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_3_1_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{57}}, {{159}}}, {{{261}}, {{363}}}, {{{465}}, {{567}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor output_0_1_2(1, channel, height, width);
    {
      const int batch = 1;
      GEN_TEST_INPUT(output_0_1_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_1_1_2(batch, 1, height, width);
    {
      const int channel = 1;
      GEN_TEST_INPUT(output_1_1_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_2_1_2(batch, channel, 1, width);
    {
      const int height = 1;
      GEN_TEST_INPUT(output_2_1_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_3_1_2(batch, channel, height, 1);
    {
      const int width = 1;
      GEN_TEST_INPUT(output_3_1_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor result_0_1_2 = input.sum(0, output_0_1_2, 1, 2);
    nntrainer::Tensor result_1_1_2 = input.sum(1, output_1_1_2, 1, 2);
    nntrainer::Tensor result_2_1_2 = input.sum(2, output_2_1_2, 1, 2);
    nntrainer::Tensor result_3_1_2 = input.sum(3, output_3_1_2, 1, 2);

    EXPECT_EQ(ans_0_1_2, result_0_1_2);
    EXPECT_EQ(ans_1_1_2, result_1_1_2);
    EXPECT_EQ(ans_2_1_2, result_2_1_2);
    EXPECT_EQ(ans_3_1_2, result_3_1_2);
  }

  // Test for alpha == 2 and beta == 0
  {
    nntrainer::Tensor ans_0_2_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{126, 132, 138, 144, 150, 156, 162, 168, 174, 180}},
          {{186, 192, 198, 204, 210, 216, 222, 228, 234, 240}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_1_2_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{24, 28, 32, 36, 40, 44, 48, 52, 56, 60}}},
         {{{104, 108, 112, 116, 120, 124, 128, 132, 136, 140}}},
         {{{184, 188, 192, 196, 200, 204, 208, 212, 216, 220}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_2_2_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}},
          {{22, 24, 26, 28, 30, 32, 34, 36, 38, 40}}},
         {{{42, 44, 46, 48, 50, 52, 54, 56, 58, 60}},
          {{62, 64, 66, 68, 70, 72, 74, 76, 78, 80}}},
         {{{82, 84, 86, 88, 90, 92, 94, 96, 98, 100}},
          {{102, 104, 106, 108, 110, 112, 114, 116, 118, 120}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_3_2_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{110}}, {{310}}}, {{{510}}, {{710}}}, {{{910}}, {{1110}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor result_0_2_0 = input.sum(0, 2);
    nntrainer::Tensor result_1_2_0 = input.sum(1, 2);
    nntrainer::Tensor result_2_2_0 = input.sum(2, 2);
    nntrainer::Tensor result_3_2_0 = input.sum(3, 2);

    EXPECT_EQ(ans_0_2_0, result_0_2_0);
    EXPECT_EQ(ans_1_2_0, result_1_2_0);
    EXPECT_EQ(ans_2_2_0, result_2_2_0);
    EXPECT_EQ(ans_3_2_0, result_3_2_0);
  }

  // Test for alpha == 2 and beta == 2
  {
    nntrainer::Tensor ans_0_2_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{128, 136, 144, 152, 160, 168, 176, 184, 192, 200}},
          {{208, 216, 224, 232, 240, 248, 256, 264, 272, 280}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_1_2_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{26, 32, 38, 44, 50, 56, 62, 68, 74, 80}}},
         {{{126, 132, 138, 144, 150, 156, 162, 168, 174, 180}}},
         {{{226, 232, 238, 244, 250, 256, 262, 268, 274, 280}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_2_2_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{4, 8, 12, 16, 20, 24, 28, 32, 36, 40}},
          {{44, 48, 52, 56, 60, 64, 68, 72, 76, 80}}},
         {{{84, 88, 92, 96, 100, 104, 108, 112, 116, 120}},
          {{124, 128, 132, 136, 140, 144, 148, 152, 156, 160}}},
         {{{164, 168, 172, 176, 180, 184, 188, 192, 196, 200}},
          {{204, 208, 212, 216, 220, 224, 228, 232, 236, 240}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor ans_3_2_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{112}}, {{314}}}, {{{516}}, {{718}}}, {{{920}}, {{1122}}}}),
      {ml::train::TensorDim::Format::NCHW,
       ml::train::TensorDim::DataType::FP32});

    nntrainer::Tensor output_0_2_2(1, channel, height, width);
    {
      const int batch = 1;
      GEN_TEST_INPUT(output_0_2_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_1_2_2(batch, 1, height, width);
    {
      const int channel = 1;
      GEN_TEST_INPUT(output_1_2_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_2_2_2(batch, channel, 1, width);
    {
      const int height = 1;
      GEN_TEST_INPUT(output_2_2_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_3_2_2(batch, channel, height, 1);
    {
      const int width = 1;
      GEN_TEST_INPUT(output_3_2_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor result_0_2_2 = input.sum(0, output_0_2_2, 2, 2);
    nntrainer::Tensor result_1_2_2 = input.sum(1, output_1_2_2, 2, 2);
    nntrainer::Tensor result_2_2_2 = input.sum(2, output_2_2_2, 2, 2);
    nntrainer::Tensor result_3_2_2 = input.sum(3, output_3_2_2, 2, 2);

    EXPECT_EQ(ans_0_2_2, result_0_2_2);
    EXPECT_EQ(ans_1_2_2, result_1_2_2);
    EXPECT_EQ(ans_2_2_2, result_2_2_2);
    EXPECT_EQ(ans_3_2_2, result_3_2_2);
  }
}

TEST(nntrainer_Tensor, sum_04_p) {
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
  EXPECT_THROW(t.sum(7), std::out_of_range);
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
  t = t.apply((std::function<float(float)>)[&](float in) { return idx++ % 2; });

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
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m), std::runtime_error);
}

TEST(nntrainer_Tensor, dot_02_n) {
  nntrainer::Tensor input(2, 3, 4, 5);
  nntrainer::Tensor m(1, 3, 4, 5);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m, true),
               std::runtime_error);
}

TEST(nntrainer_Tensor, dot_02_p) {
  nntrainer::Tensor input(2, 3, 4, 5);
  nntrainer::Tensor m(1, 3, 4, 5);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, false, true));
}

TEST(nntrainer_Tensor, dot_03_p) {
  nntrainer::Tensor input(1, 3, 4, 5);
  nntrainer::Tensor m(1, 3, 4, 5);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, true));
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

TEST(nntrainer_Tensor, dot_transpose_p) {
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
    float answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
    float answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
    float answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
    float answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
    float answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
    float answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
    float answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
    float answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
    float answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
    float answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
    float answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
    float answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
    float answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
    float answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
    float answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
    float answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
}

TEST(nntrainer_Tensor, dot_shortcuts_p) {
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
    float answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
    float answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
    float answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
    float answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
    float answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
    float answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
    float answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
    float answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
    float answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
    float answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
    float answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
    float answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
    float answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
    float answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
    float answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
    float answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2), answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
}

TEST(nntrainer_Tensor, transpose_p) {
  nntrainer::TensorDim ref_dim(3, 2, 4, 5);

  /// plain transpose
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    float answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
      98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119};
    nntrainer::Tensor answer({3, 2, 4, 5}, answer_data);
    nntrainer::Tensor m = t.transpose("0:1:2");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    float answer_data[] = {
      0,   5,   10,  15,  1,   6,   11,  16,  2,   7,   12,  17,  3,   8,
      13,  18,  4,   9,   14,  19,  20,  25,  30,  35,  21,  26,  31,  36,
      22,  27,  32,  37,  23,  28,  33,  38,  24,  29,  34,  39,  40,  45,
      50,  55,  41,  46,  51,  56,  42,  47,  52,  57,  43,  48,  53,  58,
      44,  49,  54,  59,  60,  65,  70,  75,  61,  66,  71,  76,  62,  67,
      72,  77,  63,  68,  73,  78,  64,  69,  74,  79,  80,  85,  90,  95,
      81,  86,  91,  96,  82,  87,  92,  97,  83,  88,  93,  98,  84,  89,
      94,  99,  100, 105, 110, 115, 101, 106, 111, 116, 102, 107, 112, 117,
      103, 108, 113, 118, 104, 109, 114, 119};
    nntrainer::Tensor answer({3, 2, 5, 4}, answer_data);
    nntrainer::Tensor m = t.transpose("0:2:1");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    float answer_data[] = {
      0,   1,   2,   3,   4,   20,  21,  22,  23,  24,  5,   6,   7,   8,
      9,   25,  26,  27,  28,  29,  10,  11,  12,  13,  14,  30,  31,  32,
      33,  34,  15,  16,  17,  18,  19,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  60,  61,  62,  63,  64,  45,  46,  47,  48,  49,  65,
      66,  67,  68,  69,  50,  51,  52,  53,  54,  70,  71,  72,  73,  74,
      55,  56,  57,  58,  59,  75,  76,  77,  78,  79,  80,  81,  82,  83,
      84,  100, 101, 102, 103, 104, 85,  86,  87,  88,  89,  105, 106, 107,
      108, 109, 90,  91,  92,  93,  94,  110, 111, 112, 113, 114, 95,  96,
      97,  98,  99,  115, 116, 117, 118, 119};
    nntrainer::Tensor answer({3, 4, 2, 5}, answer_data);
    nntrainer::Tensor m = t.transpose("1:0:2");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    float answer_data[] = {
      0,  20,  1,  21,  2,  22,  3,  23,  4,  24,  5,  25,  6,  26,  7,  27,
      8,  28,  9,  29,  10, 30,  11, 31,  12, 32,  13, 33,  14, 34,  15, 35,
      16, 36,  17, 37,  18, 38,  19, 39,  40, 60,  41, 61,  42, 62,  43, 63,
      44, 64,  45, 65,  46, 66,  47, 67,  48, 68,  49, 69,  50, 70,  51, 71,
      52, 72,  53, 73,  54, 74,  55, 75,  56, 76,  57, 77,  58, 78,  59, 79,
      80, 100, 81, 101, 82, 102, 83, 103, 84, 104, 85, 105, 86, 106, 87, 107,
      88, 108, 89, 109, 90, 110, 91, 111, 92, 112, 93, 113, 94, 114, 95, 115,
      96, 116, 97, 117, 98, 118, 99, 119};
    nntrainer::Tensor answer({3, 4, 5, 2}, answer_data);
    nntrainer::Tensor m = t.transpose("1:2:0");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    float answer_data[] = {
      0,  5,  10,  15,  20,  25,  30,  35, 1,  6,   11,  16,  21,  26,  31,
      36, 2,  7,   12,  17,  22,  27,  32, 37, 3,   8,   13,  18,  23,  28,
      33, 38, 4,   9,   14,  19,  24,  29, 34, 39,  40,  45,  50,  55,  60,
      65, 70, 75,  41,  46,  51,  56,  61, 66, 71,  76,  42,  47,  52,  57,
      62, 67, 72,  77,  43,  48,  53,  58, 63, 68,  73,  78,  44,  49,  54,
      59, 64, 69,  74,  79,  80,  85,  90, 95, 100, 105, 110, 115, 81,  86,
      91, 96, 101, 106, 111, 116, 82,  87, 92, 97,  102, 107, 112, 117, 83,
      88, 93, 98,  103, 108, 113, 118, 84, 89, 94,  99,  104, 109, 114, 119};
    nntrainer::Tensor answer({3, 5, 2, 4}, answer_data);
    nntrainer::Tensor m = t.transpose("2:0:1");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    float answer_data[] = {
      0,  20,  5,  25,  10, 30,  15, 35,  1,  21,  6,  26,  11, 31,  16, 36,
      2,  22,  7,  27,  12, 32,  17, 37,  3,  23,  8,  28,  13, 33,  18, 38,
      4,  24,  9,  29,  14, 34,  19, 39,  40, 60,  45, 65,  50, 70,  55, 75,
      41, 61,  46, 66,  51, 71,  56, 76,  42, 62,  47, 67,  52, 72,  57, 77,
      43, 63,  48, 68,  53, 73,  58, 78,  44, 64,  49, 69,  54, 74,  59, 79,
      80, 100, 85, 105, 90, 110, 95, 115, 81, 101, 86, 106, 91, 111, 96, 116,
      82, 102, 87, 107, 92, 112, 97, 117, 83, 103, 88, 108, 93, 113, 98, 118,
      84, 104, 89, 109, 94, 114, 99, 119};
    nntrainer::Tensor answer({3, 5, 4, 2}, answer_data);
    nntrainer::Tensor m = t.transpose("2:1:0");
    EXPECT_EQ(answer, m);
  }

  /// outplace transpose
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 2, 4, 5);
    float answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
      98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119};
    nntrainer::Tensor answer({3, 2, 4, 5}, answer_data);
    t.transpose("0:1:2", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 2, 5, 4);
    float answer_data[] = {
      0,   5,   10,  15,  1,   6,   11,  16,  2,   7,   12,  17,  3,   8,
      13,  18,  4,   9,   14,  19,  20,  25,  30,  35,  21,  26,  31,  36,
      22,  27,  32,  37,  23,  28,  33,  38,  24,  29,  34,  39,  40,  45,
      50,  55,  41,  46,  51,  56,  42,  47,  52,  57,  43,  48,  53,  58,
      44,  49,  54,  59,  60,  65,  70,  75,  61,  66,  71,  76,  62,  67,
      72,  77,  63,  68,  73,  78,  64,  69,  74,  79,  80,  85,  90,  95,
      81,  86,  91,  96,  82,  87,  92,  97,  83,  88,  93,  98,  84,  89,
      94,  99,  100, 105, 110, 115, 101, 106, 111, 116, 102, 107, 112, 117,
      103, 108, 113, 118, 104, 109, 114, 119};
    nntrainer::Tensor answer({3, 2, 5, 4}, answer_data);
    t.transpose("0:2:1", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 4, 2, 5);
    float answer_data[] = {
      0,   1,   2,   3,   4,   20,  21,  22,  23,  24,  5,   6,   7,   8,
      9,   25,  26,  27,  28,  29,  10,  11,  12,  13,  14,  30,  31,  32,
      33,  34,  15,  16,  17,  18,  19,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  60,  61,  62,  63,  64,  45,  46,  47,  48,  49,  65,
      66,  67,  68,  69,  50,  51,  52,  53,  54,  70,  71,  72,  73,  74,
      55,  56,  57,  58,  59,  75,  76,  77,  78,  79,  80,  81,  82,  83,
      84,  100, 101, 102, 103, 104, 85,  86,  87,  88,  89,  105, 106, 107,
      108, 109, 90,  91,  92,  93,  94,  110, 111, 112, 113, 114, 95,  96,
      97,  98,  99,  115, 116, 117, 118, 119};
    nntrainer::Tensor answer({3, 4, 2, 5}, answer_data);
    t.transpose("1:0:2", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 4, 5, 2);
    float answer_data[] = {
      0,  20,  1,  21,  2,  22,  3,  23,  4,  24,  5,  25,  6,  26,  7,  27,
      8,  28,  9,  29,  10, 30,  11, 31,  12, 32,  13, 33,  14, 34,  15, 35,
      16, 36,  17, 37,  18, 38,  19, 39,  40, 60,  41, 61,  42, 62,  43, 63,
      44, 64,  45, 65,  46, 66,  47, 67,  48, 68,  49, 69,  50, 70,  51, 71,
      52, 72,  53, 73,  54, 74,  55, 75,  56, 76,  57, 77,  58, 78,  59, 79,
      80, 100, 81, 101, 82, 102, 83, 103, 84, 104, 85, 105, 86, 106, 87, 107,
      88, 108, 89, 109, 90, 110, 91, 111, 92, 112, 93, 113, 94, 114, 95, 115,
      96, 116, 97, 117, 98, 118, 99, 119};
    nntrainer::Tensor answer({3, 4, 5, 2}, answer_data);
    t.transpose("1:2:0", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 5, 2, 4);
    float answer_data[] = {
      0,  5,  10,  15,  20,  25,  30,  35, 1,  6,   11,  16,  21,  26,  31,
      36, 2,  7,   12,  17,  22,  27,  32, 37, 3,   8,   13,  18,  23,  28,
      33, 38, 4,   9,   14,  19,  24,  29, 34, 39,  40,  45,  50,  55,  60,
      65, 70, 75,  41,  46,  51,  56,  61, 66, 71,  76,  42,  47,  52,  57,
      62, 67, 72,  77,  43,  48,  53,  58, 63, 68,  73,  78,  44,  49,  54,
      59, 64, 69,  74,  79,  80,  85,  90, 95, 100, 105, 110, 115, 81,  86,
      91, 96, 101, 106, 111, 116, 82,  87, 92, 97,  102, 107, 112, 117, 83,
      88, 93, 98,  103, 108, 113, 118, 84, 89, 94,  99,  104, 109, 114, 119};
    nntrainer::Tensor answer({3, 5, 2, 4}, answer_data);
    t.transpose("2:0:1", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    nntrainer::Tensor m = ranged(3, 5, 4, 2);
    float answer_data[] = {
      0,  20,  5,  25,  10, 30,  15, 35,  1,  21,  6,  26,  11, 31,  16, 36,
      2,  22,  7,  27,  12, 32,  17, 37,  3,  23,  8,  28,  13, 33,  18, 38,
      4,  24,  9,  29,  14, 34,  19, 39,  40, 60,  45, 65,  50, 70,  55, 75,
      41, 61,  46, 66,  51, 71,  56, 76,  42, 62,  47, 67,  52, 72,  57, 77,
      43, 63,  48, 68,  53, 73,  58, 78,  44, 64,  49, 69,  54, 74,  59, 79,
      80, 100, 85, 105, 90, 110, 95, 115, 81, 101, 86, 106, 91, 111, 96, 116,
      82, 102, 87, 107, 92, 112, 97, 117, 83, 103, 88, 108, 93, 113, 98, 118,
      84, 104, 89, 109, 94, 114, 99, 119};
    nntrainer::Tensor answer({3, 5, 4, 2}, answer_data);
    t.transpose("2:1:0", m);
    EXPECT_EQ(answer, m);
  }
}

TEST(nntrainer_Tensor, tranpose_dimension_not_match_n) {
  nntrainer::Tensor a(3, 2, 4, 5);
  nntrainer::Tensor b(3, 1, 2, 3);

  EXPECT_THROW(a.transpose("0:1:2", b), std::invalid_argument);
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

TEST(nntrainer_Tensor, copy_and_shares_variable_01_p) {
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

TEST(nntrainer_Tensor, copy_and_shares_variable_02_p) {
  nntrainer::Tensor A = constant(10, 3, 4, 5, 6, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::UINT16);
  nntrainer::Tensor B = A.clone();
  nntrainer::Tensor C = A;

  C.setValue(1, 1, 1, 1, 9);

  EXPECT_EQ(A, C);
  EXPECT_NE(B, C);

  C.reshape(nntrainer::TensorDim(3, 4, 6, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::UINT16));
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
  EXPECT_EQ(A_dim.getTensorDim(1), 100u);

  nntrainer::TensorDim A_dim_2 = A.getDim();
  EXPECT_EQ(A_dim_2.getTensorDim(1), 4u);
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

  std::stringstream ss, expected;
  ss << target;

  expected << '<' << typeid(target).name() << " at " << &target << ">\n"
           << "data addr: " << target.getData() << '\n'
           << "Shape: 3:1:2:3 [ FP32 : NCHW ]\n"
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
           << "data addr: " << target.getData() << '\n'
           << "Shape: 3:10:10:10 [ FP32 : NCHW ]\n"
           << "[1.2 1.2 1.2 ... 1.2 1.2 1.2]\n";
  ss << target;

  EXPECT_EQ(ss.str(), expected.str());
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

  for (unsigned int i = 0; i < ret1.size(); ++i) {
    EXPECT_FLOAT_EQ(data1[i], data2[i]);
  }
}

TEST(nntrainer_Tensor, fill_p) {
  /// same dimension, buffer size
  {
    nntrainer::Tensor target(3, 2, 4, 5);
    nntrainer::Tensor original = randUniform(3, 2, 4, 5, -1.0f, 1.0f);
    target.fill(original, false);

    EXPECT_EQ(target, original);
  }

  /// same dimension, buffer size is different (not tested)
  {
    /// there is no way to make non contiguous tensor publicily yet
    EXPECT_TRUE(true);
  }

  /// uninitialized with initialized flag is true
  {
    nntrainer::Tensor target;
    nntrainer::Tensor original = randUniform(3, 2, 4, 5, -1.0f, 1.0f);
    target.fill(original, true);

    EXPECT_EQ(target, original);
  }
}

TEST(nntrainer_Tensor, fill_uninitialized_n) {
  nntrainer::Tensor target;
  nntrainer::Tensor original = randUniform(3, 1, 2, 3, -1.0f, 1.0f);
  EXPECT_THROW(target.fill(original, false), std::invalid_argument);
}

TEST(nntrainer_Tensor, fill_different_dimension_n) {
  nntrainer::Tensor target(3, 1, 3, 2);
  nntrainer::Tensor original = randUniform(3, 1, 2, 3, -1.0f, 1.0f);
  EXPECT_THROW(target.fill(original, false), std::invalid_argument);
}

TEST(nntrainer_Tensor, DISABLED_fill_non_contiguous_n) {
  /// there is no way to make non contiguous tensor publicily yet
  EXPECT_TRUE(false);
}

TEST(nntrainer_Tensor, DISABLED_fill_different_buffer_size_n) {
  /// there is no way to make same dimension, diffrent buffersized tensor
  /// publicily yet
  EXPECT_TRUE(false);
}

TEST(nntrainer_Tensor, empty_01) {
  nntrainer::Tensor t;

  EXPECT_TRUE(t.empty());
}

TEST(nntrainer_Tensor, empty_02) {
  nntrainer::Tensor t({1, 2, 3, 4}, false);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, empty_03) {
  nntrainer::Tensor t({1, 2, 3, 4}, true);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, allocate_01_n) {
  nntrainer::Tensor t;
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_FALSE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_02_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, false);
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_03_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, true);
  EXPECT_TRUE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_04_p) {
  nntrainer::Tensor t(
    {1, 2, 3, 4, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8}},
    true);
  EXPECT_TRUE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_05_p) {
  nntrainer::Tensor t(
    {1, 2, 3, 4, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16}},
    true);
  EXPECT_TRUE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, initialize_01_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, true, nntrainer::Initializer::ONES);

  nntrainer::Tensor golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_02_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, true);

  nntrainer::Tensor golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_NE(golden, t);

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_03_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, false, nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_04_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, false);
  t.initialize(nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_05_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, false);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4);
  golden.setValue(1.f);

  /**
   * Ideally, it should be NE, but it can be equal due to no initialization
   * EXPECT_NE(golden, t);
   */

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_06_n) {
  nntrainer::Tensor t({1, 2, 3, 4}, true, nntrainer::Initializer::ONES);
  nntrainer::Tensor golden({1, 2, 3, 4}, true, nntrainer::Initializer::ZEROS);

  EXPECT_NE(golden, t);

  golden.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_07_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, true, nntrainer::Initializer::ONES);

  nntrainer::Tensor golden(1, 2, 3, 4);
  golden.setValue(1);

  EXPECT_EQ(golden, t);

  t.setValue(0, 0, 0, 0, 0);
  t.setValue(0, 0, 0, t.size() - 1, 0);
  EXPECT_NE(golden, t);

  t.initialize();
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_08_p) {
  nntrainer::Tensor t({1, 2, 3, 4}, true, nntrainer::Initializer::ONES);

  nntrainer::Tensor golden(1, 2, 3, 4);
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

TEST(nntrainer_Tensor, initialize_09_p) {
  nntrainer::Tensor t(
    {1, 2, 3, 4, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8}}, true,
    nntrainer::Initializer::ONES);
  nntrainer::Tensor golden(
    {1, 2, 3, 4, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8}}, true,
    nntrainer::Initializer::ZEROS);
  EXPECT_NE(golden, t);
  golden.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_10_p) {
  nntrainer::Tensor t(
    {1, 2, 3, 4, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16}},
    true, nntrainer::Initializer::ONES);
  nntrainer::Tensor golden(
    {1, 2, 3, 4, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16}},
    true, nntrainer::Initializer::ZEROS);
  EXPECT_NE(golden, t);
  golden.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_11_n) {
  nntrainer::Tensor t(
    {1, 2, 3, 4, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16}},
    true);

  /// @note ShortTensor does not support HE_NORMAL initialization
  EXPECT_THROW(t.initialize(nntrainer::Initializer::HE_NORMAL),
               std::invalid_argument);
}

TEST(nntrainer_Tensor, split_01_p) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                             30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 2, 4, 5}, answer_data));
    }
    {
      float answer_data[] = {40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                             70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 2, 4, 5}, answer_data));
    }
    {
      float answer_data[] = {80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                             90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 2, 4, 5}, answer_data));
    }
    EXPECT_EQ(t.split(3, 0), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                             56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87,
                             88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 1, 4, 5}, answer_data));
    }
    {
      float answer_data[] = {20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                             30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                             60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                             70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 1, 4, 5}, answer_data));
    }
    EXPECT_EQ(t.split(2, 1), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {
        0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   20,  21,  22,  23,  24,
        25, 26, 27, 28, 29, 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
        60, 61, 62, 63, 64, 65,  66,  67,  68,  69,  80,  81,  82,  83,  84,
        85, 86, 87, 88, 89, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 2, 5}, answer_data));
    }
    {
      float answer_data[] = {
        10, 11, 12, 13, 14, 15,  16,  17,  18,  19,  30,  31,  32,  33,  34,
        35, 36, 37, 38, 39, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
        70, 71, 72, 73, 74, 75,  76,  77,  78,  79,  90,  91,  92,  93,  94,
        95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 2, 5}, answer_data));
    }
    EXPECT_EQ(t.split(2, 2), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {0,  5,  10, 15, 20,  25,  30,  35,
                             40, 45, 50, 55, 60,  65,  70,  75,
                             80, 85, 90, 95, 100, 105, 110, 115};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 1}, answer_data));
    }
    {
      float answer_data[] = {1,  6,  11, 16, 21,  26,  31,  36,
                             41, 46, 51, 56, 61,  66,  71,  76,
                             81, 86, 91, 96, 101, 106, 111, 116};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 1}, answer_data));
    }
    {
      float answer_data[] = {2,  7,  12, 17, 22,  27,  32,  37,
                             42, 47, 52, 57, 62,  67,  72,  77,
                             82, 87, 92, 97, 102, 107, 112, 117};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 1}, answer_data));
    }
    {
      float answer_data[] = {3,  8,  13, 18, 23,  28,  33,  38,
                             43, 48, 53, 58, 63,  68,  73,  78,
                             83, 88, 93, 98, 103, 108, 113, 118};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 1}, answer_data));
    }
    {
      float answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 1}, answer_data));
    }
    EXPECT_EQ(t.split(5, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(1, 1, 4, 6);
    nntrainer::Tensor t = ranged(1, 1, 4, 6);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 1, 4, 3}, answer_data));
    }
    {
      float answer_data[] = {3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 1, 4, 3}, answer_data));
    }
    EXPECT_EQ(t.split(2, 3), answer);
  }
}

TEST(nntrainer_Tensor, split_02_n) {
  nntrainer::Tensor t(1, 1, 1, 1);
  EXPECT_THROW(t.split(0, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_03_n) {
  nntrainer::Tensor t(3, 1, 1, 1);
  EXPECT_THROW(t.split(2, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_04_p) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{2, 2, 4, 5}, answer_data));
    }
    {
      float answer_data[] = {80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                             90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 2, 4, 5}, answer_data));
    }
    EXPECT_EQ(t.split({2, 1}, 0), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                             56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87,
                             88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 1, 4, 5}, answer_data));
    }
    {
      float answer_data[] = {20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                             30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                             60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                             70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 1, 4, 5}, answer_data));
    }
    EXPECT_EQ(t.split({1, 1}, 1), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {
        0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   20,  21,  22,  23,  24,
        25, 26, 27, 28, 29, 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
        60, 61, 62, 63, 64, 65,  66,  67,  68,  69,  80,  81,  82,  83,  84,
        85, 86, 87, 88, 89, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 2, 5}, answer_data));
    }
    {
      float answer_data[] = {
        10, 11, 12, 13, 14, 15,  16,  17,  18,  19,  30,  31,  32,  33,  34,
        35, 36, 37, 38, 39, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
        70, 71, 72, 73, 74, 75,  76,  77,  78,  79,  90,  91,  92,  93,  94,
        95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 2, 5}, answer_data));
    }
    EXPECT_EQ(t.split({2, 2}, 2), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {0,  5,  10, 15, 20,  25,  30,  35,
                             40, 45, 50, 55, 60,  65,  70,  75,
                             80, 85, 90, 95, 100, 105, 110, 115};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 1}, answer_data));
    }
    {
      float answer_data[] = {
        1,   2,   3,   6,   7,   8,   11,  12,  13,  16,  17,  18, 21, 22, 23,
        26,  27,  28,  31,  32,  33,  36,  37,  38,  41,  42,  43, 46, 47, 48,
        51,  52,  53,  56,  57,  58,  61,  62,  63,  66,  67,  68, 71, 72, 73,
        76,  77,  78,  81,  82,  83,  86,  87,  88,  91,  92,  93, 96, 97, 98,
        101, 102, 103, 106, 107, 108, 111, 112, 113, 116, 117, 118};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 3}, answer_data));
    }
    {
      float answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 1}, answer_data));
    }
    EXPECT_EQ(t.split({1, 3, 1}, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {
        0,  1,  5,  6,  10, 11, 15, 16, 20,  21,  25,  26,  30,  31,  35,  36,
        40, 41, 45, 46, 50, 51, 55, 56, 60,  61,  65,  66,  70,  71,  75,  76,
        80, 81, 85, 86, 90, 91, 95, 96, 100, 101, 105, 106, 110, 111, 115, 116};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 2}, answer_data));
    }
    {
      float answer_data[] = {
        2,  3,  7,  8,  12, 13, 17, 18, 22,  23,  27,  28,  32,  33,  37,  38,
        42, 43, 47, 48, 52, 53, 57, 58, 62,  63,  67,  68,  72,  73,  77,  78,
        82, 83, 87, 88, 92, 93, 97, 98, 102, 103, 107, 108, 112, 113, 117, 118};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 2}, answer_data));
    }
    {
      float answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 1}, answer_data));
    }
    EXPECT_EQ(t.split({2, 2, 1}, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5);
    nntrainer::Tensor t = ranged(3, 2, 4, 5);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {
        0,  1,  5,  6,  10, 11, 15, 16, 20,  21,  25,  26,  30,  31,  35,  36,
        40, 41, 45, 46, 50, 51, 55, 56, 60,  61,  65,  66,  70,  71,  75,  76,
        80, 81, 85, 86, 90, 91, 95, 96, 100, 101, 105, 106, 110, 111, 115, 116};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 2}, answer_data));
    }
    {
      float answer_data[] = {
        2,   3,   4,   7,   8,   9,   12,  13,  14,  17,  18,  19, 22, 23, 24,
        27,  28,  29,  32,  33,  34,  37,  38,  39,  42,  43,  44, 47, 48, 49,
        52,  53,  54,  57,  58,  59,  62,  63,  64,  67,  68,  69, 72, 73, 74,
        77,  78,  79,  82,  83,  84,  87,  88,  89,  92,  93,  94, 97, 98, 99,
        102, 103, 104, 107, 108, 109, 112, 113, 114, 117, 118, 119};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{3, 2, 4, 3}, answer_data));
    }
    EXPECT_EQ(t.split({2, 3}, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(1, 1, 4, 6);
    nntrainer::Tensor t = ranged(1, 1, 4, 6);
    std::vector<nntrainer::Tensor> answer;
    {
      float answer_data[] = {0, 6, 12, 18};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 1, 4, 1}, answer_data));
    }
    {
      float answer_data[] = {1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 1, 4, 3}, answer_data));
    }
    {
      float answer_data[] = {4, 5, 10, 11, 16, 17, 22, 23};
      answer.push_back(
        nntrainer::Tensor(ml::train::TensorDim{1, 1, 4, 2}, answer_data));
    }
    EXPECT_EQ(t.split({1, 3, 2}, 3), answer);
  }
}

TEST(nntrainer_Tensor, split_05_n) {
  nntrainer::Tensor t(3, 1, 1, 1);
  EXPECT_THROW(t.split({1, 1}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_06_n) {
  nntrainer::Tensor t(3, 1, 1, 1);
  EXPECT_THROW(t.split({2, 0, 1}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_07_n) {
  nntrainer::Tensor t(3, 1, 1, 1);
  EXPECT_THROW(t.split({}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, cat_01_p) {
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(ranged(2, 1, 1, 2));
    inputs.emplace_back(ranged(2, 2, 1, 2));
    float answer_data[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 6, 7};
    nntrainer::Tensor answer(ml::train::TensorDim{2, 3, 1, 2}, answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 1), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(ranged(3, 2, 4, 5));
    inputs.emplace_back(ranged(2, 2, 4, 5));
    float answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
      45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
      75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
      90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
      105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
      45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
      75,  76,  77,  78,  79};
    nntrainer::Tensor answer(ml::train::TensorDim{5, 2, 4, 5}, answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 0), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(ranged(3, 3, 4, 5));
    inputs.emplace_back(ranged(3, 2, 4, 5));
    float answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
      10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
      24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,
      38,  39,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
      72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,
      86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
      114, 115, 116, 117, 118, 119, 40,  41,  42,  43,  44,  45,  46,  47,
      48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
      62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
      76,  77,  78,  79,  120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
      130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
      144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
      158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
      172, 173, 174, 175, 176, 177, 178, 179, 80,  81,  82,  83,  84,  85,
      86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
      114, 115, 116, 117, 118, 119};
    nntrainer::Tensor answer(ml::train::TensorDim{3, 5, 4, 5}, answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 1), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(ranged(3, 2, 1, 5));
    inputs.emplace_back(ranged(3, 2, 2, 5));
    float answer_data[] = {
      0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,
      8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44, 45, 46,
      47, 48, 49, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
    nntrainer::Tensor answer(ml::train::TensorDim{3, 2, 3, 5}, answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 2), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(3);
    inputs.emplace_back(ranged(3, 2, 4, 1));
    inputs.emplace_back(ranged(3, 2, 4, 3));
    inputs.emplace_back(ranged(3, 2, 4, 2));
    float answer_data[] = {
      0,  0,  1,  2,  0,  1,  1,  3,  4,  5,  2,  3,  2,  6,  7,  8,  4,  5,
      3,  9,  10, 11, 6,  7,  4,  12, 13, 14, 8,  9,  5,  15, 16, 17, 10, 11,
      6,  18, 19, 20, 12, 13, 7,  21, 22, 23, 14, 15, 8,  24, 25, 26, 16, 17,
      9,  27, 28, 29, 18, 19, 10, 30, 31, 32, 20, 21, 11, 33, 34, 35, 22, 23,
      12, 36, 37, 38, 24, 25, 13, 39, 40, 41, 26, 27, 14, 42, 43, 44, 28, 29,
      15, 45, 46, 47, 30, 31, 16, 48, 49, 50, 32, 33, 17, 51, 52, 53, 34, 35,
      18, 54, 55, 56, 36, 37, 19, 57, 58, 59, 38, 39, 20, 60, 61, 62, 40, 41,
      21, 63, 64, 65, 42, 43, 22, 66, 67, 68, 44, 45, 23, 69, 70, 71, 46, 47};
    nntrainer::Tensor answer(ml::train::TensorDim{3, 2, 4, 6}, answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, -1), answer);
  }
}

TEST(nntrainer_Tensor, cat_02_n) {
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(nntrainer::Tensor(2, 1, 1, 2));
    inputs.emplace_back(nntrainer::Tensor(2, 2, 1, 2));
    EXPECT_THROW(nntrainer::Tensor::cat(inputs, 2), std::invalid_argument);
  }
}

// concatenate an empty list of tensors
TEST(nntrainer_Tensor, cat_03_n) {
  std::vector<nntrainer::Tensor> inputs;
  EXPECT_THROW(nntrainer::Tensor::cat(inputs, 0), std::invalid_argument);
}

// concatenate a single tensor
TEST(nntrainer_Tensor, cat_04_n) {
  std::vector<nntrainer::Tensor> inputs;
  inputs.reserve(1);
  inputs.emplace_back(nntrainer::Tensor(2, 1, 1, 2));
  EXPECT_THROW(nntrainer::Tensor::cat(inputs, 0), std::invalid_argument);
}

// concatenate tensors with different data types
TEST(nntrainer_Tensor, cat_05_n) {
  std::vector<nntrainer::Tensor> inputs;
  inputs.reserve(2);
  inputs.emplace_back(nntrainer::Tensor(2, 1, 1, 2));
  inputs.emplace_back(nntrainer::Tensor(
    2, 1, 1, 2, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8}));
  EXPECT_THROW(nntrainer::Tensor::cat(inputs, 0), std::invalid_argument);
}

// incorrect output tensor dimension
TEST(nntrainer_Tensor, cat_06_n) {
  std::vector<nntrainer::Tensor> inputs;
  inputs.reserve(2);
  inputs.emplace_back(nntrainer::Tensor(3, 2, 4, 1));
  inputs.emplace_back(nntrainer::Tensor(3, 2, 4, 3));
  nntrainer::Tensor output(3, 2, 4, 5);
  EXPECT_THROW(nntrainer::Tensor::cat(inputs, 3, output),
               std::invalid_argument);
}

// tensors not having the same shape except for the axis
TEST(nntrainer_Tensor, cat_07_n) {
  std::vector<nntrainer::Tensor> inputs;
  inputs.reserve(2);
  inputs.emplace_back(nntrainer::Tensor(3, 2, 4, 1));
  inputs.emplace_back(nntrainer::Tensor(3, 1, 4, 3));
  EXPECT_THROW(nntrainer::Tensor::cat(inputs, 1), std::invalid_argument);
  EXPECT_THROW(nntrainer::Tensor::cat(inputs, 3), std::invalid_argument);
}

TEST(nntrainer_Tensor, zoneout_mask_01_n) {
  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 10, 10);
  nntrainer::Tensor opposite(20, 20, 20, 20);
  EXPECT_THROW(t.zoneout_mask(opposite, zoneout_rate), std::invalid_argument);
}

TEST(nntrainer_Tensor, zoneout_mask_02_p) {
  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 10, 10);
  nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
  constexpr float epsilon = 1e-3;

  EXPECT_EQ(t.size(), opposite.size());

  auto is_near = [epsilon](float val1, float val2) {
    return val2 - epsilon < val1 && val1 < val2 + epsilon;
  };

  for (unsigned int i = 0; i < opposite.size(); ++i) {
    if (is_near(opposite.getValue(i), 0.0f)) {
      EXPECT_NEAR(t.getValue(i), 1.0f, epsilon);
    } else if (is_near(opposite.getValue(i), 1.0f)) {
      EXPECT_NEAR(t.getValue(i), 0.0f, epsilon);
    } else {
      FAIL() << "This should not be happen";
    }
  }
}

TEST(nntrainer_Tensor, zoneout_mask_03_p) {
  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 100, 100);
  nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
  constexpr float epsilon = 1e-3;

  auto is_near = [epsilon](float val1, float val2) {
    return val2 - epsilon < val1 && val1 < val2 + epsilon;
  };
  auto percentage = [](unsigned int dividend, unsigned int divisor) {
    return (float)dividend / (float)divisor;
  };

  {
    unsigned int zeros = 0;
    unsigned int ones = 0;
    for (unsigned int i = 0; i < opposite.size(); ++i) {
      if (is_near(opposite.getValue(i), 0.0f)) {
        ++zeros;
      } else if (is_near(opposite.getValue(i), 1.0f)) {
        ++ones;
      } else {
        FAIL() << "This should not be happen";
      }
    }
    EXPECT_NEAR(percentage(zeros, opposite.size()), 1.0f - zoneout_rate,
                epsilon);

    // main test
    EXPECT_NEAR(percentage(ones, opposite.size()), zoneout_rate, epsilon);
  }

  {
    unsigned int zeros = 0;
    unsigned int ones = 0;
    for (unsigned int i = 0; i < t.size(); ++i) {
      if (is_near(t.getValue(i), 0.0f)) {
        ++zeros;
      } else if (is_near(t.getValue(i), 1.0f)) {
        ++ones;
      } else {
        FAIL() << "This should not be happen";
      }
    }
    EXPECT_NEAR(percentage(zeros, t.size()), zoneout_rate, epsilon);

    // main test
    EXPECT_NEAR(percentage(ones, t.size()), 1.0f - zoneout_rate, epsilon);
  }
}

TEST(nntrainer_Tensor, zoneout_mask_04_n) {
  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 100, 100);
  nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
  constexpr float epsilon = 1e-3;

  auto is_near = [epsilon](float val1, float val2) {
    return val2 - epsilon < val1 && val1 < val2 + epsilon;
  };
  auto percentage = [](unsigned int dividend, unsigned int divisor) {
    return (float)dividend / (float)divisor;
  };

  {
    unsigned int zeros = 0;
    unsigned int ones = 0;
    for (unsigned int i = 0; i < opposite.size(); ++i) {
      if (is_near(opposite.getValue(i), 0.0f)) {
        ++zeros;
      } else if (is_near(opposite.getValue(i), 1.0f)) {
        ++ones;
      } else {
        FAIL() << "This should not be happen";
      }
    }
    EXPECT_FALSE(
      is_near(percentage(ones, opposite.size()), 1.0f - zoneout_rate));
  }

  {
    unsigned int zeros = 0;
    unsigned int ones = 0;
    for (unsigned int i = 0; i < t.size(); ++i) {
      if (is_near(t.getValue(i), 0.0f)) {
        ++zeros;
      } else if (is_near(t.getValue(i), 1.0f)) {
        ++ones;
      } else {
        FAIL() << "This should not be happen";
      }
    }
    EXPECT_FALSE(is_near(percentage(ones, t.size()), zoneout_rate));
  }
}

TEST(nntrainer_Tensor, TensorMap_p) {
  float dat[] = {1, 2, 3};

  {
    nntrainer::Tensor a = nntrainer::Tensor::Map(dat, 3 * sizeof(float), {3});
    /// check if a.getData() has same address with dat
    EXPECT_EQ(dat, a.getData());
    {
      /// check if b.getData() has same address with data
      nntrainer::Tensor b = a;
      EXPECT_EQ(dat, b.getData());
    }
  }
  /// check if dat is accessible after destruction of all the tensor
  EXPECT_FLOAT_EQ(dat[2], 3);
}

TEST(nntrainer_Tensor, TensorWrap_01_n) {
  float dat[] = {1, 2, 3};
  EXPECT_THROW(nntrainer::Tensor::Map(dat, 3, nntrainer::TensorDim({})),
               std::invalid_argument);
}

TEST(nntrainer_Tensor, TensorWrap_02_n) {
  float dat[] = {1, 2, 3};
  EXPECT_THROW(nntrainer::Tensor::Map(dat, 3, {4}), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_strided_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor result = input.add_strided(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  float *outdata = new float[(input.size())];

  EXPECT_EQ(result.getFormat(), input.getFormat());

  std::transform(indata, indata + batch * height * width * channel, indata,
                 outdata, std::plus<float>());

  for (int i = 0; i < batch * height * width * channel; ++i) {
    if (data[i] != outdata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }
  delete[] outdata;

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_strided_02_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1, channel);

  EXPECT_THROW({ input.add_strided(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, add_strided_03_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, height, width, channel);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.add_strided(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_strided_04_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, height, width, channel);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (height * width * channel) + j * (width * channel) +
                         k * channel + 2);

  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.add_strided(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_strided_05_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor result = input.add_strided(input, 10.0);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  float *indata_beta = new float[(input.size())];
  float *outdata = new float[(input.size())];

  EXPECT_EQ(result.getFormat(), input.getFormat());

  std::transform(
    indata, indata + batch * height * width * channel, indata_beta,
    std::bind(std::multiplies<float>(), std::placeholders::_1, 10.0));

  std::transform(indata, indata + batch * height * width * channel, indata_beta,
                 outdata, std::plus<float>());

  for (int i = 0; i < batch * height * width * channel; ++i) {
    if (data[i] != outdata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }
  delete[] indata_beta;
  delete[] outdata;

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_strided_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.multiply_strided(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  float *outdata = new float[(input.size())];

  std::transform(indata, indata + batch * height * width * channel, indata,
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

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1);

  EXPECT_THROW({ input.multiply_strided(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);
  // input is not allocated now : alloc_now == false
  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.multiply_strided(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  // test is not allocated.
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.multiply_strided(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);
  // output is not aloocated
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.multiply_strided(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_06_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;
  const int size = 90;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor output(batch, channel, height, width);
  GEN_TEST_INPUT(output, i * (batch * height) + j * (width) + k + 1);

  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  float outdata_beta[size];
  float indata_mul[size];
  float outdata[size];

  std::transform(
    indata, indata + batch * height * width * channel, outdata_beta,
    std::bind(std::multiplies<float>(), std::placeholders::_1, 10.0));

  std::transform(indata, indata + batch * height * width * channel, indata,
                 indata_mul, std::multiplies<float>());
  std::transform(indata_mul, indata_mul + batch * height * width * channel,
                 outdata_beta, outdata, std::plus<float>());

  input.multiply_strided(input, output, 10.0);

  float *data = output.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != outdata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

// /**
//  * @brief dequantize FP32 tensor
//  */
// TEST(nntrainer_Tensor, dequantize_01_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);
//   input.setScaleFactors({1.5, 1.0, 0.5});
//   input.setZeroPoints({1, 4, 7});

//   nntrainer::Tensor output(batch, channel, height, width);

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

// /**
//  * @brief dequantize tensor with different dimension
//  */
// TEST(nntrainer_Tensor, dequantize_02_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch + 1, channel, height + 1, width + 1,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);
//   input.setScaleFactors({1.5, 1.0, 0.5});
//   input.setZeroPoints({1, 4, 7});

//   nntrainer::Tensor output(batch, channel, height, width);

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

// /**
//  * @brief dequantize tensor with no scale factors
//  */
// TEST(nntrainer_Tensor, dequantize_03_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   nntrainer::Tensor output(batch, channel, height, width);

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

// /**
//  * @brief dequantize tensor with incorrect number of scale factors
//  */
// TEST(nntrainer_Tensor, dequantize_04_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   nntrainer::Tensor output(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

//   input.setScaleFactors({2.0, 1.5, 1.0, 0.5});
//   input.setZeroPoints({2, 3, 4, 5});
//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
//   EXPECT_NO_THROW({ input.dequantize(output, 2); });
// }

// /**
//  * @brief dequantize tensor to QINT8
//  */
// TEST(nntrainer_Tensor, dequantize_05_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);
//   input.setScaleFactors({1.5, 1.0, 0.5});
//   input.setZeroPoints({1, 4, 7});

//   nntrainer::Tensor output(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

TEST(nntrainer_Tensor, sin_contiguous_p) {
  int batch = 1;
  int channel = 1;
  int height = 1440;
  int width = 1440;

  const int MOD = 10;

  const float eps = 1e-6;

  nntrainer::Tensor input(batch, channel, height, width);
  nntrainer::Tensor sin_output(batch, channel, height, width);

  GEN_TEST_INPUT(input, (i * (channel * width * height) + j * (height * width) +
                         k * (width) + l + 1) %
                          MOD);

  nntrainer::Tensor result_sine(batch, channel, height, width);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          result_sine.setValue(b, c, h, w,
                               std::sin(input.getValue(b, c, h, w)));
        }
      }
    }
  }

  input.sin(sin_output);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          EXPECT_NEAR(sin_output.getValue(b, c, h, w),
                      result_sine.getValue(b, c, h, w), eps);
        }
      }
    }
  }
}

TEST(nntrainer_Tensor, cos_contiguous_p) {
  int batch = 1;
  int channel = 1;
  int height = 1440;
  int width = 1440;

  nntrainer::Tensor input(batch, channel, height, width);
  nntrainer::Tensor cos_output(batch, channel, height, width);

  const int MOD = 10;
  const float eps = 1e-6;

  GEN_TEST_INPUT(input, (i * (channel * width * height) + j * (height * width) +
                         k * (width) + l + 1) %
                          MOD);

  nntrainer::Tensor result_cosine(batch, channel, height, width);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          result_cosine.setValue(b, c, h, w,
                                 std::cos(input.getValue(b, c, h, w)));
        }
      }
    }
  }

  input.cos(cos_output);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          EXPECT_NEAR(cos_output.getValue(b, c, h, w),
                      result_cosine.getValue(b, c, h, w), eps);
        }
      }
    }
  }
}

TEST(nntrainer_Tensor, cos_uncontiguous_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);
  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_output(batch, channel, height, width);
  nntrainer::Tensor ground_truth(batch, channel, height, width);

  const int MOD = 10;
  const float eps = 1e-5;

  GEN_TEST_INPUT(input, (i * (channel * width * height) + j * (height * width) +
                         k * (width) + l + 1) %
                          MOD);

  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  ground_truth.copy_with_stride(shared_input);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          ground_truth.setValue(b, c, h, w,
                                std::cos(ground_truth.getValue(b, c, h, w)));
        }
      }
    }
  }

  shared_input.cos(shared_output);

  EXPECT_EQ(shared_output, ground_truth);
}

TEST(nntrainer_Tensor, sin_uncontiguous_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);
  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_output(batch, channel, height, width);
  nntrainer::Tensor ground_truth(batch, channel, height, width);

  const int MOD = 10;
  const float eps = 1e-5;

  GEN_TEST_INPUT(input, (i * (channel * width * height) + j * (height * width) +
                         k * (width) + l + 1) %
                          MOD);

  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);

  ground_truth.copy_with_stride(shared_input);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          ground_truth.setValue(b, c, h, w,
                                std::sin(ground_truth.getValue(b, c, h, w)));
        }
      }
    }
  }

  shared_input.sin(shared_output);

  EXPECT_EQ(shared_output, ground_truth);
}

TEST(nntrainer_Tensor, sin_unmatched_dim_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor output(batch, channel, height, width);

  const int MOD = 10;

  GEN_TEST_INPUT(input, (i * (channel * width * height) + j * (height * width) +
                         k * (width) + l + 1) %
                          MOD);

  EXPECT_THROW({ input.sin(output); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, inv_sqrt_i_uncontiguous_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);
  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor ground_truth(batch, channel, height, width);

  const int MOD = 10;

  GEN_TEST_INPUT(input, (i * (channel * width * height) + j * (height * width) +
                         k * (width) + l + 1) %
                            MOD +
                          1);

  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  ground_truth.copy_with_stride(shared_input);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          ground_truth.setValue(
            b, c, h, w, 1 / std::sqrt(ground_truth.getValue(b, c, h, w)));
        }
      }
    }
  }

  shared_input.inv_sqrt_i();

  const float eps = 1e-5;

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          EXPECT_NEAR(shared_input.getValue(b, c, h, w),
                      ground_truth.getValue(b, c, h, w), eps);
        }
      }
    }
  }
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
