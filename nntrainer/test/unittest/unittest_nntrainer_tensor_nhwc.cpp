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

#define NHWC_ nntrainer::Tformat::NHWC
#define FP32_ nntrainer::Tdatatype::FP32

TEST(nntrainer_TensorDim, ctor_initializer_nhwc_p) {
  unsigned int b = 3;
  unsigned int c = 2;
  unsigned int h = 4;
  unsigned int w = 5;

  nntrainer::TensorDim t = {c};
  EXPECT_EQ(nntrainer::TensorDim(1, 1, 1, c), t);
  EXPECT_EQ(nntrainer::TensorDim(c), t);

  t = {w, c};
  EXPECT_EQ(nntrainer::TensorDim(1, 1, w, c), t);
  EXPECT_EQ(nntrainer::TensorDim(w, c), t);

  t = {h, w, c};
  EXPECT_EQ(nntrainer::TensorDim(1, h, w, c), t);
  EXPECT_EQ(nntrainer::TensorDim(h, w, c), t);

  t = {b, h, w, c};
  EXPECT_EQ(nntrainer::TensorDim(b, h, w, c), t);
}

TEST(nntrainer_TensorDim, default_constructor_with_tensor_type_nhwc_p) {
  unsigned int b = 3;
  unsigned int c = 2;
  unsigned int h = 4;
  unsigned int w = 5;

  nntrainer::TensorDim::TensorType tensor_type = {NHWC_, FP32_};

  nntrainer::TensorDim t = {c, tensor_type};
  EXPECT_EQ(nntrainer::TensorDim(1, 1, 1, c, tensor_type), t);
  EXPECT_EQ(nntrainer::TensorDim(c, tensor_type), t);

  t = {w, c, tensor_type};
  EXPECT_EQ(nntrainer::TensorDim(1, 1, w, c, tensor_type), t);
  EXPECT_EQ(nntrainer::TensorDim(w, c, tensor_type), t);

  t = {h, w, c, tensor_type};
  EXPECT_EQ(nntrainer::TensorDim(1, h, w, c, tensor_type), t);
  EXPECT_EQ(nntrainer::TensorDim(h, w, c, tensor_type), t);

  t = {b, h, w, c, tensor_type};
  EXPECT_EQ(nntrainer::TensorDim(b, h, w, c, tensor_type), t);
}

TEST(nntrianer_TensorDim, effective_dimension_nhwc_p) {
  nntrainer::TensorDim t(3, 2, 4, 5, NHWC_, FP32_);
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

TEST(nntrainer_TensorDim, ctor_initializer_nhwc_n) {
  EXPECT_THROW(nntrainer::TensorDim t({1, 2, 3, 4, 5}, NHWC_, FP32_),
               std::invalid_argument);
}

TEST(nntrainer_TensorDim, setTensorDim_01_nhwc_p) {
  int status = ML_ERROR_NONE;

  nntrainer::TensorDim tensor_dim;
  status = tensor_dim.setTensorDim("1:2:3:4", {NHWC_, FP32_});
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_TensorDim, setTensorDim_02__nhwc_n) {
  int status = ML_ERROR_NONE;

  nntrainer::TensorDim tensor_dim;
  status = tensor_dim.setTensorDim("1:2:3:4:5", {NHWC_, FP32_});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_TensorDim, setTensorDim_03_nhwc_n) {
  nntrainer::TensorDim d(NHWC_, FP32_);

  EXPECT_THROW(d.setTensorDim(0, 0), std::invalid_argument);
  EXPECT_THROW(d.setTensorDim(1, 0), std::invalid_argument);
  EXPECT_THROW(d.setTensorDim(2, 0), std::invalid_argument);
  EXPECT_THROW(d.setTensorDim(3, 0), std::invalid_argument);
}

TEST(nntrainer_TensorDim, setTensorDim_04_nhwc_p) {
  nntrainer::TensorDim d(NHWC_, FP32_);

  d.setTensorDim(0, 4);
  d.setTensorDim(1, 5);
  d.setTensorDim(2, 6);
  d.setTensorDim(3, 7);

  EXPECT_EQ(d.batch(), 4u);
  EXPECT_EQ(d.height(), 6u);
  EXPECT_EQ(d.width(), 7u);
  EXPECT_EQ(d.channel(), 5u);
}

TEST(nntrainer_Tensor, Tensor_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Tensor tensor = nntrainer::Tensor(1, 2, 3, NHWC_, FP32_);
  tensor.setZero();
  ASSERT_NE(nullptr, tensor.getData());
  if (tensor.getValue(0, 0, 0, 0) != 0.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_03_nhwc_p) {
  int status = ML_ERROR_NONE;
  int height = 3;
  int width = 10;
  int channel = 3;
  std::vector<std::vector<std::vector<float>>> in;
  for (int k = 0; k < height; ++k) {
    std::vector<std::vector<float>> ttv;
    for (int i = 0; i < width; ++i) {
      std::vector<float> tv;
      for (int j = 0; j < channel; ++j) {
        tv.push_back(k * width * channel + i * channel + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::Tensor tensor = nntrainer::Tensor(in, {NHWC_, FP32_});
  ASSERT_NE(nullptr, tensor.getData());

  if (tensor.getValue(0, 1, 0, 0) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_i_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l);

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

TEST(nntrainer_Tensor, multiply_i_02_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (channel * height * width) +
                               j * (width * channel) + k * channel + l);

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

TEST(nntrainer_Tensor, multiply_i_03_nhwc_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT(input, i * (channel * height * width) + j * (width * channel) +
                          k * channel + l);

  nntrainer::Tensor target2(batch, channel, height - 2, width - 1, NHWC_,
                            FP32_);
  status = input.multiply_i(target2);

  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  nntrainer::Tensor target3(batch, channel, height, width);
  status = input.multiply_i(target3);

  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_01_nhwc_p) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 2, 4, 5, NHWC_, FP32_);

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
    nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 5, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,    0,    0,    0,    4,    5,    6,    7,    16,   18,   20,   22,
      36,   39,   42,   45,   64,   68,   72,   76,   100,  105,  110,  115,
      144,  150,  156,  162,  196,  203,  210,  217,  256,  264,  272,  280,
      324,  333,  342,  351,  400,  410,  420,  430,  484,  495,  506,  517,
      576,  588,  600,  612,  676,  689,  702,  715,  784,  798,  812,  826,
      900,  915,  930,  945,  1024, 1040, 1056, 1072, 1156, 1173, 1190, 1207,
      1296, 1314, 1332, 1350, 1444, 1463, 1482, 1501, 1600, 1620, 1640, 1660,
      1764, 1785, 1806, 1827, 1936, 1958, 1980, 2002, 2116, 2139, 2162, 2185,
      2304, 2328, 2352, 2376, 2500, 2525, 2550, 2575, 2704, 2730, 2756, 2782,
      2916, 2943, 2970, 2997, 3136, 3164, 3192, 3220, 3364, 3393, 3422, 3451};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 4, 5, 1, NHWC_, FP32_);
    float answer_data[] = {
      0,    1,    4,    9,    0,    5,    12,   21,   32,   45,   60,   77,
      48,   65,   84,   105,  128,  153,  180,  209,  160,  189,  220,  253,
      288,  325,  364,  405,  336,  377,  420,  465,  512,  561,  612,  665,
      576,  629,  684,  741,  800,  861,  924,  989,  880,  945,  1012, 1081,
      1152, 1225, 1300, 1377, 1248, 1325, 1404, 1485, 1568, 1653, 1740, 1829,
      1680, 1769, 1860, 1953, 2048, 2145, 2244, 2345, 2176, 2277, 2380, 2485,
      2592, 2701, 2812, 2925, 2736, 2849, 2964, 3081, 3200, 3321, 3444, 3569,
      3360, 3485, 3612, 3741, 3872, 4005, 4140, 4277, 4048, 4185, 4324, 4465,
      4608, 4753, 4900, 5049, 4800, 4949, 5100, 5253, 5408, 5565, 5724, 5885,
      5616, 5777, 5940, 6105, 6272, 6441, 6612, 6785, 6496, 6669, 6844, 7021};

    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 1, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,   0,   0,   0,   4,   5,   6,   7,   0,   0,   0,   0,   12,  13,
      14,  15,  0,   0,   0,   0,   20,  21,  22,  23,  0,   0,   0,   0,
      28,  29,  30,  31,  0,   0,   0,   0,   36,  37,  38,  39,  80,  82,
      84,  86,  132, 135, 138, 141, 96,  98,  100, 102, 156, 159, 162, 165,
      112, 114, 116, 118, 180, 183, 186, 189, 128, 130, 132, 134, 204, 207,
      210, 213, 144, 146, 148, 150, 228, 231, 234, 237, 320, 324, 328, 332,
      420, 425, 430, 435, 352, 356, 360, 364, 460, 465, 470, 475, 384, 388,
      392, 396, 500, 505, 510, 515, 416, 420, 424, 428, 540, 545, 550, 555,
      448, 452, 456, 460, 580, 585, 590, 595};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 4, 1, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,   1,   4,   9,   16,  25,  36,  49,  0,   9,   20,  33,  48,  65,
      84,  105, 0,   17,  36,  57,  80,  105, 132, 161, 0,   25,  52,  81,
      112, 145, 180, 217, 0,   33,  68,  105, 144, 185, 228, 273, 0,   41,
      84,  129, 176, 225, 276, 329, 0,   49,  100, 153, 208, 265, 324, 385,
      0,   57,  116, 177, 240, 305, 372, 441, 0,   65,  132, 201, 272, 345,
      420, 497, 0,   73,  148, 225, 304, 385, 468, 553, 0,   81,  164, 249,
      336, 425, 516, 609, 0,   89,  180, 273, 368, 465, 564, 665, 0,   97,
      196, 297, 400, 505, 612, 721, 0,   105, 212, 321, 432, 545, 660, 777,
      0,   113, 228, 345, 464, 585, 708, 833};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 5, 1, NHWC_, FP32_);
    float answer_data[] = {
      0,    0,    0,    0,    0,    0,    0,    0,    8,    9,    10,   11,
      12,   13,   14,   15,   32,   34,   36,   38,   40,   42,   44,   46,
      72,   75,   78,   81,   84,   87,   90,   93,   128,  132,  136,  140,
      144,  148,  152,  156,  200,  205,  210,  215,  220,  225,  230,  235,
      288,  294,  300,  306,  312,  318,  324,  330,  392,  399,  406,  413,
      420,  427,  434,  441,  512,  520,  528,  536,  544,  552,  560,  568,
      648,  657,  666,  675,  684,  693,  702,  711,  800,  810,  820,  830,
      840,  850,  860,  870,  968,  979,  990,  1001, 1012, 1023, 1034, 1045,
      1152, 1164, 1176, 1188, 1200, 1212, 1224, 1236, 1352, 1365, 1378, 1391,
      1404, 1417, 1430, 1443, 1568, 1582, 1596, 1610, 1624, 1638, 1652, 1666};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);

    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 1, 1, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,  0,  0,  0,   4,   5,   6,   7,  0,  0,  0,   0,   12,  13,  14,
      15, 0,  0,  0,   0,   20,  21,  22, 23, 0,  0,   0,   0,   28,  29,
      30, 31, 0,  0,   0,   0,   36,  37, 38, 39, 0,   0,   0,   0,   44,
      45, 46, 47, 0,   0,   0,   0,   52, 53, 54, 55,  0,   0,   0,   0,
      60, 61, 62, 63,  0,   0,   0,   0,  68, 69, 70,  71,  0,   0,   0,
      0,  76, 77, 78,  79,  0,   0,   0,  0,  84, 85,  86,  87,  0,   0,
      0,  0,  92, 93,  94,  95,  0,   0,  0,  0,  100, 101, 102, 103, 0,
      0,  0,  0,  108, 109, 110, 111, 0,  0,  0,  0,   116, 117, 118, 119};

    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 4, 1, 1, NHWC_, FP32_);
    float answer_data[] = {
      0, 1,   4,   9,   0, 5,   12,  21,  0, 9,   20,  33,  0, 13,  28,  45,
      0, 17,  36,  57,  0, 21,  44,  69,  0, 25,  52,  81,  0, 29,  60,  93,
      0, 33,  68,  105, 0, 37,  76,  117, 0, 41,  84,  129, 0, 45,  92,  141,
      0, 49,  100, 153, 0, 53,  108, 165, 0, 57,  116, 177, 0, 61,  124, 189,
      0, 65,  132, 201, 0, 69,  140, 213, 0, 73,  148, 225, 0, 77,  156, 237,
      0, 81,  164, 249, 0, 85,  172, 261, 0, 89,  180, 273, 0, 93,  188, 285,
      0, 97,  196, 297, 0, 101, 204, 309, 0, 105, 212, 321, 0, 109, 220, 333,
      0, 113, 228, 345, 0, 117, 236, 357};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 1, 1, NHWC_, FP32_);
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
    nntrainer::TensorDim ref_dim(3, 5, 1, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 1, 4, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 1, 4, NHWC_, FP32_);
    float answer_data[] = {0,   0,   0,   0,   0,   5,   6,   7,   8,   9,
                           20,  22,  24,  26,  28,  45,  48,  51,  54,  57,
                           80,  84,  88,  92,  96,  125, 130, 135, 140, 145,
                           180, 186, 192, 198, 204, 245, 252, 259, 266, 273,
                           320, 328, 336, 344, 352, 405, 414, 423, 432, 441,
                           500, 510, 520, 530, 540, 605, 616, 627, 638, 649};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_supported_01_nhwc_n) {
  nntrainer::Tensor target(3, 1, 3, 1, NHWC_, FP32_);
  nntrainer::Tensor target2(3, 1, 3, 3, NHWC_, FP32_);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_broadcastable_02_nhwc_n) {
  nntrainer::Tensor target(3, 2, 4, 5, NHWC_, FP32_);
  nntrainer::Tensor target2(3, 2, 3, 1, NHWC_, FP32_);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l);

  nntrainer::Tensor result = input.multiply(0.0);
  if (result.getValue(0, 0, 1, 1) != 0.0)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_02_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT(input,
                 i * (batch * height * width) + j * (height * width) + k + 1);

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

TEST(nntrainer_Tensor, multiply_03_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT(input,
                 i * (height * width) + j * (height * width) + k * width + l);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1, NHWC_, FP32_);

  EXPECT_THROW({ input.multiply(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_04_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(batch, 2 * channel, height, width, NHWC_, FP32_);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);
  // shared_input is not continuous
  EXPECT_THROW(shared_input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_05_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, 2 * channel, height, width, NHWC_, FP32_);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.multiply(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_06_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);
  // input is not allocated now : alloc_now == false
  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + l);

  EXPECT_THROW(input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_07_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (height * width * channel) + j * (width * channel) +
                          k * channel + l);
  // test is not allocated.
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_08_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (height * width * channel) + j * (width * channel) +
                          k * channel + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (height * width * channel) + j * (width * channel) +
                         k * channel + 2);
  // output is not aloocated
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.multiply(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_float_01_nhwc_p) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor expected(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(expected, (i * (height * width * channel) +
                                 j * (width * channel) + k * channel + 1) *
                                  2);

  nntrainer::Tensor result = input.multiply(2.0);

  EXPECT_EQ(result, expected);
}

TEST(nntrainer_Tensor, divide_i_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l);

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

TEST(nntrainer_Tensor, divide_i_02_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l + 1);

  status = input.divide_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(indata[i], float(1.0));
  }
}

TEST(nntrainer_Tensor, divide_i_01_nhwc_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l + 1);

  status = input.divide_i((float)0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_i_02_nhwc_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l + 1);

  nntrainer::Tensor original(batch, channel, height - 2, width - 1, NHWC_,
                             FP32_);

  status = input.divide_i(original);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_01_nhwc_p) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l + 1);

  nntrainer::Tensor result = input.divide(1.0);

  float *previous = input.getData();
  ASSERT_NE(nullptr, previous);
  float *data = result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i]);
  }
}

TEST(nntrainer_Tensor, divide_02_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l + 1);

  EXPECT_THROW({ input.divide(0.0); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_03_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l + 1);

  nntrainer::Tensor test(batch - 1, channel - 1, height - 1, width - 1, NHWC_,
                         FP32_);

  EXPECT_THROW({ input.divide(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_04_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(batch, 2 * channel, height, width, NHWC_, FP32_);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_05_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, 2 * channel, height, width, NHWC_, FP32_);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.divide(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_06_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);

  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + l + 1);

  EXPECT_THROW(input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_07_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (height * width) + k * channel + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_08_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.divide(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_i_broadcast_01_nhwc_p) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(1, 2, 4, 5, NHWC_, FP32_);
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
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 1, 4, 5, NHWC_, FP32_);
    m.add_i(1);
    float answer_data[] = {

      1,       2, 1.5,     2, 1.66667, 2, 1.75,    2, 1.8,     2, 1.83333, 2,
      1.85714, 2, 1.875,   2, 1.88889, 2, 1.9,     2, 1.90909, 2, 1.91667, 2,
      1.92308, 2, 1.92857, 2, 1.93333, 2, 1.9375,  2, 1.94118, 2, 1.94444, 2,
      1.94737, 2, 1.95,    2, 1.95238, 2, 1.95455, 2, 1.95652, 2, 1.95833, 2,
      1.96,    2, 1.96154, 2, 1.96296, 2, 1.96429, 2, 1.96552, 2, 1.96667, 2,
      1.96774, 2, 1.96875, 2, 1.9697,  2, 1.97059, 2, 1.97143, 2, 1.97222, 2,
      1.97297, 2, 1.97368, 2, 1.97436, 2, 1.975,   2, 1.97561, 2, 1.97619, 2,
      1.97674, 2, 1.97727, 2, 1.97778, 2, 1.97826, 2, 1.97872, 2, 1.97917, 2,
      1.97959, 2, 1.98,    2, 1.98039, 2, 1.98077, 2, 1.98113, 2, 1.98148, 2,
      1.98182, 2, 1.98214, 2, 1.98246, 2, 1.98276, 2, 1.98305, 2, 1.98333, 2};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 2, 1, 1, NHWC_, FP32_);
    m.add_i(1);
    float answer_data[] = {
      1,         1,         3,         2,         5,         3,
      7,         4,         9,         5,         11,        6,
      13,        7,         15,        8,         17,        9,
      19,        10,        21,        11,        23,        12,
      25,        13,        27,        14,        29,        15,
      31,        16,        33,        17,        35,        18,
      37,        19,        39,        20,        13.666667, 10.5,
      14.333333, 11,        15,        11.5,      15.666667, 12,
      16.333334, 12.5,      17,        13,        17.666666, 13.5,
      18.333334, 14,        19,        14.5,      19.666666, 15,
      20.333334, 15.5,      21,        16,        21.666666, 16.5,
      22.333334, 17,        23,        17.5,      23.666666, 18,
      24.333334, 18.5,      25,        19,        25.666666, 19.5,
      26.333334, 20,        16.200001, 13.666667, 16.6,      14,
      17,        14.333333, 17.4,      14.666667, 17.799999, 15,
      18.200001, 15.333333, 18.6,      15.666667, 19,        16,
      19.4,      16.333334, 19.799999, 16.666666, 20.200001, 17,
      20.6,      17.333334, 21,        17.666666, 21.4,      18,
      21.799999, 18.333334, 22.200001, 18.666666, 22.6,      19,
      23,        19.333334, 23.4,      19.666666, 23.799999, 20};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(1, 2, 4, 1, NHWC_, FP32_);
    m.add_i(1);
    float answer_data[] = {
      1,         1,         3,         2,         5,         3,
      7,         4,         9,         5,         3.6666667, 3,
      4.3333335, 3.5,       5,         4,         5.6666665, 4.5,
      6.3333335, 5,         4.1999998, 3.6666667, 4.5999999, 4,
      5,         4.3333335, 5.4000001, 4.6666665, 5.8000002, 5,
      4.4285712, 4,         4.7142859, 4.25,      5,         4.5,
      5.2857141, 4.75,      5.5714288, 5,         41,        21,
      43,        22,        45,        23,        47,        24,
      49,        25,        17,        13,        17.666666, 13.5,
      18.333334, 14,        19,        14.5,      19.666666, 15,
      12.2,      10.333333, 12.6,      10.666667, 13,        11,
      13.4,      11.333333, 13.8,      11.666667, 10.142858, 9,
      10.428572, 9.25,      10.714286, 9.5,       11,        9.75,
      11.285714, 10,        81,        41,        83,        42,
      85,        43,        87,        44,        89,        45,
      30.333334, 23,        31,        23.5,      31.666666, 24,
      32.333332, 24.5,      33,        25,        20.200001, 17,
      20.6,      17.333334, 21,        17.666666, 21.4,      18,
      21.799999, 18.333334, 15.857142, 14,        16.142857, 14.25,
      16.428572, 14.5,      16.714285, 14.75,     17,        15};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 1, 1, 5, NHWC_, FP32_);
    m.add_i(1);
    float answer_data[] = {
      1,         2,         1.5,       2,         1.6666666, 2,
      1.75,      2,         1.8,       2,         11,        12,
      6.5,       7,         5,         5.3333335, 4.25,      4.5,
      3.8,       4,         21,        22,        11.5,      12,
      8.333333,  8.666667,  6.75,      7,         5.8000002, 6,
      31,        32,        16.5,      17,        11.666667, 12,
      9.25,      9.5,       7.8000002, 8,         6.8333335, 7,
      6.1428571, 6.2857141, 5.625,     5.75,      5.2222223, 5.3333335,
      4.9000001, 5,         8.5,       8.666667,  7.5714288, 7.7142859,
      6.875,     7,         6.3333335, 6.4444447, 5.9000001, 6,
      10.166667, 10.333333, 9,         9.1428576, 8.125,     8.25,
      7.4444447, 7.5555553, 6.9000001, 7,         11.833333, 12,
      10.428572, 10.571428, 9.375,     9.5,       8.5555553, 8.666667,
      7.9000001, 8,         7.3636365, 7.4545455, 6.9166665, 7,
      6.5384617, 6.6153846, 6.2142859, 6.2857141, 5.9333334, 6,
      8.272727,  8.363636,  7.75,      7.8333335, 7.3076925, 7.3846154,
      6.9285712, 7,         6.5999999, 6.6666665, 9.181818,  9.272727,
      8.583333,  8.666667,  8.0769234, 8.1538458, 7.6428571, 7.7142859,
      7.2666669, 7.3333335, 10.090909, 10.181818, 9.416667,  9.5,
      8.8461542, 8.9230766, 8.3571424, 8.4285717, 7.9333334, 8};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(1, 2, 1, 1, NHWC_, FP32_);
    m.add_i(1);
    float answer_data[] = {

      1,  1,   3,  2,   5,  3,   7,  4,   9,  5,   11,  6,   13,  7,   15,
      8,  17,  9,  19,  10, 21,  11, 23,  12, 25,  13,  27,  14,  29,  15,
      31, 16,  33, 17,  35, 18,  37, 19,  39, 20,  41,  21,  43,  22,  45,
      23, 47,  24, 49,  25, 51,  26, 53,  27, 55,  28,  57,  29,  59,  30,
      61, 31,  63, 32,  65, 33,  67, 34,  69, 35,  71,  36,  73,  37,  75,
      38, 77,  39, 79,  40, 81,  41, 83,  42, 85,  43,  87,  44,  89,  45,
      91, 46,  93, 47,  95, 48,  97, 49,  99, 50,  101, 51,  103, 52,  105,
      53, 107, 54, 109, 55, 111, 56, 113, 57, 115, 58,  117, 59,  119, 60};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(1, 1, 4, 1, NHWC_, FP32_);
    m.add_i(1);
    float answer_data[] = {
      1,         2,         3,     4,         5,         6,
      7,         8,         9,     10,        5.5,       6,
      6.5,       7,         7.5,   8,         8.5,       9,
      9.5,       10,        7,     7.3333335, 7.6666665, 8,
      8.333333,  8.666667,  9,     9.333333,  9.666667,  10,
      7.75,      8,         8.25,  8.5,       8.75,      9,
      9.25,      9.5,       9.75,  10,        41,        42,
      43,        44,        45,    46,        47,        48,
      49,        50,        25.5,  26,        26.5,      27,
      27.5,      28,        28.5,  29,        29.5,      30,
      20.333334, 20.666666, 21,    21.333334, 21.666666, 22,
      22.333334, 22.666666, 23,    23.333334, 17.75,     18,
      18.25,     18.5,      18.75, 19,        19.25,     19.5,
      19.75,     20,        81,    82,        83,        84,
      85,        86,        87,    88,        89,        90,
      45.5,      46,        46.5,  47,        47.5,      48,
      48.5,      49,        49.5,  50,        33.666668, 34,
      34.333332, 34.666668, 35,    35.333332, 35.666668, 36,
      36.333332, 36.666668, 27.75, 28,        28.25,     28.5,
      28.75,     29,        29.25, 29.5,      29.75,     30};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 4, 5, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 1, 1, 1, NHWC_, FP32_);
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
    nntrainer::TensorDim ref_dim(3, 2, 5, 1, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 2, 5, 1, NHWC_, FP32_);
    t.add_i(1);
    nntrainer::Tensor m = ranged(3, 2, 1, 1, NHWC_, FP32_);
    m.add_i(1);
    float answer_data[] = {
      1,       1,       3,       2,   5, 3,       7,       4,       9,       5,
      3.66667, 3,       4.33333, 3.5, 5, 4,       5.66667, 4.5,     6.33333, 5,
      4.2,     3.66667, 4.6,     4,   5, 4.33333, 5.4,     4.66667, 5.8,     5};

    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, divide_i_broadcast_not_supported_01_nhwc_n) {
  nntrainer::Tensor target(3, 1, 3, 1, NHWC_, FP32_);
  nntrainer::Tensor target2(3, 1, 3, 3, NHWC_, FP32_);

  EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_i_broadcast_not_broadcastable_02_nhwc_n) {
  nntrainer::Tensor target(3, 2, 4, 5, NHWC_, FP32_);
  nntrainer::Tensor target2(3, 2, 3, 1, NHWC_, FP32_);

  EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief operand dimension is not right
 */
TEST(nntrainer_Tensor, add_i_01_nhwc_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor target2(batch, height - 2, width - 3, NHWC_, FP32_);

  status = target.add_i(target2);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_i_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 3;

  nntrainer::Tensor target(batch, height, width, channel, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(target, i * (height * width * channel) +
                                j * (width * channel) + k * channel + 1 + l);

  nntrainer::Tensor original(batch, height, width, channel);
  original.copy(target);

  status = target.add_i(2.1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *previous = original.getData();
  ASSERT_NE(nullptr, previous);
  float *data = target.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i] + (float)2.1);
  }
}

TEST(nntrainer_Tensor, add_i_02_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 2;

  nntrainer::Tensor target(batch, height, width, channel, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(target, i * (height * width * channel) +
                                j * (width * channel) + k * channel + 1);

  nntrainer::Tensor original(height, width, channel, NHWC_, FP32_);
  original.copy(target);

  status = target.add_i(target, 3.0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *previous = original.getData();
  ASSERT_NE(nullptr, previous);
  float *data = target.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i] * 4.0);
  }
}

TEST(nntrainer_Tensor, add_i_broadcast_01_nhwc_p) {
  nntrainer::TensorDim ref_dim(3, 4, 5, 2, NHWC_, FP32_);
  {
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 4, 5, 2, NHWC_, FP32_);
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
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 5, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,   1,   2,   3,   5,   6,   7,   8,   10,  11,  12,  13,  15,  16,
      17,  18,  20,  21,  22,  23,  25,  26,  27,  28,  30,  31,  32,  33,
      35,  36,  37,  38,  40,  41,  42,  43,  45,  46,  47,  48,  50,  51,
      52,  53,  55,  56,  57,  58,  60,  61,  62,  63,  65,  66,  67,  68,
      70,  71,  72,  73,  75,  76,  77,  78,  80,  81,  82,  83,  85,  86,
      87,  88,  90,  91,  92,  93,  95,  96,  97,  98,  100, 101, 102, 103,
      105, 106, 107, 108, 110, 111, 112, 113, 115, 116, 117, 118, 120, 121,
      122, 123, 125, 126, 127, 128, 130, 131, 132, 133, 135, 136, 137, 138,
      140, 141, 142, 143, 145, 146, 147, 148};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 4, 5, 1, NHWC_, FP32_);
    float answer_data[] = {
      0,   2,   4,   6,   4,   6,   8,   10,  12,  14,  16,  18,  16,  18,
      20,  22,  24,  26,  28,  30,  28,  30,  32,  34,  36,  38,  40,  42,
      40,  42,  44,  46,  48,  50,  52,  54,  52,  54,  56,  58,  60,  62,
      64,  66,  64,  66,  68,  70,  72,  74,  76,  78,  76,  78,  80,  82,
      84,  86,  88,  90,  88,  90,  92,  94,  96,  98,  100, 102, 100, 102,
      104, 106, 108, 110, 112, 114, 112, 114, 116, 118, 120, 122, 124, 126,
      124, 126, 128, 130, 132, 134, 136, 138, 136, 138, 140, 142, 144, 146,
      148, 150, 148, 150, 152, 154, 156, 158, 160, 162, 160, 162, 164, 166,
      168, 170, 172, 174, 172, 174, 176, 178};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 1, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,   1,   2,   3,   5,   6,   7,   8,   8,   9,   10,  11,  13,  14,
      15,  16,  16,  17,  18,  19,  21,  22,  23,  24,  24,  25,  26,  27,
      29,  30,  31,  32,  32,  33,  34,  35,  37,  38,  39,  40,  42,  43,
      44,  45,  47,  48,  49,  50,  50,  51,  52,  53,  55,  56,  57,  58,
      58,  59,  60,  61,  63,  64,  65,  66,  66,  67,  68,  69,  71,  72,
      73,  74,  74,  75,  76,  77,  79,  80,  81,  82,  84,  85,  86,  87,
      89,  90,  91,  92,  92,  93,  94,  95,  97,  98,  99,  100, 100, 101,
      102, 103, 105, 106, 107, 108, 108, 109, 110, 111, 113, 114, 115, 116,
      116, 117, 118, 119, 121, 122, 123, 124};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 4, 1, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,   2,   4,   6,   8,   10,  12,  14,  8,   10,  12,  14,  16,  18,
      20,  22,  16,  18,  20,  22,  24,  26,  28,  30,  24,  26,  28,  30,
      32,  34,  36,  38,  32,  34,  36,  38,  40,  42,  44,  46,  40,  42,
      44,  46,  48,  50,  52,  54,  48,  50,  52,  54,  56,  58,  60,  62,
      56,  58,  60,  62,  64,  66,  68,  70,  64,  66,  68,  70,  72,  74,
      76,  78,  72,  74,  76,  78,  80,  82,  84,  86,  80,  82,  84,  86,
      88,  90,  92,  94,  88,  90,  92,  94,  96,  98,  100, 102, 96,  98,
      100, 102, 104, 106, 108, 110, 104, 106, 108, 110, 112, 114, 116, 118,
      112, 114, 116, 118, 120, 122, 124, 126};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 5, 1, NHWC_, FP32_);
    float answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   9,   10,  11,  12,  13,  14,
      15,  16,  18,  19,  20,  21,  22,  23,  24,  25,  27,  28,  29,  30,
      31,  32,  33,  34,  36,  37,  38,  39,  40,  41,  42,  43,  45,  46,
      47,  48,  49,  50,  51,  52,  54,  55,  56,  57,  58,  59,  60,  61,
      63,  64,  65,  66,  67,  68,  69,  70,  72,  73,  74,  75,  76,  77,
      78,  79,  81,  82,  83,  84,  85,  86,  87,  88,  90,  91,  92,  93,
      94,  95,  96,  97,  99,  100, 101, 102, 103, 104, 105, 106, 108, 109,
      110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 124,
      126, 127, 128, 129, 130, 131, 132, 133};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 1, 1, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,   1,   2,   3,   5,   6,   7,   8,   8,   9,   10,  11,  13,  14,
      15,  16,  16,  17,  18,  19,  21,  22,  23,  24,  24,  25,  26,  27,
      29,  30,  31,  32,  32,  33,  34,  35,  37,  38,  39,  40,  40,  41,
      42,  43,  45,  46,  47,  48,  48,  49,  50,  51,  53,  54,  55,  56,
      56,  57,  58,  59,  61,  62,  63,  64,  64,  65,  66,  67,  69,  70,
      71,  72,  72,  73,  74,  75,  77,  78,  79,  80,  80,  81,  82,  83,
      85,  86,  87,  88,  88,  89,  90,  91,  93,  94,  95,  96,  96,  97,
      98,  99,  101, 102, 103, 104, 104, 105, 106, 107, 109, 110, 111, 112,
      112, 113, 114, 115, 117, 118, 119, 120};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 4, 1, 1, NHWC_, FP32_);
    float answer_data[] = {
      0,   2,   4,   6,   4,   6,   8,   10,  8,   10,  12,  14,  12,  14,
      16,  18,  16,  18,  20,  22,  20,  22,  24,  26,  24,  26,  28,  30,
      28,  30,  32,  34,  32,  34,  36,  38,  36,  38,  40,  42,  40,  42,
      44,  46,  44,  46,  48,  50,  48,  50,  52,  54,  52,  54,  56,  58,
      56,  58,  60,  62,  60,  62,  64,  66,  64,  66,  68,  70,  68,  70,
      72,  74,  72,  74,  76,  78,  76,  78,  80,  82,  80,  82,  84,  86,
      84,  86,  88,  90,  88,  90,  92,  94,  92,  94,  96,  98,  96,  98,
      100, 102, 100, 102, 104, 106, 104, 106, 108, 110, 108, 110, 112, 114,
      112, 114, 116, 118, 116, 118, 120, 122};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 1, 1, NHWC_, FP32_);
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
    nntrainer::Tensor t = ranged(3, 4, 5, 2, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 1, 1, 1, NHWC_, FP32_);
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
    nntrainer::TensorDim ref_dim(3, 5, 1, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 1, 4, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 1, 1, 4, NHWC_, FP32_);
    float answer_data[] = {0,  1,  2,  3,  4,  6,  7,  8,  9,  10, 12, 13,
                           14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27,
                           28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42,
                           43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56,
                           57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(1, 1, 2, 1, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(1, 1, 2, 1, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 1, 2, 1, NHWC_, FP32_);
    float answer_data[] = {0.0, 2.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(16, 1, 1, 1, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(16, 1, 1, 1, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(1, 1, 1, 1, NHWC_, FP32_);
    float answer_data[] = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                           8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, add_i_broadcast_not_supported_01_nhwc_n) {
  nntrainer::Tensor target(3, 1, 3, 1, NHWC_, FP32_);
  nntrainer::Tensor target2(3, 1, 3, 3, NHWC_, FP32_);

  EXPECT_EQ(target.add_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_i_broadcast_not_broadcastable_02_nhwc_n) {
  nntrainer::Tensor target(3, 2, 4, 5, NHWC_, FP32_);
  nntrainer::Tensor target2(3, 2, 3, 1, NHWC_, FP32_);

  EXPECT_EQ(target.add_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);

  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l);

  nntrainer::Tensor result = input.add(1.0);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    if (data[i] != indata[i] + (float)1.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_02_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);
  nntrainer::Tensor result = input.add(input);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  EXPECT_EQ(result.getFormat(), input.getFormat());

  for (int i = 0; i < batch * height * width * channel; ++i) {
    if (data[i] != indata[i] + indata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_03_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1, channel, NHWC_,
                         FP32_);

  EXPECT_THROW({ input.add(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, add_04_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(batch, channel * 2, height, width, NHWC_, FP32_);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.add(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_05_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, height, width, channel, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, height, width, channel * 2, NHWC_, FP32_);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.add(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_06_nhw_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, height, width, channel, NHWC_, FP32_);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + 1);

  EXPECT_THROW(input.add(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_08_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, height, width, channel, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + 2);

  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.add(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, pow_01_nhwc_p) {

  nntrainer::Tensor input = constant(4.0, 3, 2, 4, 5, NHWC_, FP32_);

  nntrainer::Tensor actual, expected;

  actual = input.pow(0.5f);
  expected = constant(2.0, 3, 2, 4, 5, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = input.pow(2.0f);
  expected = constant(16.0, 3, 2, 4, 5, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = input.pow(-0.5f);
  expected = constant(0.5, 3, 2, 4, 5, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, subtract_i_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 3;

  nntrainer::Tensor target(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(target, i * (height * width * channel) +
                                j * (width * channel) + k * channel + l);

  nntrainer::Tensor original;
  original.copy(target);

  status = target.subtract_i(2.1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *previous = original.getData();
  ASSERT_NE(nullptr, previous);
  float *data = target.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i] - (float)2.1);
  }
}

TEST(nntrainer_Tensor, subtract_i_02_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 3;

  nntrainer::Tensor target(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(target, i * (height * width * channel) +
                                j * (width * channel) + k * channel + l);

  status = target.subtract_i(target);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float *data = target.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0);
  }
}

TEST(nntrainer_Tensor, subtract_i_03_nhwc_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 3;

  nntrainer::Tensor target(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(target, i * (height * width * channel) +
                                j * (width * channel) + k * channel + l);

  nntrainer::Tensor target2(batch, height, width - 3, channel - 1, NHWC_,
                            FP32_);

  status = target.subtract_i(target2);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, subtract_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor result = input.subtract(1.0);

  float *data = result.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    if (data[i] != indata[i] - 1.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, subtract_02_nhwc_p) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor result = input.subtract(input);

  EXPECT_EQ(constant(0.0, batch, channel, height, width, NHWC_, FP32_), result);
}

TEST(nntrainer_Tensor, subtract_03_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor test(batch - 1, channel - 1, height, width - 1, NHWC_,
                         FP32_);

  EXPECT_THROW({ input.subtract(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_04_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(batch, 2 * channel, height, width, NHWC_, FP32_);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_05_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, 2 * channel, height, width, NHWC_, FP32_);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.subtract(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_06_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + 1);

  EXPECT_THROW(input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_07_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_08_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 2);

  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.subtract(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_float_01_nhwc_p) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor expected(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(expected, i * (height * width * channel) +
                                  j * (width * channel) + k * channel);

  nntrainer::Tensor result = input.subtract(1.0);

  EXPECT_EQ(result, expected);
}

TEST(nntrainer_Tensor, sum_01_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);

  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l);

  EXPECT_THROW({ input.sum(4); }, std::out_of_range);
}

TEST(nntrainer_Tensor, sum_02_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l);

  EXPECT_THROW({ input.sum(-1); }, std::out_of_range);
}

TEST(nntrainer_Tensor, sum_02_nhwc_p) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::Tensor ans0(
    std::vector<std::vector<std::vector<std::vector<float>>>>({{{{123, 126},
                                                                 {129, 132},
                                                                 {135, 138},
                                                                 {141, 144},
                                                                 {147, 150},
                                                                 {153, 156},
                                                                 {159, 162},
                                                                 {165, 168},
                                                                 {171, 174},
                                                                 {177, 180}},
                                                                {{183, 186},
                                                                 {189, 192},
                                                                 {195, 198},
                                                                 {201, 204},
                                                                 {207, 210},
                                                                 {213, 216},
                                                                 {219, 222},
                                                                 {225, 228},
                                                                 {231, 234},
                                                                 {237, 240}}}}),
    {NHWC_, FP32_});

  nntrainer::Tensor ans1(
    std::vector<std::vector<std::vector<std::vector<float>>>>(
      {{{{3}, {7}, {11}, {15}, {19}, {23}, {27}, {31}, {35}, {39}},
        {{43}, {47}, {51}, {55}, {59}, {63}, {67}, {71}, {75}, {79}}},
       {{{83}, {87}, {91}, {95}, {99}, {103}, {107}, {111}, {115}, {119}},
        {{123}, {127}, {131}, {135}, {139}, {143}, {147}, {151}, {155}, {159}}},
       {{{163}, {167}, {171}, {175}, {179}, {183}, {187}, {191}, {195}, {199}},
        {{203},
         {207},
         {211},
         {215},
         {219},
         {223},
         {227},
         {231},
         {235},
         {239}}}}),
    {NHWC_, FP32_});

  nntrainer::Tensor ans2(
    std::vector<std::vector<std::vector<std::vector<float>>>>({{{{22, 24},
                                                                 {26, 28},
                                                                 {30, 32},
                                                                 {34, 36},
                                                                 {38, 40},
                                                                 {42, 44},
                                                                 {46, 48},
                                                                 {50, 52},
                                                                 {54, 56},
                                                                 {58, 60}}},
                                                               {{{102, 104},
                                                                 {106, 108},
                                                                 {110, 112},
                                                                 {114, 116},
                                                                 {118, 120},
                                                                 {122, 124},
                                                                 {126, 128},
                                                                 {130, 132},
                                                                 {134, 136},
                                                                 {138, 140}}},
                                                               {{{182, 184},
                                                                 {186, 188},
                                                                 {190, 192},
                                                                 {194, 196},
                                                                 {198, 200},
                                                                 {202, 204},
                                                                 {206, 208},
                                                                 {210, 212},
                                                                 {214, 216},
                                                                 {218, 220}}}}),
    {NHWC_, FP32_});

  nntrainer::Tensor ans3(
    std::vector<std::vector<std::vector<std::vector<float>>>>(
      {{{{100, 110}}, {{300, 310}}},
       {{{500, 510}}, {{700, 710}}},
       {{{900, 910}}, {{1100, 1110}}}}),
    {NHWC_, FP32_});

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * (channel) + l + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result1 = input.sum(1);
  nntrainer::Tensor result2 = input.sum(2);
  nntrainer::Tensor result3 = input.sum(3);

  EXPECT_EQ(ans0, result0);
  EXPECT_EQ(ans1, result1);
  EXPECT_EQ(ans2, result2);
  EXPECT_EQ(ans3, result3);
}

TEST(nntrainer_Tensor, sum_03_nhwc_p) {
  const int batch = 3;
  const int channel = 2;
  const int height = 2;
  const int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * channel * width) +
                               j * (width * channel) + k * (channel) + l + 1);
  // Test for alpha == 1 and beta == 0 and dimension of reduced axis == 1
  {
    nntrainer::Tensor ans_0_1_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{123, 126},
           {129, 132},
           {135, 138},
           {141, 144},
           {147, 150},
           {153, 156},
           {159, 162},
           {165, 168},
           {171, 174},
           {177, 180}},
          {{183, 186},
           {189, 192},
           {195, 198},
           {201, 204},
           {207, 210},
           {213, 216},
           {219, 222},
           {225, 228},
           {231, 234},
           {237, 240}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_1_1_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(

        {{{{3}, {7}, {11}, {15}, {19}, {23}, {27}, {31}, {35}, {39}},
          {{43}, {47}, {51}, {55}, {59}, {63}, {67}, {71}, {75}, {79}}},
         {{{83}, {87}, {91}, {95}, {99}, {103}, {107}, {111}, {115}, {119}},
          {{123},
           {127},
           {131},
           {135},
           {139},
           {143},
           {147},
           {151},
           {155},
           {159}}},
         {{{163},
           {167},
           {171},
           {175},
           {179},
           {183},
           {187},
           {191},
           {195},
           {199}},
          {{203},
           {207},
           {211},
           {215},
           {219},
           {223},
           {227},
           {231},
           {235},
           {239}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_2_1_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(

        {{{{22, 24},
           {26, 28},
           {30, 32},
           {34, 36},
           {38, 40},
           {42, 44},
           {46, 48},
           {50, 52},
           {54, 56},
           {58, 60}}},
         {{{102, 104},
           {106, 108},
           {110, 112},
           {114, 116},
           {118, 120},
           {122, 124},
           {126, 128},
           {130, 132},
           {134, 136},
           {138, 140}}},
         {{{182, 184},
           {186, 188},
           {190, 192},
           {194, 196},
           {198, 200},
           {202, 204},
           {206, 208},
           {210, 212},
           {214, 216},
           {218, 220}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_3_1_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{100, 110}}, {{300, 310}}},
         {{{500, 510}}, {{700, 710}}},
         {{{900, 910}}, {{1100, 1110}}}}),
      {NHWC_, FP32_});

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
        {{{{125, 130},
           {135, 140},
           {145, 150},
           {155, 160},
           {165, 170},
           {175, 180},
           {185, 190},
           {195, 200},
           {205, 210},
           {215, 220}},
          {{225, 230},
           {235, 240},
           {245, 250},
           {255, 260},
           {265, 270},
           {275, 280},
           {285, 290},
           {295, 300},
           {305, 310},
           {315, 320}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_1_1_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{5}, {11}, {17}, {23}, {29}, {35}, {41}, {47}, {53}, {59}},
          {{65}, {71}, {77}, {83}, {89}, {95}, {101}, {107}, {113}, {119}}},
         {{{125},
           {131},
           {137},
           {143},
           {149},
           {155},
           {161},
           {167},
           {173},
           {179}},
          {{185},
           {191},
           {197},
           {203},
           {209},
           {215},
           {221},
           {227},
           {233},
           {239}}},
         {{{245},
           {251},
           {257},
           {263},
           {269},
           {275},
           {281},
           {287},
           {293},
           {299}},
          {{305},
           {311},
           {317},
           {323},
           {329},
           {335},
           {341},
           {347},
           {353},
           {359}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_2_1_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{24, 28},
           {32, 36},
           {40, 44},
           {48, 52},
           {56, 60},
           {64, 68},
           {72, 76},
           {80, 84},
           {88, 92},
           {96, 100}}},
         {{{144, 148},
           {152, 156},
           {160, 164},
           {168, 172},
           {176, 180},
           {184, 188},
           {192, 196},
           {200, 204},
           {208, 212},
           {216, 220}}},
         {{{264, 268},
           {272, 276},
           {280, 284},
           {288, 292},
           {296, 300},
           {304, 308},
           {312, 316},
           {320, 324},
           {328, 332},
           {336, 340}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_3_1_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(

        {{{{102, 114}}, {{306, 318}}},
         {{{510, 522}}, {{714, 726}}},
         {{{918, 930}}, {{1122, 1134}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor output_0_1_2(1, channel, height, width, NHWC_, FP32_);
    {
      const int batch = 1;
      GEN_TEST_INPUT_NHWC(output_0_1_2, i * (channel * height * width) +
                                          j * (width * channel) +
                                          k * (channel) + l + 1);
    }
    nntrainer::Tensor output_1_1_2(batch, 1, height, width, NHWC_, FP32_);
    {
      const int channel = 1;
      GEN_TEST_INPUT_NHWC(output_1_1_2, i * (channel * height * width) +
                                          j * (width * channel) +
                                          k * (channel) + l + 1);
    }
    nntrainer::Tensor output_2_1_2(batch, channel, 1, width, NHWC_, FP32_);
    {
      const int height = 1;
      GEN_TEST_INPUT_NHWC(output_2_1_2, i * (channel * height * width) +
                                          j * (width * channel) +
                                          k * (channel) + l + 1);
    }
    nntrainer::Tensor output_3_1_2(batch, channel, height, 1, NHWC_, FP32_);
    {
      const int width = 1;
      GEN_TEST_INPUT_NHWC(output_3_1_2, i * (channel * height * width) +
                                          j * (channel * width) +
                                          k * (channel) + l + 1);
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
        {{{{246, 252},
           {258, 264},
           {270, 276},
           {282, 288},
           {294, 300},
           {306, 312},
           {318, 324},
           {330, 336},
           {342, 348},
           {354, 360}},
          {{366, 372},
           {378, 384},
           {390, 396},
           {402, 408},
           {414, 420},
           {426, 432},
           {438, 444},
           {450, 456},
           {462, 468},
           {474, 480}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_1_2_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{6}, {14}, {22}, {30}, {38}, {46}, {54}, {62}, {70}, {78}},
          {{86}, {94}, {102}, {110}, {118}, {126}, {134}, {142}, {150}, {158}}},
         {{{166},
           {174},
           {182},
           {190},
           {198},
           {206},
           {214},
           {222},
           {230},
           {238}},
          {{246},
           {254},
           {262},
           {270},
           {278},
           {286},
           {294},
           {302},
           {310},
           {318}}},
         {{{326},
           {334},
           {342},
           {350},
           {358},
           {366},
           {374},
           {382},
           {390},
           {398}},
          {{406},
           {414},
           {422},
           {430},
           {438},
           {446},
           {454},
           {462},
           {470},
           {478}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_2_2_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(

        {{{{44, 48},
           {52, 56},
           {60, 64},
           {68, 72},
           {76, 80},
           {84, 88},
           {92, 96},
           {100, 104},
           {108, 112},
           {116, 120}}},
         {{{204, 208},
           {212, 216},
           {220, 224},
           {228, 232},
           {236, 240},
           {244, 248},
           {252, 256},
           {260, 264},
           {268, 272},
           {276, 280}}},
         {{{364, 368},
           {372, 376},
           {380, 384},
           {388, 392},
           {396, 400},
           {404, 408},
           {412, 416},
           {420, 424},
           {428, 432},
           {436, 440}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_3_2_0(
      std::vector<std::vector<std::vector<std::vector<float>>>>(

        {{{{200, 220}}, {{600, 620}}},
         {{{1000, 1020}}, {{1400, 1420}}},
         {{{1800, 1820}}, {{2200, 2220}}}}),
      {NHWC_, FP32_});

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
        {{{{248, 256},
           {264, 272},
           {280, 288},
           {296, 304},
           {312, 320},
           {328, 336},
           {344, 352},
           {360, 368},
           {376, 384},
           {392, 400}},
          {{408, 416},
           {424, 432},
           {440, 448},
           {456, 464},
           {472, 480},
           {488, 496},
           {504, 512},
           {520, 528},
           {536, 544},
           {552, 560}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_1_2_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{8}, {18}, {28}, {38}, {48}, {58}, {68}, {78}, {88}, {98}},
          {{108},
           {118},
           {128},
           {138},
           {148},
           {158},
           {168},
           {178},
           {188},
           {198}}},
         {{{208},
           {218},
           {228},
           {238},
           {248},
           {258},
           {268},
           {278},
           {288},
           {298}},
          {{308},
           {318},
           {328},
           {338},
           {348},
           {358},
           {368},
           {378},
           {388},
           {398}}},
         {{{408},
           {418},
           {428},
           {438},
           {448},
           {458},
           {468},
           {478},
           {488},
           {498}},
          {{508},
           {518},
           {528},
           {538},
           {548},
           {558},
           {568},
           {578},
           {588},
           {598}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_2_2_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{46, 52},
           {58, 64},
           {70, 76},
           {82, 88},
           {94, 100},
           {106, 112},
           {118, 124},
           {130, 136},
           {142, 148},
           {154, 160}}},
         {{{246, 252},
           {258, 264},
           {270, 276},
           {282, 288},
           {294, 300},
           {306, 312},
           {318, 324},
           {330, 336},
           {342, 348},
           {354, 360}}},
         {{{446, 452},
           {458, 464},
           {470, 476},
           {482, 488},
           {494, 500},
           {506, 512},
           {518, 524},
           {530, 536},
           {542, 548},
           {554, 560}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor ans_3_2_2(
      std::vector<std::vector<std::vector<std::vector<float>>>>(
        {{{{202, 224}}, {{606, 628}}},
         {{{1010, 1032}}, {{1414, 1436}}},
         {{{1818, 1840}}, {{2222, 2244}}}}),
      {NHWC_, FP32_});

    nntrainer::Tensor output_0_2_2(1, channel, height, width, NHWC_, FP32_);
    {
      const int batch = 1;
      GEN_TEST_INPUT_NHWC(output_0_2_2, i * (channel * height * width) +
                                          j * (channel * width) +
                                          k * (channel) + l + 1);
    }
    nntrainer::Tensor output_1_2_2(batch, 1, height, width, NHWC_, FP32_);
    {
      const int channel = 1;
      GEN_TEST_INPUT_NHWC(output_1_2_2, i * (channel * height * width) +
                                          j * (channel * width) +
                                          k * (channel) + l + 1);
    }
    nntrainer::Tensor output_2_2_2(batch, channel, 1, width, NHWC_, FP32_);
    {
      const int height = 1;
      GEN_TEST_INPUT_NHWC(output_2_2_2, i * (channel * height * width) +
                                          j * (channel * width) +
                                          k * (channel) + l + 1);
    }
    nntrainer::Tensor output_3_2_2(batch, channel, height, 1, NHWC_, FP32_);
    {
      const int width = 1;
      GEN_TEST_INPUT_NHWC(output_3_2_2, i * (channel * height * width) +
                                          j * (channel * width) +
                                          k * (channel) + l + 1);
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

TEST(nntrainer_Tensor, sum_04_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (channel * width) + k * channel + l + 1);

  nntrainer::Tensor result = input.sum_by_batch();
  if (result.getValue(0, 0, 0, 0) != 820 ||
      result.getValue(1, 0, 0, 0) != 2420 ||
      result.getValue(2, 0, 0, 0) != 4020)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiple_sum_invalid_args_01_hnwc_n) {
  nntrainer::Tensor t = constant(1.0, 1, 1, 1, 1, NHWC_, FP32_);
  EXPECT_THROW(t.sum(std::vector<unsigned int>()), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiple_sum_out_of_range_nhwc_n) {
  nntrainer::Tensor t = constant(1.0, 1, 1, 1, 1, NHWC_, FP32_);
  EXPECT_THROW(t.sum(7), std::out_of_range);
}

TEST(nntrainer_Tensor, multiple_sum_nhwc_p) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7, NHWC_, FP32_);
  nntrainer::Tensor actual, expected;

  actual = t.sum({0, 1});
  expected = constant(2 * 3, 1, 1, 5, 7, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = t.sum({1, 2, 3});
  expected = constant(3 * 5 * 7, 2, 1, 1, 1, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = t.sum({3, 1});
  expected = constant(7 * 3, 2, 1, 5, 1, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = t.sum({3, 1}, 0.5);
  expected = constant(7 * 3 * 0.5, 2, 1, 5, 1, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_nhwc_p) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7, NHWC_, FP32_);

  nntrainer::Tensor actual, expected;

  actual = t.average();
  expected = constant(1.0, 1, 1, 1, 1, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  int idx = 0;
  t = t.apply((std::function<float(float)>)[&](float in) { return idx++ % 2; });

  actual = t.average();
  expected = constant(0.5, 1, 1, 1, 1, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_axis_nhwc_p) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2, NHWC_, FP32_);
  int idx = 0;
  std::function<float(float)> f = [&](float in) { return idx++ % 2; };
  t = t.apply(f);

  nntrainer::Tensor actual, expected;

  actual = t.average(0);
  expected = constant(0, 1, 2, 2, 2, NHWC_, FP32_).apply(f);
  EXPECT_EQ(actual, expected);

  actual = t.average(1);
  expected = constant(0.5, 2, 1, 2, 2, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = t.average(2);
  expected = constant(0, 2, 2, 1, 2, NHWC_, FP32_).apply(f);
  EXPECT_EQ(actual, expected);

  actual = t.average(3);
  expected = constant(0, 2, 2, 2, 1, NHWC_, FP32_).apply(f);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_axis_out_of_range_01_nhwc_n) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2, NHWC_, FP32_);
  EXPECT_THROW(t.average(-1), std::out_of_range);
}

TEST(nntrainer_Tensor, average_axis_out_of_range_02_nhwc_n) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2, NHWC_, FP32_);
  EXPECT_THROW(t.average(7), std::out_of_range);
}

TEST(nntrainer_Tensor, average_multiple_axes_nhwc_p) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7, NHWC_, FP32_);
  nntrainer::Tensor actual, expected;

  actual = t.average({0, 1, 2});
  expected = constant(1.0, 1, 1, 1, 7, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = t.average({0, 1, 2, 3});
  expected = constant(1.0, 1, 1, 1, 1, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = t.average({3, 1});
  expected = constant(1.0, 2, 1, 5, 1, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);

  actual = t.average({3, 1, 1, 1, 3});
  expected = constant(1.0, 2, 1, 5, 1, NHWC_, FP32_);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_multiple_axes_01_nhwc_n) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7, NHWC_, FP32_);
  EXPECT_THROW(t.average({5, 7}), std::out_of_range);
}

/// @note this test case demonstrates it is dangerous to use sharedConstTensor
/// to const correct the inner data.
TEST(nntrainer_Tensor,
     constructor_from_shared_const_ptr_shares_variable_nhwc_n) {
  nntrainer::sharedConstTensor A =
    MAKE_SHARED_TENSOR(constant(1.0f, 3, 4, 5, 6, NHWC_, FP32_));

  nntrainer::Tensor B = *A;
  nntrainer::Tensor C = A->clone();

  B.setValue(2, 3, 4, 5, 2.0f);
  EXPECT_EQ(*A, B);
  EXPECT_NE(*A, C);

  C.reshape(nntrainer::TensorDim(3, 4, 6, 5, NHWC_, FP32_));
  EXPECT_EQ(A->getDim(), B.getDim());
  EXPECT_NE(A->getDim(), C.getDim());
}

TEST(nntrainer_Tensor, dot_01_nhwc_n) {
  nntrainer::Tensor input(2, 4, 5, 3, NHWC_, FP32_);
  nntrainer::Tensor m(1, 4, 5, 3, NHWC_, FP32_);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m), std::runtime_error);
}

TEST(nntrainer_Tensor, dot_02_nhwc_n) {
  nntrainer::Tensor input(2, 3, 4, 5, NHWC_, FP32_);
  nntrainer::Tensor m(1, 3, 4, 5, NHWC_, FP32_);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m, true),
               std::runtime_error);
}

TEST(nntrainer_Tensor, dot_02_nhwc_p) {
  nntrainer::Tensor input(2, 3, 4, 5, NHWC_, FP32_);
  nntrainer::Tensor m(1, 3, 4, 5, NHWC_, FP32_);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, false, true));
}

TEST(nntrainer_Tensor, dot_03_nhwc_p) {
  nntrainer::Tensor input(1, 3, 4, 5, NHWC_, FP32_);
  nntrainer::Tensor m(1, 3, 4, 5, NHWC_, FP32_);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, true));
}

TEST(nntrainer_Tensor, dot_04_nhwc_n) {
  nntrainer::Tensor input(2, 4, 5, 3, NHWC_, FP32_);
  nntrainer::Tensor m(1, 4, 5, 1, NHWC_, FP32_);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m), std::runtime_error);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, false, true));
}

TEST(nntrainer_Tensor, dot_05_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 2;
  int channel = 3;
  int height = 4;
  int width = 5;
  float ans[2][4][5][40] = {0};

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);

  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * (channel) + l + 1);

  nntrainer::Tensor weight(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(weight, i * (height * width * channel) +
                                j * (width * channel) + k * (channel) + l + 1);

  weight.reshape(nntrainer::TensorDim(1, 3, 8, 5, NHWC_, FP32_));

  nntrainer::Tensor result = input.dot(weight, false, true);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int k = 0; k < batch * height * width; k++) {
          ans[b][h][w][k] = 0;
          for (int c = 0; c < channel; c++) {
            float val1 = input.getValue(b, c, h, w);
            float val2 = weight.getValue(0, c, 0, k);
            ans[b][h][w][k] += val1 * val2;
          }
        }
      }
    }
  }

  for (unsigned int i = 0; i < result.batch(); ++i) {
    for (unsigned int j = 0; j < result.height(); ++j) {
      for (unsigned int k = 0; k < result.width(); ++k) {
        for (unsigned int c = 0; c < result.channel(); ++c) {
          float val1 = ans[i][j][k][c];
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

TEST(nntrainer_Tensor, dot_06_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 1;
  int width = 1;
  float ans[3][1][1][3] = {
    {{{30, 36, 42}}}, {{{66, 81, 96}}}, {{{102, 126, 150}}}};

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * (channel) + l + 1);

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

TEST(nntrainer_Tensor, dot_transpose_nhwc_p) {
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
}

TEST(nntrainer_Tensor, dot_shortcuts_nhwc_p) {
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 4, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 4, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 4, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 2, 3, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 3, 1, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    float a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, NHWC_, FP32_), a_data);
    float b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 3, 2, 1, NHWC_, FP32_), b_data);
    float answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 2, 1, 1, NHWC_, FP32_),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
}

TEST(nntrainer_Tensor, empty_nhwc_01) {
  nntrainer::Tensor t;

  EXPECT_TRUE(t.empty());
}

TEST(nntrainer_Tensor, empty_nhwc_02) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), false);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, empty_nhwc_03) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), true);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, fill_p) {
  /// same dimension, buffer size
  {
    nntrainer::Tensor target(3, 2, 4, 5, NHWC_, FP32_);
    nntrainer::Tensor original =
      randUniform(3, 2, 4, 5, -1.0f, 1.0f, NHWC_, FP32_);
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
    nntrainer::Tensor original =
      randUniform(3, 2, 4, 5, -1.0f, 1.0f, NHWC_, FP32_);
    target.fill(original, true);

    EXPECT_EQ(target, original);
  }
}

TEST(nntrainer_Tensor, fill_uninitialized_n) {
  nntrainer::Tensor target;
  nntrainer::Tensor original =
    randUniform(3, 1, 2, 3, -1.0f, 1.0f, NHWC_, FP32_);
  EXPECT_THROW(target.fill(original, false), std::invalid_argument);
}

TEST(nntrainer_Tensor, fill_different_dimension_n) {
  nntrainer::Tensor target(3, 1, 3, 2, NHWC_, FP32_);
  nntrainer::Tensor original =
    randUniform(3, 1, 2, 3, -1.0f, 1.0f, NHWC_, FP32_);
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

TEST(nntrainer_Tensor, add_strided_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);
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

TEST(nntrainer_Tensor, add_strided_02_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1, channel, NHWC_,
                         FP32_);

  EXPECT_THROW({ input.add_strided(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, add_strided_03_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, height, width, channel, NHWC_, FP32_);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + 1);

  EXPECT_THROW(input.add_strided(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_strided_04_nhwc_n) {
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, height, width, channel, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);

  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + 2);

  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.add_strided(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_strided_05_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 3;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);
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

TEST(nntrainer_Tensor, multiply_strided_01_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (batch * height * width) +
                               j * (height * width) + k + 1);

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

TEST(nntrainer_Tensor, multiply_strided_02_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (height * width) + j * (height * width) +
                               k * width + l);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1, NHWC_, FP32_);

  EXPECT_THROW({ input.multiply_strided(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_03_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);
  // input is not allocated now : alloc_now == false
  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + l);

  EXPECT_THROW(input.multiply_strided(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_04_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + l);
  // test is not allocated.
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.multiply_strided(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_05_nhwc_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width, NHWC_, FP32_);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT_NHWC(input, i * (height * width * channel) +
                               j * (width * channel) + k * channel + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT_NHWC(test, i * (height * width * channel) +
                              j * (width * channel) + k * channel + 2);
  // output is not aloocated
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.multiply_strided(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_06_nhwc_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;
  const int size = 90;

  nntrainer::Tensor input(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(input, i * (batch * height * width) +
                               j * (height * width) + k + 1);

  nntrainer::Tensor output(batch, channel, height, width, NHWC_, FP32_);
  GEN_TEST_INPUT_NHWC(output, i * (batch * height * width) +
                                j * (height * width) + k + 1);

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

TEST(nntrainer_Tensor, allocate_01_nhwc_n) {
  nntrainer::Tensor t;
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_FALSE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_02_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), false);
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_03_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), true);
  EXPECT_TRUE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, initialize_01_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), true,
                      nntrainer::Initializer::ONES);

  nntrainer::Tensor golden(1, 2, 3, 4, NHWC_, FP32_);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_02_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), true);

  nntrainer::Tensor golden(1, 2, 3, 4, NHWC_, FP32_);
  golden.setValue(1);

  EXPECT_NE(golden, t);

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_03_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), false,
                      nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4, NHWC_, FP32_);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_04_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), false);
  t.initialize(nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4, NHWC_, FP32_);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_05_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), false);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4, NHWC_, FP32_);
  golden.setValue(1.f);

  /**
   * Ideally, it should be NE, but it can be equal due to no initialization
   * EXPECT_NE(golden, t);
   */

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_06_nhwc_n) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), true,
                      nntrainer::Initializer::ONES);
  nntrainer::Tensor golden(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), true,
                           nntrainer::Initializer::ZEROS);

  EXPECT_NE(golden, t);

  golden.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_07_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), true,
                      nntrainer::Initializer::ONES);

  nntrainer::Tensor golden(1, 2, 3, 4, NHWC_, FP32_);
  golden.setValue(1);

  EXPECT_EQ(golden, t);

  t.setValue(0, 0, 0, 0, 0);
  t.setValue(0, t.size() - 1, 0, 0, 0);
  EXPECT_NE(golden, t);

  t.initialize();
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_08_nhwc_p) {
  nntrainer::Tensor t(nntrainer::TensorDim(1, 2, 3, 4, NHWC_, FP32_), true,
                      nntrainer::Initializer::ONES);

  nntrainer::Tensor golden(1, 2, 3, 4, NHWC_, FP32_);
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

TEST(nntrainer_Tensor, reshape_01_nhwc_n) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6, NHWC_, FP32_);

  EXPECT_THROW(A.reshape(nntrainer::TensorDim(9, 9, 9, 9, NHWC_, FP32_)),
               std::invalid_argument);
}

TEST(nntrainer_Tensor, reshape_02_nhwc_p) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6, NHWC_, FP32_);
  nntrainer::TensorDim A_dim = A.getDim();

  /** Changing the dim of a tensor only affects local copy of the dim */
  A_dim.setTensorDim(1, 100);
  EXPECT_EQ(A_dim.getTensorDim(1), 100u);

  nntrainer::TensorDim A_dim_2 = A.getDim();
  EXPECT_EQ(A_dim_2.getTensorDim(1), 4u);
}

TEST(nntrainer_Tensor, save_read_01_nhwc_p) {
  int batch = 3;
  int channel = 4;
  int height = 5;
  int width = 6;
  nntrainer::Tensor target(3, 4, 5, 6, NHWC_, FP32_);
  nntrainer::Tensor readed(3, 4, 5, 6, NHWC_, FP32_);

  GEN_TEST_INPUT_NHWC(target, i * (height * width * channel) +
                                j * (width * channel) + k * channel + 1);

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

TEST(nntrainer_Tensor, save_read_02_nhwc_n) {
  int batch = 3;
  int channel = 4;
  int height = 5;
  int width = 6;
  nntrainer::Tensor target(3, 4, 5, 6, NHWC_, FP32_);
  nntrainer::Tensor readed(3, 4, 1, 1, NHWC_, FP32_);

  GEN_TEST_INPUT_NHWC(target, i * (height * width * channel) +
                                j * (width * channel) + k * channel + 1);

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

TEST(nntrainer_Tensor, set_01_nhwc_p) {
  nntrainer::Tensor tensor = nntrainer::Tensor(1, 1, 1, 1, NHWC_, FP32_);

  tensor.setZero();
  EXPECT_EQ(tensor.getValue(0, 0, 0, 0), 0.0);

  tensor.setRandUniform(-0.5, 0);
  float val = tensor.getValue(0, 0, 0, 0);
  EXPECT_TRUE(val >= -0.5 && val < 0);
}

TEST(nntrainer_Tensor, print_small_size_nhwc_p) {
  nntrainer::Tensor target = constant(1.0, 3, 3, 1, 2, NHWC_, FP32_);

  std::stringstream ss, expected;
  ss << target;

  expected << '<' << typeid(target).name() << " at " << &target << ">\n"
           << "data addr: " << target.getData() << '\n'
           << "Shape: 3:3:1:2 [ FP32 : NHWC ]\n"
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

TEST(nntrainer_Tensor, copy_and_reshape_nhwc_n) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6, NHWC_, FP32_);
  nntrainer::Tensor B = A;
  nntrainer::Tensor C = A.clone();

  EXPECT_THROW(B.reshape(nntrainer::TensorDim(9, 9, 9, 9, NHWC_, FP32_)),
               std::invalid_argument);
}

TEST(nntrainer_Tensor, copy_and_shares_variable_nhwc_p) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6, NHWC_, FP32_);
  nntrainer::Tensor B = A.clone();
  nntrainer::Tensor C = A;

  C.setValue(1, 1, 1, 1, 2.0f);

  EXPECT_EQ(A, C);
  EXPECT_NE(B, C);

  C.reshape(nntrainer::TensorDim(3, 4, 6, 5, NHWC_, FP32_));
  EXPECT_EQ(A.getDim(), B.getDim());
  EXPECT_NE(A.getDim(), C.getDim());
}

TEST(nntrainer_Tensor, cat_01_nhwc_p) {
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(ranged(2, 2, 1, 1, NHWC_, FP32_));
    inputs.emplace_back(ranged(2, 2, 2, 1, NHWC_, FP32_));
    float answer_data[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 6, 7};
    nntrainer::Tensor answer(ml::train::TensorDim({2, 2, 3, 1}, {NHWC_, FP32_}),
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 2), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(ranged(3, 2, 4, 5, NHWC_, FP32_));
    inputs.emplace_back(ranged(2, 2, 4, 5, NHWC_, FP32_));
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
    nntrainer::Tensor answer(ml::train::TensorDim({5, 2, 4, 5}, {NHWC_, FP32_}),
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 0), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(ranged(3, 5, 3, 4, NHWC_, FP32_));
    inputs.emplace_back(ranged(3, 5, 2, 4, NHWC_, FP32_));
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
    nntrainer::Tensor answer(ml::train::TensorDim({3, 5, 5, 4}, {NHWC_, FP32_}),
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 2), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(ranged(3, 5, 2, 1, NHWC_, FP32_));
    inputs.emplace_back(ranged(3, 5, 2, 2, NHWC_, FP32_));
    float answer_data[] = {
      0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,
      8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44, 45, 46,
      47, 48, 49, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
    nntrainer::Tensor answer(ml::train::TensorDim({3, 5, 2, 3}, {NHWC_, FP32_}),
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 3), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(3);
    inputs.emplace_back(ranged(3, 1, 2, 4, NHWC_, FP32_));
    inputs.emplace_back(ranged(3, 3, 2, 4, NHWC_, FP32_));
    inputs.emplace_back(ranged(3, 2, 2, 4, NHWC_, FP32_));
    float answer_data[] = {
      0,  0,  1,  2,  0,  1,  1,  3,  4,  5,  2,  3,  2,  6,  7,  8,  4,  5,
      3,  9,  10, 11, 6,  7,  4,  12, 13, 14, 8,  9,  5,  15, 16, 17, 10, 11,
      6,  18, 19, 20, 12, 13, 7,  21, 22, 23, 14, 15, 8,  24, 25, 26, 16, 17,
      9,  27, 28, 29, 18, 19, 10, 30, 31, 32, 20, 21, 11, 33, 34, 35, 22, 23,
      12, 36, 37, 38, 24, 25, 13, 39, 40, 41, 26, 27, 14, 42, 43, 44, 28, 29,
      15, 45, 46, 47, 30, 31, 16, 48, 49, 50, 32, 33, 17, 51, 52, 53, 34, 35,
      18, 54, 55, 56, 36, 37, 19, 57, 58, 59, 38, 39, 20, 60, 61, 62, 40, 41,
      21, 63, 64, 65, 42, 43, 22, 66, 67, 68, 44, 45, 23, 69, 70, 71, 46, 47};
    nntrainer::Tensor answer(ml::train::TensorDim({3, 6, 2, 4}, {NHWC_, FP32_}),
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 1), answer);
  }
}

TEST(nntrainer_Tensor, cat_02_nhwc_n) {
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(nntrainer::Tensor(2, 1, 1, 2, NHWC_, FP32_));
    inputs.emplace_back(nntrainer::Tensor(2, 2, 1, 2, NHWC_, FP32_));
    EXPECT_THROW(nntrainer::Tensor::cat(inputs, 2), std::invalid_argument);
  }
}

TEST(nntrainer_Tensor, TensorMap_nhwc_p) {
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

TEST(nntrainer_Tensor, TensorWrap_01_nhwc_n) {
  float dat[] = {1, 2, 3};
  EXPECT_THROW(nntrainer::Tensor::Map(dat, 3, nntrainer::TensorDim({})),
               std::invalid_argument);
}

TEST(nntrainer_Tensor, TensorWrap_02_nhwc_n) {
  float dat[] = {1, 2, 3};
  EXPECT_THROW(nntrainer::Tensor::Map(dat, 3, {4}), std::invalid_argument);
}

// TEST(nntrainer_Tensor, TensorPaddedValue_nhwc_p) {
//   nntrainer::Tensor a = ranged(1, 1, 3, 3, NHWC_, FP32_);
//   float default_padded = -1;

//   for (int i = 0; i < 5; ++i) {
//     for (int j = 0; j < 5; ++j) {
//       float expected = default_padded;
//       if (1 <= i && i <= 3 && 1 <= j && j <= 3) {
//         expected = (i - 1) * 3 + (j - 1);
//       }
//       float actual = a.getValuePaddedVirtual(0, 0, i, j, 1, 1,
//       default_padded); EXPECT_FLOAT_EQ(actual, expected);
//     }
//   }
// }

TEST(nntrainer_Tensor, zoneout_mask_01_nhwc_n) {
  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 10, 10, NHWC_, FP32_);
  nntrainer::Tensor opposite(20, 20, 20, 20, NHWC_, FP32_);
  EXPECT_THROW(t.zoneout_mask(opposite, zoneout_rate), std::invalid_argument);
}

TEST(nntrainer_Tensor, zoneout_mask_02_nhwc_p) {
  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 10, 10, NHWC_, FP32_);
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

TEST(nntrainer_Tensor, zoneout_mask_03_nhwc_p) {
  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 100, 100, NHWC_, FP32_);
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

TEST(nntrainer_Tensor, zoneout_mask_04_nhwc_n) {
  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 100, 100, NHWC_, FP32_);
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

TEST(nntrainer_Tensor, split_01_nhwc_p) {
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(3);
    {
      float answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                             30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
      answer.emplace_back(ml::train::TensorDim({1, 5, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                             70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
      answer.emplace_back(ml::train::TensorDim({1, 5, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                             90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(ml::train::TensorDim({1, 5, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split(3, 0), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      float answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                             56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87,
                             88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
      answer.emplace_back(ml::train::TensorDim({3, 5, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                             30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                             60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                             70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(ml::train::TensorDim({3, 5, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split(2, 2), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      float answer_data[] = {
        0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   20,  21,  22,  23,  24,
        25, 26, 27, 28, 29, 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
        60, 61, 62, 63, 64, 65,  66,  67,  68,  69,  80,  81,  82,  83,  84,
        85, 86, 87, 88, 89, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
      answer.emplace_back(ml::train::TensorDim({3, 5, 2, 2}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {
        10, 11, 12, 13, 14, 15,  16,  17,  18,  19,  30,  31,  32,  33,  34,
        35, 36, 37, 38, 39, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
        70, 71, 72, 73, 74, 75,  76,  77,  78,  79,  90,  91,  92,  93,  94,
        95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(ml::train::TensorDim({3, 5, 2, 2}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split(2, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(5);
    {
      float answer_data[] = {0,  5,  10, 15, 20,  25,  30,  35,
                             40, 45, 50, 55, 60,  65,  70,  75,
                             80, 85, 90, 95, 100, 105, 110, 115};
      answer.emplace_back(ml::train::TensorDim({3, 1, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {1,  6,  11, 16, 21,  26,  31,  36,
                             41, 46, 51, 56, 61,  66,  71,  76,
                             81, 86, 91, 96, 101, 106, 111, 116};
      answer.emplace_back(ml::train::TensorDim({3, 1, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {2,  7,  12, 17, 22,  27,  32,  37,
                             42, 47, 52, 57, 62,  67,  72,  77,
                             82, 87, 92, 97, 102, 107, 112, 117};
      answer.emplace_back(ml::train::TensorDim({3, 1, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {3,  8,  13, 18, 23,  28,  33,  38,
                             43, 48, 53, 58, 63,  68,  73,  78,
                             83, 88, 93, 98, 103, 108, 113, 118};
      answer.emplace_back(ml::train::TensorDim({3, 1, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.emplace_back(ml::train::TensorDim({3, 1, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split(5, 1), answer);
  }
  {
    nntrainer::TensorDim ref_dim(1, 6, 1, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(1, 6, 1, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      float answer_data[] = {0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20};
      answer.emplace_back(ml::train::TensorDim({1, 3, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23};
      answer.emplace_back(ml::train::TensorDim({1, 3, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split(2, 1), answer);
  }
}

TEST(nntrainer_Tensor, split_02_nhwc_n) {
  nntrainer::Tensor t(1, 1, 1, 1, NHWC_, FP32_);
  EXPECT_THROW(t.split(0, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_03_nhwc_n) {
  nntrainer::Tensor t(3, 1, 1, 1, NHWC_, FP32_);
  EXPECT_THROW(t.split(2, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_04_nhwc_p) {
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      float answer_data[] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
      answer.emplace_back(ml::train::TensorDim({2, 5, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                             90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(ml::train::TensorDim({1, 5, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split({2, 1}, 0), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      float answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                             56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87,
                             88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
      answer.emplace_back(ml::train::TensorDim({3, 5, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                             30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                             60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                             70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(ml::train::TensorDim({3, 5, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split({1, 1}, 2), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      float answer_data[] = {
        0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   20,  21,  22,  23,  24,
        25, 26, 27, 28, 29, 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
        60, 61, 62, 63, 64, 65,  66,  67,  68,  69,  80,  81,  82,  83,  84,
        85, 86, 87, 88, 89, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
      answer.emplace_back(ml::train::TensorDim({3, 5, 2, 2}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {
        10, 11, 12, 13, 14, 15,  16,  17,  18,  19,  30,  31,  32,  33,  34,
        35, 36, 37, 38, 39, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
        70, 71, 72, 73, 74, 75,  76,  77,  78,  79,  90,  91,  92,  93,  94,
        95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(ml::train::TensorDim({3, 5, 2, 2}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split({2, 2}, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(3);
    {
      float answer_data[] = {0,  5,  10, 15, 20,  25,  30,  35,
                             40, 45, 50, 55, 60,  65,  70,  75,
                             80, 85, 90, 95, 100, 105, 110, 115};
      answer.emplace_back(ml::train::TensorDim({3, 1, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {
        1,   2,   3,   6,   7,   8,   11,  12,  13,  16,  17,  18, 21, 22, 23,
        26,  27,  28,  31,  32,  33,  36,  37,  38,  41,  42,  43, 46, 47, 48,
        51,  52,  53,  56,  57,  58,  61,  62,  63,  66,  67,  68, 71, 72, 73,
        76,  77,  78,  81,  82,  83,  86,  87,  88,  91,  92,  93, 96, 97, 98,
        101, 102, 103, 106, 107, 108, 111, 112, 113, 116, 117, 118};
      answer.emplace_back(ml::train::TensorDim({3, 3, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.emplace_back(ml::train::TensorDim({3, 1, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split({1, 3, 1}, 1), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(3);
    {
      float answer_data[] = {
        0,  1,  5,  6,  10, 11, 15, 16, 20,  21,  25,  26,  30,  31,  35,  36,
        40, 41, 45, 46, 50, 51, 55, 56, 60,  61,  65,  66,  70,  71,  75,  76,
        80, 81, 85, 86, 90, 91, 95, 96, 100, 101, 105, 106, 110, 111, 115, 116};
      answer.emplace_back(ml::train::TensorDim({3, 2, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {
        2,  3,  7,  8,  12, 13, 17, 18, 22,  23,  27,  28,  32,  33,  37,  38,
        42, 43, 47, 48, 52, 53, 57, 58, 62,  63,  67,  68,  72,  73,  77,  78,
        82, 83, 87, 88, 92, 93, 97, 98, 102, 103, 107, 108, 112, 113, 117, 118};
      answer.emplace_back(ml::train::TensorDim({3, 2, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.emplace_back(ml::train::TensorDim({3, 1, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split({2, 2, 1}, 1), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      float answer_data[] = {
        0,  1,  5,  6,  10, 11, 15, 16, 20,  21,  25,  26,  30,  31,  35,  36,
        40, 41, 45, 46, 50, 51, 55, 56, 60,  61,  65,  66,  70,  71,  75,  76,
        80, 81, 85, 86, 90, 91, 95, 96, 100, 101, 105, 106, 110, 111, 115, 116};
      answer.emplace_back(ml::train::TensorDim({3, 2, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {
        2,   3,   4,   7,   8,   9,   12,  13,  14,  17,  18,  19, 22, 23, 24,
        27,  28,  29,  32,  33,  34,  37,  38,  39,  42,  43,  44, 47, 48, 49,
        52,  53,  54,  57,  58,  59,  62,  63,  64,  67,  68,  69, 72, 73, 74,
        77,  78,  79,  82,  83,  84,  87,  88,  89,  92,  93,  94, 97, 98, 99,
        102, 103, 104, 107, 108, 109, 112, 113, 114, 117, 118, 119};
      answer.emplace_back(ml::train::TensorDim({3, 3, 2, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split({2, 3}, 1), answer);
  }
  {
    nntrainer::TensorDim ref_dim(1, 6, 1, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(1, 6, 1, 4, NHWC_, FP32_);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(3);
    {
      float answer_data[] = {0, 6, 12, 18};
      answer.emplace_back(ml::train::TensorDim({1, 1, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21};
      answer.emplace_back(ml::train::TensorDim({1, 3, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    {
      float answer_data[] = {4, 5, 10, 11, 16, 17, 22, 23};
      answer.emplace_back(ml::train::TensorDim({1, 2, 1, 4}, {NHWC_, FP32_}),
                          answer_data);
    }
    EXPECT_EQ(t.split({1, 3, 2}, 1), answer);
  }
}

TEST(nntrainer_Tensor, split_05_nhwc_n) {
  nntrainer::Tensor t(3, 1, 1, 1, NHWC_, FP32_);
  EXPECT_THROW(t.split({1, 1}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_06_nhwc_n) {
  nntrainer::Tensor t(3, 1, 1, 1, NHWC_, FP32_);
  EXPECT_THROW(t.split({2, 0, 1}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_07_nhwc_n) {
  nntrainer::Tensor t(3, 1, 1, 1, NHWC_, FP32_);
  EXPECT_THROW(t.split({}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, transpose_nhwc_p) {
  nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);

  /// plain transpose
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
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
    nntrainer::Tensor answer(ml::train::TensorDim({3, 5, 2, 4}, {NHWC_, FP32_}),
                             answer_data);
    nntrainer::Tensor m = t.transpose("0:1:2");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
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
    nntrainer::Tensor answer(ml::train::TensorDim({3, 4, 2, 5}, {NHWC_, FP32_}),
                             answer_data);
    nntrainer::Tensor m = t.transpose("2:1:0");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
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
    nntrainer::Tensor answer(ml::train::TensorDim({3, 5, 4, 2}, {NHWC_, FP32_}),
                             answer_data);
    nntrainer::Tensor m = t.transpose("0:2:1");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    float answer_data[] = {
      0,  20,  1,  21,  2,  22,  3,  23,  4,  24,  5,  25,  6,  26,  7,  27,
      8,  28,  9,  29,  10, 30,  11, 31,  12, 32,  13, 33,  14, 34,  15, 35,
      16, 36,  17, 37,  18, 38,  19, 39,  40, 60,  41, 61,  42, 62,  43, 63,
      44, 64,  45, 65,  46, 66,  47, 67,  48, 68,  49, 69,  50, 70,  51, 71,
      52, 72,  53, 73,  54, 74,  55, 75,  56, 76,  57, 77,  58, 78,  59, 79,
      80, 100, 81, 101, 82, 102, 83, 103, 84, 104, 85, 105, 86, 106, 87, 107,
      88, 108, 89, 109, 90, 110, 91, 111, 92, 112, 93, 113, 94, 114, 95, 115,
      96, 116, 97, 117, 98, 118, 99, 119};
    nntrainer::Tensor answer(ml::train::TensorDim({3, 2, 4, 5}, {NHWC_, FP32_}),
                             answer_data);
    nntrainer::Tensor m = t.transpose("1:2:0");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    float answer_data[] = {
      0,  5,  10,  15,  20,  25,  30,  35, 1,  6,   11,  16,  21,  26,  31,
      36, 2,  7,   12,  17,  22,  27,  32, 37, 3,   8,   13,  18,  23,  28,
      33, 38, 4,   9,   14,  19,  24,  29, 34, 39,  40,  45,  50,  55,  60,
      65, 70, 75,  41,  46,  51,  56,  61, 66, 71,  76,  42,  47,  52,  57,
      62, 67, 72,  77,  43,  48,  53,  58, 63, 68,  73,  78,  44,  49,  54,
      59, 64, 69,  74,  79,  80,  85,  90, 95, 100, 105, 110, 115, 81,  86,
      91, 96, 101, 106, 111, 116, 82,  87, 92, 97,  102, 107, 112, 117, 83,
      88, 93, 98,  103, 108, 113, 118, 84, 89, 94,  99,  104, 109, 114, 119};
    nntrainer::Tensor answer(ml::train::TensorDim({3, 4, 5, 2}, {NHWC_, FP32_}),
                             answer_data);
    nntrainer::Tensor m = t.transpose("2:0:1");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    float answer_data[] = {
      0,  20,  5,  25,  10, 30,  15, 35,  1,  21,  6,  26,  11, 31,  16, 36,
      2,  22,  7,  27,  12, 32,  17, 37,  3,  23,  8,  28,  13, 33,  18, 38,
      4,  24,  9,  29,  14, 34,  19, 39,  40, 60,  45, 65,  50, 70,  55, 75,
      41, 61,  46, 66,  51, 71,  56, 76,  42, 62,  47, 67,  52, 72,  57, 77,
      43, 63,  48, 68,  53, 73,  58, 78,  44, 64,  49, 69,  54, 74,  59, 79,
      80, 100, 85, 105, 90, 110, 95, 115, 81, 101, 86, 106, 91, 111, 96, 116,
      82, 102, 87, 107, 92, 112, 97, 117, 83, 103, 88, 108, 93, 113, 98, 118,
      84, 104, 89, 109, 94, 114, 99, 119};
    nntrainer::Tensor answer(ml::train::TensorDim({3, 2, 5, 4}, {NHWC_, FP32_}),
                             answer_data);
    nntrainer::Tensor m = t.transpose("1:0:2");
    EXPECT_EQ(answer, m);
  }

  /// outplace transpose
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 5, 2, 4, NHWC_, FP32_);
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
    nntrainer::Tensor answer(ml::train::TensorDim({3, 5, 2, 4}, {NHWC_, FP32_}),
                             answer_data);
    t.transpose("0:1:2", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 4, 2, 5, NHWC_, FP32_);
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
    nntrainer::Tensor answer(ml::train::TensorDim({3, 4, 2, 5}, {NHWC_, FP32_}),
                             answer_data);
    t.transpose("2:1:0", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 5, 4, 2, NHWC_, FP32_);
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
    nntrainer::Tensor answer(ml::train::TensorDim({3, 5, 4, 2}, {NHWC_, FP32_}),
                             answer_data);
    t.transpose("0:2:1", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 2, 4, 5, NHWC_, FP32_);
    float answer_data[] = {
      0,  20,  1,  21,  2,  22,  3,  23,  4,  24,  5,  25,  6,  26,  7,  27,
      8,  28,  9,  29,  10, 30,  11, 31,  12, 32,  13, 33,  14, 34,  15, 35,
      16, 36,  17, 37,  18, 38,  19, 39,  40, 60,  41, 61,  42, 62,  43, 63,
      44, 64,  45, 65,  46, 66,  47, 67,  48, 68,  49, 69,  50, 70,  51, 71,
      52, 72,  53, 73,  54, 74,  55, 75,  56, 76,  57, 77,  58, 78,  59, 79,
      80, 100, 81, 101, 82, 102, 83, 103, 84, 104, 85, 105, 86, 106, 87, 107,
      88, 108, 89, 109, 90, 110, 91, 111, 92, 112, 93, 113, 94, 114, 95, 115,
      96, 116, 97, 117, 98, 118, 99, 119};
    nntrainer::Tensor answer(ml::train::TensorDim({3, 2, 4, 5}, {NHWC_, FP32_}),
                             answer_data);
    t.transpose("1:2:0", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 4, 5, 2, NHWC_, FP32_);
    float answer_data[] = {
      0,  5,  10,  15,  20,  25,  30,  35, 1,  6,   11,  16,  21,  26,  31,
      36, 2,  7,   12,  17,  22,  27,  32, 37, 3,   8,   13,  18,  23,  28,
      33, 38, 4,   9,   14,  19,  24,  29, 34, 39,  40,  45,  50,  55,  60,
      65, 70, 75,  41,  46,  51,  56,  61, 66, 71,  76,  42,  47,  52,  57,
      62, 67, 72,  77,  43,  48,  53,  58, 63, 68,  73,  78,  44,  49,  54,
      59, 64, 69,  74,  79,  80,  85,  90, 95, 100, 105, 110, 115, 81,  86,
      91, 96, 101, 106, 111, 116, 82,  87, 92, 97,  102, 107, 112, 117, 83,
      88, 93, 98,  103, 108, 113, 118, 84, 89, 94,  99,  104, 109, 114, 119};
    nntrainer::Tensor answer(ml::train::TensorDim({3, 4, 5, 2}, {NHWC_, FP32_}),
                             answer_data);
    t.transpose("2:0:1", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor t = ranged(3, 5, 2, 4, NHWC_, FP32_);
    nntrainer::Tensor m = ranged(3, 2, 5, 4, NHWC_, FP32_);
    float answer_data[] = {
      0,  20,  5,  25,  10, 30,  15, 35,  1,  21,  6,  26,  11, 31,  16, 36,
      2,  22,  7,  27,  12, 32,  17, 37,  3,  23,  8,  28,  13, 33,  18, 38,
      4,  24,  9,  29,  14, 34,  19, 39,  40, 60,  45, 65,  50, 70,  55, 75,
      41, 61,  46, 66,  51, 71,  56, 76,  42, 62,  47, 67,  52, 72,  57, 77,
      43, 63,  48, 68,  53, 73,  58, 78,  44, 64,  49, 69,  54, 74,  59, 79,
      80, 100, 85, 105, 90, 110, 95, 115, 81, 101, 86, 106, 91, 111, 96, 116,
      82, 102, 87, 107, 92, 112, 97, 117, 83, 103, 88, 108, 93, 113, 98, 118,
      84, 104, 89, 109, 94, 114, 99, 119};
    nntrainer::Tensor answer(ml::train::TensorDim({3, 2, 5, 4}, {NHWC_, FP32_}),
                             answer_data);
    t.transpose("1:0:2", m);
    EXPECT_EQ(answer, m);
  }
}

TEST(nntrainer_Tensor, tranpose_dimension_not_match_nhwc_01_n) {
  nntrainer::Tensor a(3, 5, 2, 4, NHWC_, FP32_);
  nntrainer::Tensor b(3, 3, 1, 2, NHWC_, FP32_);

  EXPECT_THROW(a.transpose("0:1:2", b), std::invalid_argument);
}

TEST(nntrainer_Tensor, tranpose_dimension_not_match_nhwc_02_n) {
  nntrainer::Tensor a(3, 5, 2, 4, NHWC_, FP32_);
  nntrainer::Tensor b(3, 3, 1, 2, NHWC_, FP32_);

  EXPECT_THROW(a.transpose("0:1", b), std::invalid_argument);
}

TEST(nntrainer_Tensor, tranpose_dimension_not_match_nhwc_03_n) {
  nntrainer::Tensor a(3, 5, 2, 4, NHWC_, FP32_);
  nntrainer::Tensor b(3, 3, 1, 2, NHWC_, FP32_);

  EXPECT_THROW(a.transpose("1:2:3:4", b), std::invalid_argument);
}

TEST(nntrainer_Tensor, tranpose_invalid_format_01_n) {
  nntrainer::Tensor a(3, 5, 2, 4, NHWC_, FP32_);
  nntrainer::Tensor b(3, 3, 1, 2, NHWC_, FP32_);

  EXPECT_THROW(a.transpose("1<->4", b), std::invalid_argument);
}

TEST(nntrainer_Tensor, tranpose_invalid_format_02_n) {
  nntrainer::Tensor a(3, 5, 2, 4, NHWC_, FP32_);
  nntrainer::Tensor b(3, 3, 1, 2, NHWC_, FP32_);

  EXPECT_THROW(a.transpose("2,0,1,3", b), std::invalid_argument);
}

TEST(nntrainer_Tensor, tranpose_invalid_format_03_n) {
  nntrainer::Tensor a(3, 5, 2, 4, NHWC_, FP32_);
  nntrainer::Tensor b(3, 3, 1, 2, NHWC_, FP32_);

  EXPECT_THROW(a.transpose("2-0-1-3", b), std::invalid_argument);
}

TEST(nntrainer_Tensor, tranpose_invalid_format_04_n) {
  nntrainer::Tensor a(3, 5, 2, 4, NHWC_, FP32_);
  nntrainer::Tensor b(3, 3, 1, 2, NHWC_, FP32_);

  EXPECT_THROW(a.transpose("2/0/1/3", b), std::invalid_argument);
}

// /**
//  * @brief dequantize tensor with different format
//  */
// TEST(nntrainer_Tensor, dequantize_01_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);
//   input.setScaleFactors({1.5, 1.0, 0.5});
//   input.setZeroPoints({1, 0, 3});

//   nntrainer::Tensor output(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NHWC, nntrainer::Tdatatype::FP32});

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

// /**
//  * @brief dequantize tensor with different format
//  */
// TEST(nntrainer_Tensor, dequantize_02_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NHWC, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);
//   input.setScaleFactors({1.5, 1.0, 0.5});
//   input.setZeroPoints({1, 0, 3});

//   nntrainer::Tensor output(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

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
