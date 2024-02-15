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
  ASSERT_NE(nullptr, tensor.getData<_FP16>());
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
  ASSERT_NE(nullptr, tensor.getData<_FP16>());

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
  ASSERT_NE(nullptr, tensor.getData<_FP16>());

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
  ASSERT_NE(nullptr, tensor.getData<_FP16>());

  if (tensor.getValue<_FP16>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_04_p) {
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
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::TensorV2 t0 = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});

  // copy assignment operator
  nntrainer::TensorV2 t1 = t0;

  if (t1.getValue<_FP16>(0, 0, 0, 1) != 1.0)
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
  std::vector<std::vector<std::vector<_FP16>>> in;

  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<_FP16>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<_FP16> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::TensorV2 t0 = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});

  // copy assignment operator
  nntrainer::TensorV2 t1 = nntrainer::TensorV2(
    batch, height, width, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  t1.setRandNormal(2.3, 0.5);

  _FP16 val_t0 = t0.getValue<_FP16>(0, 0, 0, 1);
  _FP16 val_t1 = t1.getValue<_FP16>(0, 0, 0, 1);

  swap(t0, t1);

  if (t0.getValue<_FP16>(0, 0, 0, 1) != val_t1)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);

  if (t1.getValue<_FP16>(0, 0, 0, 1) != val_t0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_06_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  std::vector<std::vector<std::vector<float>>> in;
  std::vector<std::vector<std::vector<_FP16>>> in2;

  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<float>> ttv;
    std::vector<std::vector<_FP16>> ttv2;
    for (int i = 0; i < height; ++i) {
      std::vector<float> tv;
      std::vector<_FP16> tv2;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
        tv2.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
      ttv2.push_back(tv2);
    }
    in.push_back(ttv);
    in2.push_back(ttv2);
  }

  nntrainer::TensorV2 t0 = nntrainer::TensorV2(
    in, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  nntrainer::TensorV2 t1 = nntrainer::TensorV2(
    in2, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});

  EXPECT_NE(t0, t1);
}

TEST(nntrainer_Tensor, empty_01) {
  nntrainer::TensorV2 t("", nntrainer::Tformat::NCHW,
                        nntrainer::Tdatatype::FP16);

  EXPECT_TRUE(t.empty());
}

TEST(nntrainer_Tensor, empty_02) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    false);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, empty_03) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, allocate_01_n) {
  nntrainer::TensorV2 t;
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_FALSE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_02_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    false);
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_03_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true);
  EXPECT_TRUE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, initialize_01_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true, nntrainer::Initializer::ONES);

  nntrainer::TensorV2 golden(1, 2, 3, 4, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_02_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true);

  nntrainer::TensorV2 golden(1, 2, 3, 4, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  golden.setValue(1);

  EXPECT_NE(golden, t);

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_03_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    false, nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::TensorV2 golden(1, 2, 3, 4, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);

  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_04_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    false);
  t.initialize(nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::TensorV2 golden(1, 2, 3, 4, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  ;
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_05_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    false);
  t.allocate();

  nntrainer::TensorV2 golden(1, 2, 3, 4, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);

  golden.setValue(1.f);

  /**
   * Ideally, it should be NE, but it can be equal due to no initialization
   * EXPECT_NE(golden, t);
   */

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_06_n) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true, nntrainer::Initializer::ONES);
  nntrainer::TensorV2 golden(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true, nntrainer::Initializer::ZEROS);

  EXPECT_NE(golden, t);

  golden.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_07_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true, nntrainer::Initializer::ONES);

  nntrainer::TensorV2 golden(1, 2, 3, 4, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);

  golden.setValue(1);

  EXPECT_EQ(golden, t);

  t.setValue(0, 0, 0, 0, 0);
  t.setValue(0, 0, 0, t.size() - 1, 0);
  EXPECT_NE(golden, t);

  t.initialize();
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_08_p) {
  nntrainer::TensorV2 t(
    {{1, 2, 3, 4}, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true, nntrainer::Initializer::ONES);

  nntrainer::TensorV2 golden(1, 2, 3, 4, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);

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

TEST(nntrainer_Tensor, multiply_i_01_fp16_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 original;
  original.copy(input);

  status = input.multiply_i(2.0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *data = original.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i] + data[i], indata[i]);
  }
}

TEST(nntrainer_Tensor, multiply_i_02_fp16_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 original;
  original.copy(input);

  status = input.multiply_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *data = original.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i] * data[i], indata[i]);
  }
}

TEST(nntrainer_Tensor, multiply_i_03_fp16_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 target2(batch, channel, height - 2, width - 1,
                              nntrainer::Tformat::NCHW,
                              nntrainer::Tdatatype::FP16);
  status = input.multiply_i(target2);

  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_01_fp16_p) {
  unsigned int N = 120;
  _FP16 *answer_data = new _FP16[N];
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(1, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);

    float float_data[] = {
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

    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());

    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(3, 1, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(3, 2, 4, 1, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(1, 2, 1, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {
      0,   1,   4,    9,   16,  0,   6,   14,  24,  36,  0,   11,  24,  39,
      56,  0,   16,   34,  54,  76,  100, 126, 154, 184, 216, 125, 156, 189,
      224, 261, 150,  186, 224, 264, 306, 175, 216, 259, 304, 351, 0,   41,
      84,  129, 176,  0,   46,  94,  144, 196, 0,   51,  104, 159, 216, 0,
      56,  114, 174,  236, 300, 366, 434, 504, 576, 325, 396, 469, 544, 621,
      350, 426, 504,  584, 666, 375, 456, 539, 624, 711, 0,   81,  164, 249,
      336, 0,   86,   174, 264, 356, 0,   91,  184, 279, 376, 0,   96,  194,
      294, 396, 500,  606, 714, 824, 936, 525, 636, 749, 864, 981, 550, 666,
      784, 904, 1026, 575, 696, 819, 944, 1071};
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(3, 1, 4, 1, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(1, 1, 1, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {
      0, 1,   4,   9,   16,  0, 6,   14,  24,  36,  0, 11,  24,  39,  56,
      0, 16,  34,  54,  76,  0, 21,  44,  69,  96,  0, 26,  54,  84,  116,
      0, 31,  64,  99,  136, 0, 36,  74,  114, 156, 0, 41,  84,  129, 176,
      0, 46,  94,  144, 196, 0, 51,  104, 159, 216, 0, 56,  114, 174, 236,
      0, 61,  124, 189, 256, 0, 66,  134, 204, 276, 0, 71,  144, 219, 296,
      0, 76,  154, 234, 316, 0, 81,  164, 249, 336, 0, 86,  174, 264, 356,
      0, 91,  184, 279, 376, 0, 96,  194, 294, 396, 0, 101, 204, 309, 416,
      0, 106, 214, 324, 436, 0, 111, 224, 339, 456, 0, 116, 234, 354, 476};
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(1, 2, 1, 1, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119};
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 1, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  160, 162, 164, 166,
      168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194,
      196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222,
      224, 226, 228, 230, 232, 234, 236, 238};
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 5, 1, 4, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 4, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    float float_data[] = {0,   1,   4,   9,   0,   5,   12,  21,  0,   9,
                          20,  33,  0,   13,  28,  45,  0,   17,  36,  57,
                          80,  105, 132, 161, 96,  125, 156, 189, 112, 145,
                          180, 217, 128, 165, 204, 245, 144, 185, 228, 273,
                          320, 369, 420, 473, 352, 405, 460, 517, 384, 441,
                          500, 561, 416, 477, 540, 605, 448, 513, 580, 649};
    std::transform(float_data, float_data + 60, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  delete[] answer_data;
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_supported_01_n) {

  nntrainer::TensorV2 target(3, 1, 3, 1, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  nntrainer::TensorV2 target2(3, 1, 3, 3, nntrainer::Tformat::NCHW,
                              nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_broadcastable_02_n) {
  nntrainer::TensorV2 target(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  nntrainer::TensorV2 target2(3, 2, 3, 1, nntrainer::Tformat::NCHW,
                              nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 result = input.multiply(0.0);
  if (result.getValue<_FP16>(0, 0, 1, 1) != 0.0)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 result = input.multiply(input);

  _FP16 *data = result.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
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

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 test(batch - 1, height - 1, width - 1,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  EXPECT_THROW({ input.multiply(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(batch, channel, height, 2 * width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
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
  nntrainer::TensorV2 test(batch, channel, height, 2 * width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
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

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 expected(batch, channel, height, width,
                               nntrainer::Tformat::NCHW,
                               nntrainer::Tdatatype::FP16);
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

  nntrainer::TensorV2 input(
    batch, channel, height, width,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 result = input.multiply_strided(input);

  _FP16 *data = result.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  _FP16 *outdata = new _FP16[(input.size())];

  std::transform(indata, indata + batch * height * width * channel, indata,
                 outdata, std::multiplies<_FP16>());

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

  nntrainer::TensorV2 input(
    batch, channel, height, width,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 test(batch - 1, height - 1, width - 1);

  EXPECT_THROW({ input.multiply_strided(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_strided_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(
    batch, channel, height, width,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
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

  nntrainer::TensorDim dim(
    batch, channel, height, width,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});

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

  nntrainer::TensorDim dim(
    batch, channel, height, width,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});

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

  nntrainer::TensorV2 input(
    batch, channel, height, width,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 output(
    batch, channel, height, width,
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
  GEN_TEST_INPUT(output, i * (batch * height) + j * (width) + k + 1);

  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  _FP16 *outdata_beta = new _FP16[(input.size())];
  _FP16 *indata_mul = new _FP16[(input.size())];
  _FP16 *outdata = new _FP16[(input.size())];

  std::transform(indata, indata + batch * height * width * channel,
                 outdata_beta,
                 std::bind(std::multiplies<_FP16>(), std::placeholders::_1,
                           static_cast<_FP16>(10.0)));

  std::transform(indata, indata + batch * height * width * channel, indata,
                 indata_mul, std::multiplies<_FP16>());
  std::transform(indata_mul, indata_mul + batch * height * width * channel,
                 outdata_beta, outdata, std::plus<_FP16>());

  input.multiply_strided(input, output, 10.0);

  _FP16 *data = output.getData<_FP16>();
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

TEST(nntrainer_Tensor, divide_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 original;
  original.copy(input);

  status = input.divide_i(2.0f);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *data = original.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
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

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  status = input.divide_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(indata[i], _FP16(1.0));
  }
}

TEST(nntrainer_Tensor, divide_i_01_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  status = input.divide_i((_FP16)0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_i_02_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::TensorV2 original(batch, channel, height - 2, width - 1,
                               nntrainer::Tformat::NCHW,
                               nntrainer::Tdatatype::FP16);

  status = input.divide_i(original);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 result = input.divide(1.0);

  _FP16 *previous = input.getData<_FP16>();
  ASSERT_NE(nullptr, previous);
  _FP16 *data = result.getData<_FP16>();
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

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW({ input.divide(0.0); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorV2 input(batch, channel, height, width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::TensorV2 test(batch - 1, channel, height - 1, width - 1,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  EXPECT_THROW({ input.divide(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(batch, channel, height, 2 * width,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  nntrainer::TensorV2 shared_input =
    input.getSharedDataTensor(dim, 0, false, "");
  nntrainer::TensorV2 test(dim);

  EXPECT_THROW(shared_input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim);
  nntrainer::TensorV2 test(batch, channel, height, 2 * width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  nntrainer::TensorV2 shared_test = test.getSharedDataTensor(dim, 0, false, "");

  EXPECT_THROW(input.divide(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim, false);
  nntrainer::TensorV2 test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::TensorV2 input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::TensorV2 test(dim, false);

  EXPECT_THROW(input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::TensorV2 input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::TensorV2 test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::TensorV2 output(dim, false);

  EXPECT_THROW(input.divide(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_i_broadcast_01_p) {
  unsigned int N = 120;
  _FP16 *answer_data = new _FP16[N];
  nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                               nntrainer::Tdatatype::FP16);
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(1, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(3, 1, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(3, 2, 4, 1, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(1, 2, 1, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(3, 1, 4, 1, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(1, 1, 1, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(1, 2, 1, 1, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 1, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + N, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 5, 1, 4, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::TensorV2 m = rangedV2(3, 1, 1, 4, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    m.add_i(1);
    float float_data[] = {
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
    std::transform(float_data, float_data + 60, answer_data,
                   static_cast_func<_FP16>());
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  delete[] answer_data;
}

TEST(nntrainer_Tensor, divide_i_broadcast_not_supported_01_n) {
  nntrainer::TensorV2 target(3, 1, 3, 1, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  nntrainer::TensorV2 target2(3, 1, 3, 3, nntrainer::Tformat::NCHW,
                              nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_i_broadcast_not_broadcastable_02_n) {
  nntrainer::TensorV2 target(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  nntrainer::TensorV2 target2(3, 2, 3, 1, nntrainer::Tformat::NCHW,
                              nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
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
