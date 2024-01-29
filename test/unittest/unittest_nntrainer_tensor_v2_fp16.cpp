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
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 t = rangedV2(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    nntrainer::TensorV2 m = rangedV2(1, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(1),
                           static_cast<_FP16>(4),    static_cast<_FP16>(9),
                           static_cast<_FP16>(16),   static_cast<_FP16>(25),
                           static_cast<_FP16>(36),   static_cast<_FP16>(49),
                           static_cast<_FP16>(64),   static_cast<_FP16>(81),
                           static_cast<_FP16>(100),  static_cast<_FP16>(121),
                           static_cast<_FP16>(144),  static_cast<_FP16>(169),
                           static_cast<_FP16>(196),  static_cast<_FP16>(225),
                           static_cast<_FP16>(256),  static_cast<_FP16>(289),
                           static_cast<_FP16>(324),  static_cast<_FP16>(361),
                           static_cast<_FP16>(400),  static_cast<_FP16>(441),
                           static_cast<_FP16>(484),  static_cast<_FP16>(529),
                           static_cast<_FP16>(576),  static_cast<_FP16>(625),
                           static_cast<_FP16>(676),  static_cast<_FP16>(729),
                           static_cast<_FP16>(784),  static_cast<_FP16>(841),
                           static_cast<_FP16>(900),  static_cast<_FP16>(961),
                           static_cast<_FP16>(1024), static_cast<_FP16>(1089),
                           static_cast<_FP16>(1156), static_cast<_FP16>(1225),
                           static_cast<_FP16>(1296), static_cast<_FP16>(1369),
                           static_cast<_FP16>(1444), static_cast<_FP16>(1521),
                           static_cast<_FP16>(0),    static_cast<_FP16>(41),
                           static_cast<_FP16>(84),   static_cast<_FP16>(129),
                           static_cast<_FP16>(176),  static_cast<_FP16>(225),
                           static_cast<_FP16>(276),  static_cast<_FP16>(329),
                           static_cast<_FP16>(384),  static_cast<_FP16>(441),
                           static_cast<_FP16>(500),  static_cast<_FP16>(561),
                           static_cast<_FP16>(624),  static_cast<_FP16>(689),
                           static_cast<_FP16>(756),  static_cast<_FP16>(825),
                           static_cast<_FP16>(896),  static_cast<_FP16>(969),
                           static_cast<_FP16>(1044), static_cast<_FP16>(1121),
                           static_cast<_FP16>(1200), static_cast<_FP16>(1281),
                           static_cast<_FP16>(1364), static_cast<_FP16>(1449),
                           static_cast<_FP16>(1536), static_cast<_FP16>(1625),
                           static_cast<_FP16>(1716), static_cast<_FP16>(1809),
                           static_cast<_FP16>(1904), static_cast<_FP16>(2001),
                           static_cast<_FP16>(2100), static_cast<_FP16>(2201),
                           static_cast<_FP16>(2304), static_cast<_FP16>(2409),
                           static_cast<_FP16>(2516), static_cast<_FP16>(2625),
                           static_cast<_FP16>(2736), static_cast<_FP16>(2849),
                           static_cast<_FP16>(2964), static_cast<_FP16>(3081),
                           static_cast<_FP16>(0),    static_cast<_FP16>(81),
                           static_cast<_FP16>(164),  static_cast<_FP16>(249),
                           static_cast<_FP16>(336),  static_cast<_FP16>(425),
                           static_cast<_FP16>(516),  static_cast<_FP16>(609),
                           static_cast<_FP16>(704),  static_cast<_FP16>(801),
                           static_cast<_FP16>(900),  static_cast<_FP16>(1001),
                           static_cast<_FP16>(1104), static_cast<_FP16>(1209),
                           static_cast<_FP16>(1316), static_cast<_FP16>(1425),
                           static_cast<_FP16>(1536), static_cast<_FP16>(1649),
                           static_cast<_FP16>(1764), static_cast<_FP16>(1881),
                           static_cast<_FP16>(2000), static_cast<_FP16>(2121),
                           static_cast<_FP16>(2244), static_cast<_FP16>(2369),
                           static_cast<_FP16>(2496), static_cast<_FP16>(2625),
                           static_cast<_FP16>(2756), static_cast<_FP16>(2889),
                           static_cast<_FP16>(3024), static_cast<_FP16>(3161),
                           static_cast<_FP16>(3300), static_cast<_FP16>(3441),
                           static_cast<_FP16>(3584), static_cast<_FP16>(3729),
                           static_cast<_FP16>(3876), static_cast<_FP16>(4025),
                           static_cast<_FP16>(4176), static_cast<_FP16>(4329),
                           static_cast<_FP16>(4484), static_cast<_FP16>(4641)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(1),
                           static_cast<_FP16>(4),    static_cast<_FP16>(9),
                           static_cast<_FP16>(16),   static_cast<_FP16>(25),
                           static_cast<_FP16>(36),   static_cast<_FP16>(49),
                           static_cast<_FP16>(64),   static_cast<_FP16>(81),
                           static_cast<_FP16>(100),  static_cast<_FP16>(121),
                           static_cast<_FP16>(144),  static_cast<_FP16>(169),
                           static_cast<_FP16>(196),  static_cast<_FP16>(225),
                           static_cast<_FP16>(256),  static_cast<_FP16>(289),
                           static_cast<_FP16>(324),  static_cast<_FP16>(361),
                           static_cast<_FP16>(0),    static_cast<_FP16>(21),
                           static_cast<_FP16>(44),   static_cast<_FP16>(69),
                           static_cast<_FP16>(96),   static_cast<_FP16>(125),
                           static_cast<_FP16>(156),  static_cast<_FP16>(189),
                           static_cast<_FP16>(224),  static_cast<_FP16>(261),
                           static_cast<_FP16>(300),  static_cast<_FP16>(341),
                           static_cast<_FP16>(384),  static_cast<_FP16>(429),
                           static_cast<_FP16>(476),  static_cast<_FP16>(525),
                           static_cast<_FP16>(576),  static_cast<_FP16>(629),
                           static_cast<_FP16>(684),  static_cast<_FP16>(741),
                           static_cast<_FP16>(800),  static_cast<_FP16>(861),
                           static_cast<_FP16>(924),  static_cast<_FP16>(989),
                           static_cast<_FP16>(1056), static_cast<_FP16>(1125),
                           static_cast<_FP16>(1196), static_cast<_FP16>(1269),
                           static_cast<_FP16>(1344), static_cast<_FP16>(1421),
                           static_cast<_FP16>(1500), static_cast<_FP16>(1581),
                           static_cast<_FP16>(1664), static_cast<_FP16>(1749),
                           static_cast<_FP16>(1836), static_cast<_FP16>(1925),
                           static_cast<_FP16>(2016), static_cast<_FP16>(2109),
                           static_cast<_FP16>(2204), static_cast<_FP16>(2301),
                           static_cast<_FP16>(1200), static_cast<_FP16>(1281),
                           static_cast<_FP16>(1364), static_cast<_FP16>(1449),
                           static_cast<_FP16>(1536), static_cast<_FP16>(1625),
                           static_cast<_FP16>(1716), static_cast<_FP16>(1809),
                           static_cast<_FP16>(1904), static_cast<_FP16>(2001),
                           static_cast<_FP16>(2100), static_cast<_FP16>(2201),
                           static_cast<_FP16>(2304), static_cast<_FP16>(2409),
                           static_cast<_FP16>(2516), static_cast<_FP16>(2625),
                           static_cast<_FP16>(2736), static_cast<_FP16>(2849),
                           static_cast<_FP16>(2964), static_cast<_FP16>(3081),
                           static_cast<_FP16>(3200), static_cast<_FP16>(3321),
                           static_cast<_FP16>(3444), static_cast<_FP16>(3569),
                           static_cast<_FP16>(3696), static_cast<_FP16>(3825),
                           static_cast<_FP16>(3956), static_cast<_FP16>(4089),
                           static_cast<_FP16>(4224), static_cast<_FP16>(4361),
                           static_cast<_FP16>(4500), static_cast<_FP16>(4641),
                           static_cast<_FP16>(4784), static_cast<_FP16>(4929),
                           static_cast<_FP16>(5076), static_cast<_FP16>(5225),
                           static_cast<_FP16>(5376), static_cast<_FP16>(5529),
                           static_cast<_FP16>(5684), static_cast<_FP16>(5841),
                           static_cast<_FP16>(4000), static_cast<_FP16>(4141),
                           static_cast<_FP16>(4284), static_cast<_FP16>(4429),
                           static_cast<_FP16>(4576), static_cast<_FP16>(4725),
                           static_cast<_FP16>(4876), static_cast<_FP16>(5029),
                           static_cast<_FP16>(5184), static_cast<_FP16>(5341),
                           static_cast<_FP16>(5500), static_cast<_FP16>(5661),
                           static_cast<_FP16>(5824), static_cast<_FP16>(5989),
                           static_cast<_FP16>(6156), static_cast<_FP16>(6325),
                           static_cast<_FP16>(6496), static_cast<_FP16>(6669),
                           static_cast<_FP16>(6844), static_cast<_FP16>(7021)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(5),
                           static_cast<_FP16>(6),    static_cast<_FP16>(7),
                           static_cast<_FP16>(8),    static_cast<_FP16>(9),
                           static_cast<_FP16>(20),   static_cast<_FP16>(22),
                           static_cast<_FP16>(24),   static_cast<_FP16>(26),
                           static_cast<_FP16>(28),   static_cast<_FP16>(45),
                           static_cast<_FP16>(48),   static_cast<_FP16>(51),
                           static_cast<_FP16>(54),   static_cast<_FP16>(57),
                           static_cast<_FP16>(80),   static_cast<_FP16>(84),
                           static_cast<_FP16>(88),   static_cast<_FP16>(92),
                           static_cast<_FP16>(96),   static_cast<_FP16>(125),
                           static_cast<_FP16>(130),  static_cast<_FP16>(135),
                           static_cast<_FP16>(140),  static_cast<_FP16>(145),
                           static_cast<_FP16>(180),  static_cast<_FP16>(186),
                           static_cast<_FP16>(192),  static_cast<_FP16>(198),
                           static_cast<_FP16>(204),  static_cast<_FP16>(245),
                           static_cast<_FP16>(252),  static_cast<_FP16>(259),
                           static_cast<_FP16>(266),  static_cast<_FP16>(273),
                           static_cast<_FP16>(320),  static_cast<_FP16>(328),
                           static_cast<_FP16>(336),  static_cast<_FP16>(344),
                           static_cast<_FP16>(352),  static_cast<_FP16>(405),
                           static_cast<_FP16>(414),  static_cast<_FP16>(423),
                           static_cast<_FP16>(432),  static_cast<_FP16>(441),
                           static_cast<_FP16>(500),  static_cast<_FP16>(510),
                           static_cast<_FP16>(520),  static_cast<_FP16>(530),
                           static_cast<_FP16>(540),  static_cast<_FP16>(605),
                           static_cast<_FP16>(616),  static_cast<_FP16>(627),
                           static_cast<_FP16>(638),  static_cast<_FP16>(649),
                           static_cast<_FP16>(720),  static_cast<_FP16>(732),
                           static_cast<_FP16>(744),  static_cast<_FP16>(756),
                           static_cast<_FP16>(768),  static_cast<_FP16>(845),
                           static_cast<_FP16>(858),  static_cast<_FP16>(871),
                           static_cast<_FP16>(884),  static_cast<_FP16>(897),
                           static_cast<_FP16>(980),  static_cast<_FP16>(994),
                           static_cast<_FP16>(1008), static_cast<_FP16>(1022),
                           static_cast<_FP16>(1036), static_cast<_FP16>(1125),
                           static_cast<_FP16>(1140), static_cast<_FP16>(1155),
                           static_cast<_FP16>(1170), static_cast<_FP16>(1185),
                           static_cast<_FP16>(1280), static_cast<_FP16>(1296),
                           static_cast<_FP16>(1312), static_cast<_FP16>(1328),
                           static_cast<_FP16>(1344), static_cast<_FP16>(1445),
                           static_cast<_FP16>(1462), static_cast<_FP16>(1479),
                           static_cast<_FP16>(1496), static_cast<_FP16>(1513),
                           static_cast<_FP16>(1620), static_cast<_FP16>(1638),
                           static_cast<_FP16>(1656), static_cast<_FP16>(1674),
                           static_cast<_FP16>(1692), static_cast<_FP16>(1805),
                           static_cast<_FP16>(1824), static_cast<_FP16>(1843),
                           static_cast<_FP16>(1862), static_cast<_FP16>(1881),
                           static_cast<_FP16>(2000), static_cast<_FP16>(2020),
                           static_cast<_FP16>(2040), static_cast<_FP16>(2060),
                           static_cast<_FP16>(2080), static_cast<_FP16>(2205),
                           static_cast<_FP16>(2226), static_cast<_FP16>(2247),
                           static_cast<_FP16>(2268), static_cast<_FP16>(2289),
                           static_cast<_FP16>(2420), static_cast<_FP16>(2442),
                           static_cast<_FP16>(2464), static_cast<_FP16>(2486),
                           static_cast<_FP16>(2508), static_cast<_FP16>(2645),
                           static_cast<_FP16>(2668), static_cast<_FP16>(2691),
                           static_cast<_FP16>(2714), static_cast<_FP16>(2737)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(1),
                           static_cast<_FP16>(4),    static_cast<_FP16>(9),
                           static_cast<_FP16>(16),   static_cast<_FP16>(0),
                           static_cast<_FP16>(6),    static_cast<_FP16>(14),
                           static_cast<_FP16>(24),   static_cast<_FP16>(36),
                           static_cast<_FP16>(0),    static_cast<_FP16>(11),
                           static_cast<_FP16>(24),   static_cast<_FP16>(39),
                           static_cast<_FP16>(56),   static_cast<_FP16>(0),
                           static_cast<_FP16>(16),   static_cast<_FP16>(34),
                           static_cast<_FP16>(54),   static_cast<_FP16>(76),
                           static_cast<_FP16>(0),    static_cast<_FP16>(21),
                           static_cast<_FP16>(44),   static_cast<_FP16>(69),
                           static_cast<_FP16>(96),   static_cast<_FP16>(0),
                           static_cast<_FP16>(26),   static_cast<_FP16>(54),
                           static_cast<_FP16>(84),   static_cast<_FP16>(116),
                           static_cast<_FP16>(0),    static_cast<_FP16>(31),
                           static_cast<_FP16>(64),   static_cast<_FP16>(99),
                           static_cast<_FP16>(136),  static_cast<_FP16>(0),
                           static_cast<_FP16>(36),   static_cast<_FP16>(74),
                           static_cast<_FP16>(114),  static_cast<_FP16>(156),
                           static_cast<_FP16>(200),  static_cast<_FP16>(246),
                           static_cast<_FP16>(294),  static_cast<_FP16>(344),
                           static_cast<_FP16>(396),  static_cast<_FP16>(225),
                           static_cast<_FP16>(276),  static_cast<_FP16>(329),
                           static_cast<_FP16>(384),  static_cast<_FP16>(441),
                           static_cast<_FP16>(250),  static_cast<_FP16>(306),
                           static_cast<_FP16>(364),  static_cast<_FP16>(424),
                           static_cast<_FP16>(486),  static_cast<_FP16>(275),
                           static_cast<_FP16>(336),  static_cast<_FP16>(399),
                           static_cast<_FP16>(464),  static_cast<_FP16>(531),
                           static_cast<_FP16>(300),  static_cast<_FP16>(366),
                           static_cast<_FP16>(434),  static_cast<_FP16>(504),
                           static_cast<_FP16>(576),  static_cast<_FP16>(325),
                           static_cast<_FP16>(396),  static_cast<_FP16>(469),
                           static_cast<_FP16>(544),  static_cast<_FP16>(621),
                           static_cast<_FP16>(350),  static_cast<_FP16>(426),
                           static_cast<_FP16>(504),  static_cast<_FP16>(584),
                           static_cast<_FP16>(666),  static_cast<_FP16>(375),
                           static_cast<_FP16>(456),  static_cast<_FP16>(539),
                           static_cast<_FP16>(624),  static_cast<_FP16>(711),
                           static_cast<_FP16>(800),  static_cast<_FP16>(891),
                           static_cast<_FP16>(984),  static_cast<_FP16>(1079),
                           static_cast<_FP16>(1176), static_cast<_FP16>(850),
                           static_cast<_FP16>(946),  static_cast<_FP16>(1044),
                           static_cast<_FP16>(1144), static_cast<_FP16>(1246),
                           static_cast<_FP16>(900),  static_cast<_FP16>(1001),
                           static_cast<_FP16>(1104), static_cast<_FP16>(1209),
                           static_cast<_FP16>(1316), static_cast<_FP16>(950),
                           static_cast<_FP16>(1056), static_cast<_FP16>(1164),
                           static_cast<_FP16>(1274), static_cast<_FP16>(1386),
                           static_cast<_FP16>(1000), static_cast<_FP16>(1111),
                           static_cast<_FP16>(1224), static_cast<_FP16>(1339),
                           static_cast<_FP16>(1456), static_cast<_FP16>(1050),
                           static_cast<_FP16>(1166), static_cast<_FP16>(1284),
                           static_cast<_FP16>(1404), static_cast<_FP16>(1526),
                           static_cast<_FP16>(1100), static_cast<_FP16>(1221),
                           static_cast<_FP16>(1344), static_cast<_FP16>(1469),
                           static_cast<_FP16>(1596), static_cast<_FP16>(1150),
                           static_cast<_FP16>(1276), static_cast<_FP16>(1404),
                           static_cast<_FP16>(1534), static_cast<_FP16>(1666)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(1),
                           static_cast<_FP16>(4),    static_cast<_FP16>(9),
                           static_cast<_FP16>(16),   static_cast<_FP16>(0),
                           static_cast<_FP16>(6),    static_cast<_FP16>(14),
                           static_cast<_FP16>(24),   static_cast<_FP16>(36),
                           static_cast<_FP16>(0),    static_cast<_FP16>(11),
                           static_cast<_FP16>(24),   static_cast<_FP16>(39),
                           static_cast<_FP16>(56),   static_cast<_FP16>(0),
                           static_cast<_FP16>(16),   static_cast<_FP16>(34),
                           static_cast<_FP16>(54),   static_cast<_FP16>(76),
                           static_cast<_FP16>(100),  static_cast<_FP16>(126),
                           static_cast<_FP16>(154),  static_cast<_FP16>(184),
                           static_cast<_FP16>(216),  static_cast<_FP16>(125),
                           static_cast<_FP16>(156),  static_cast<_FP16>(189),
                           static_cast<_FP16>(224),  static_cast<_FP16>(261),
                           static_cast<_FP16>(150),  static_cast<_FP16>(186),
                           static_cast<_FP16>(224),  static_cast<_FP16>(264),
                           static_cast<_FP16>(306),  static_cast<_FP16>(175),
                           static_cast<_FP16>(216),  static_cast<_FP16>(259),
                           static_cast<_FP16>(304),  static_cast<_FP16>(351),
                           static_cast<_FP16>(0),    static_cast<_FP16>(41),
                           static_cast<_FP16>(84),   static_cast<_FP16>(129),
                           static_cast<_FP16>(176),  static_cast<_FP16>(0),
                           static_cast<_FP16>(46),   static_cast<_FP16>(94),
                           static_cast<_FP16>(144),  static_cast<_FP16>(196),
                           static_cast<_FP16>(0),    static_cast<_FP16>(51),
                           static_cast<_FP16>(104),  static_cast<_FP16>(159),
                           static_cast<_FP16>(216),  static_cast<_FP16>(0),
                           static_cast<_FP16>(56),   static_cast<_FP16>(114),
                           static_cast<_FP16>(174),  static_cast<_FP16>(236),
                           static_cast<_FP16>(300),  static_cast<_FP16>(366),
                           static_cast<_FP16>(434),  static_cast<_FP16>(504),
                           static_cast<_FP16>(576),  static_cast<_FP16>(325),
                           static_cast<_FP16>(396),  static_cast<_FP16>(469),
                           static_cast<_FP16>(544),  static_cast<_FP16>(621),
                           static_cast<_FP16>(350),  static_cast<_FP16>(426),
                           static_cast<_FP16>(504),  static_cast<_FP16>(584),
                           static_cast<_FP16>(666),  static_cast<_FP16>(375),
                           static_cast<_FP16>(456),  static_cast<_FP16>(539),
                           static_cast<_FP16>(624),  static_cast<_FP16>(711),
                           static_cast<_FP16>(0),    static_cast<_FP16>(81),
                           static_cast<_FP16>(164),  static_cast<_FP16>(249),
                           static_cast<_FP16>(336),  static_cast<_FP16>(0),
                           static_cast<_FP16>(86),   static_cast<_FP16>(174),
                           static_cast<_FP16>(264),  static_cast<_FP16>(356),
                           static_cast<_FP16>(0),    static_cast<_FP16>(91),
                           static_cast<_FP16>(184),  static_cast<_FP16>(279),
                           static_cast<_FP16>(376),  static_cast<_FP16>(0),
                           static_cast<_FP16>(96),   static_cast<_FP16>(194),
                           static_cast<_FP16>(294),  static_cast<_FP16>(396),
                           static_cast<_FP16>(500),  static_cast<_FP16>(606),
                           static_cast<_FP16>(714),  static_cast<_FP16>(824),
                           static_cast<_FP16>(936),  static_cast<_FP16>(525),
                           static_cast<_FP16>(636),  static_cast<_FP16>(749),
                           static_cast<_FP16>(864),  static_cast<_FP16>(981),
                           static_cast<_FP16>(550),  static_cast<_FP16>(666),
                           static_cast<_FP16>(784),  static_cast<_FP16>(904),
                           static_cast<_FP16>(1026), static_cast<_FP16>(575),
                           static_cast<_FP16>(696),  static_cast<_FP16>(819),
                           static_cast<_FP16>(944),  static_cast<_FP16>(1071)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(5),
                           static_cast<_FP16>(6),    static_cast<_FP16>(7),
                           static_cast<_FP16>(8),    static_cast<_FP16>(9),
                           static_cast<_FP16>(20),   static_cast<_FP16>(22),
                           static_cast<_FP16>(24),   static_cast<_FP16>(26),
                           static_cast<_FP16>(28),   static_cast<_FP16>(45),
                           static_cast<_FP16>(48),   static_cast<_FP16>(51),
                           static_cast<_FP16>(54),   static_cast<_FP16>(57),
                           static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(25),
                           static_cast<_FP16>(26),   static_cast<_FP16>(27),
                           static_cast<_FP16>(28),   static_cast<_FP16>(29),
                           static_cast<_FP16>(60),   static_cast<_FP16>(62),
                           static_cast<_FP16>(64),   static_cast<_FP16>(66),
                           static_cast<_FP16>(68),   static_cast<_FP16>(105),
                           static_cast<_FP16>(108),  static_cast<_FP16>(111),
                           static_cast<_FP16>(114),  static_cast<_FP16>(117),
                           static_cast<_FP16>(160),  static_cast<_FP16>(164),
                           static_cast<_FP16>(168),  static_cast<_FP16>(172),
                           static_cast<_FP16>(176),  static_cast<_FP16>(225),
                           static_cast<_FP16>(230),  static_cast<_FP16>(235),
                           static_cast<_FP16>(240),  static_cast<_FP16>(245),
                           static_cast<_FP16>(300),  static_cast<_FP16>(306),
                           static_cast<_FP16>(312),  static_cast<_FP16>(318),
                           static_cast<_FP16>(324),  static_cast<_FP16>(385),
                           static_cast<_FP16>(392),  static_cast<_FP16>(399),
                           static_cast<_FP16>(406),  static_cast<_FP16>(413),
                           static_cast<_FP16>(240),  static_cast<_FP16>(244),
                           static_cast<_FP16>(248),  static_cast<_FP16>(252),
                           static_cast<_FP16>(256),  static_cast<_FP16>(325),
                           static_cast<_FP16>(330),  static_cast<_FP16>(335),
                           static_cast<_FP16>(340),  static_cast<_FP16>(345),
                           static_cast<_FP16>(420),  static_cast<_FP16>(426),
                           static_cast<_FP16>(432),  static_cast<_FP16>(438),
                           static_cast<_FP16>(444),  static_cast<_FP16>(525),
                           static_cast<_FP16>(532),  static_cast<_FP16>(539),
                           static_cast<_FP16>(546),  static_cast<_FP16>(553),
                           static_cast<_FP16>(640),  static_cast<_FP16>(648),
                           static_cast<_FP16>(656),  static_cast<_FP16>(664),
                           static_cast<_FP16>(672),  static_cast<_FP16>(765),
                           static_cast<_FP16>(774),  static_cast<_FP16>(783),
                           static_cast<_FP16>(792),  static_cast<_FP16>(801),
                           static_cast<_FP16>(900),  static_cast<_FP16>(910),
                           static_cast<_FP16>(920),  static_cast<_FP16>(930),
                           static_cast<_FP16>(940),  static_cast<_FP16>(1045),
                           static_cast<_FP16>(1056), static_cast<_FP16>(1067),
                           static_cast<_FP16>(1078), static_cast<_FP16>(1089),
                           static_cast<_FP16>(800),  static_cast<_FP16>(808),
                           static_cast<_FP16>(816),  static_cast<_FP16>(824),
                           static_cast<_FP16>(832),  static_cast<_FP16>(945),
                           static_cast<_FP16>(954),  static_cast<_FP16>(963),
                           static_cast<_FP16>(972),  static_cast<_FP16>(981),
                           static_cast<_FP16>(1100), static_cast<_FP16>(1110),
                           static_cast<_FP16>(1120), static_cast<_FP16>(1130),
                           static_cast<_FP16>(1140), static_cast<_FP16>(1265),
                           static_cast<_FP16>(1276), static_cast<_FP16>(1287),
                           static_cast<_FP16>(1298), static_cast<_FP16>(1309)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(1),
                           static_cast<_FP16>(4),   static_cast<_FP16>(9),
                           static_cast<_FP16>(16),  static_cast<_FP16>(0),
                           static_cast<_FP16>(6),   static_cast<_FP16>(14),
                           static_cast<_FP16>(24),  static_cast<_FP16>(36),
                           static_cast<_FP16>(0),   static_cast<_FP16>(11),
                           static_cast<_FP16>(24),  static_cast<_FP16>(39),
                           static_cast<_FP16>(56),  static_cast<_FP16>(0),
                           static_cast<_FP16>(16),  static_cast<_FP16>(34),
                           static_cast<_FP16>(54),  static_cast<_FP16>(76),
                           static_cast<_FP16>(0),   static_cast<_FP16>(21),
                           static_cast<_FP16>(44),  static_cast<_FP16>(69),
                           static_cast<_FP16>(96),  static_cast<_FP16>(0),
                           static_cast<_FP16>(26),  static_cast<_FP16>(54),
                           static_cast<_FP16>(84),  static_cast<_FP16>(116),
                           static_cast<_FP16>(0),   static_cast<_FP16>(31),
                           static_cast<_FP16>(64),  static_cast<_FP16>(99),
                           static_cast<_FP16>(136), static_cast<_FP16>(0),
                           static_cast<_FP16>(36),  static_cast<_FP16>(74),
                           static_cast<_FP16>(114), static_cast<_FP16>(156),
                           static_cast<_FP16>(0),   static_cast<_FP16>(41),
                           static_cast<_FP16>(84),  static_cast<_FP16>(129),
                           static_cast<_FP16>(176), static_cast<_FP16>(0),
                           static_cast<_FP16>(46),  static_cast<_FP16>(94),
                           static_cast<_FP16>(144), static_cast<_FP16>(196),
                           static_cast<_FP16>(0),   static_cast<_FP16>(51),
                           static_cast<_FP16>(104), static_cast<_FP16>(159),
                           static_cast<_FP16>(216), static_cast<_FP16>(0),
                           static_cast<_FP16>(56),  static_cast<_FP16>(114),
                           static_cast<_FP16>(174), static_cast<_FP16>(236),
                           static_cast<_FP16>(0),   static_cast<_FP16>(61),
                           static_cast<_FP16>(124), static_cast<_FP16>(189),
                           static_cast<_FP16>(256), static_cast<_FP16>(0),
                           static_cast<_FP16>(66),  static_cast<_FP16>(134),
                           static_cast<_FP16>(204), static_cast<_FP16>(276),
                           static_cast<_FP16>(0),   static_cast<_FP16>(71),
                           static_cast<_FP16>(144), static_cast<_FP16>(219),
                           static_cast<_FP16>(296), static_cast<_FP16>(0),
                           static_cast<_FP16>(76),  static_cast<_FP16>(154),
                           static_cast<_FP16>(234), static_cast<_FP16>(316),
                           static_cast<_FP16>(0),   static_cast<_FP16>(81),
                           static_cast<_FP16>(164), static_cast<_FP16>(249),
                           static_cast<_FP16>(336), static_cast<_FP16>(0),
                           static_cast<_FP16>(86),  static_cast<_FP16>(174),
                           static_cast<_FP16>(264), static_cast<_FP16>(356),
                           static_cast<_FP16>(0),   static_cast<_FP16>(91),
                           static_cast<_FP16>(184), static_cast<_FP16>(279),
                           static_cast<_FP16>(376), static_cast<_FP16>(0),
                           static_cast<_FP16>(96),  static_cast<_FP16>(194),
                           static_cast<_FP16>(294), static_cast<_FP16>(396),
                           static_cast<_FP16>(0),   static_cast<_FP16>(101),
                           static_cast<_FP16>(204), static_cast<_FP16>(309),
                           static_cast<_FP16>(416), static_cast<_FP16>(0),
                           static_cast<_FP16>(106), static_cast<_FP16>(214),
                           static_cast<_FP16>(324), static_cast<_FP16>(436),
                           static_cast<_FP16>(0),   static_cast<_FP16>(111),
                           static_cast<_FP16>(224), static_cast<_FP16>(339),
                           static_cast<_FP16>(456), static_cast<_FP16>(0),
                           static_cast<_FP16>(116), static_cast<_FP16>(234),
                           static_cast<_FP16>(354), static_cast<_FP16>(476)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(20),  static_cast<_FP16>(21),
                           static_cast<_FP16>(22),  static_cast<_FP16>(23),
                           static_cast<_FP16>(24),  static_cast<_FP16>(25),
                           static_cast<_FP16>(26),  static_cast<_FP16>(27),
                           static_cast<_FP16>(28),  static_cast<_FP16>(29),
                           static_cast<_FP16>(30),  static_cast<_FP16>(31),
                           static_cast<_FP16>(32),  static_cast<_FP16>(33),
                           static_cast<_FP16>(34),  static_cast<_FP16>(35),
                           static_cast<_FP16>(36),  static_cast<_FP16>(37),
                           static_cast<_FP16>(38),  static_cast<_FP16>(39),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(60),  static_cast<_FP16>(61),
                           static_cast<_FP16>(62),  static_cast<_FP16>(63),
                           static_cast<_FP16>(64),  static_cast<_FP16>(65),
                           static_cast<_FP16>(66),  static_cast<_FP16>(67),
                           static_cast<_FP16>(68),  static_cast<_FP16>(69),
                           static_cast<_FP16>(70),  static_cast<_FP16>(71),
                           static_cast<_FP16>(72),  static_cast<_FP16>(73),
                           static_cast<_FP16>(74),  static_cast<_FP16>(75),
                           static_cast<_FP16>(76),  static_cast<_FP16>(77),
                           static_cast<_FP16>(78),  static_cast<_FP16>(79),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(100), static_cast<_FP16>(101),
                           static_cast<_FP16>(102), static_cast<_FP16>(103),
                           static_cast<_FP16>(104), static_cast<_FP16>(105),
                           static_cast<_FP16>(106), static_cast<_FP16>(107),
                           static_cast<_FP16>(108), static_cast<_FP16>(109),
                           static_cast<_FP16>(110), static_cast<_FP16>(111),
                           static_cast<_FP16>(112), static_cast<_FP16>(113),
                           static_cast<_FP16>(114), static_cast<_FP16>(115),
                           static_cast<_FP16>(116), static_cast<_FP16>(117),
                           static_cast<_FP16>(118), static_cast<_FP16>(119)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(40),  static_cast<_FP16>(41),
                           static_cast<_FP16>(42),  static_cast<_FP16>(43),
                           static_cast<_FP16>(44),  static_cast<_FP16>(45),
                           static_cast<_FP16>(46),  static_cast<_FP16>(47),
                           static_cast<_FP16>(48),  static_cast<_FP16>(49),
                           static_cast<_FP16>(50),  static_cast<_FP16>(51),
                           static_cast<_FP16>(52),  static_cast<_FP16>(53),
                           static_cast<_FP16>(54),  static_cast<_FP16>(55),
                           static_cast<_FP16>(56),  static_cast<_FP16>(57),
                           static_cast<_FP16>(58),  static_cast<_FP16>(59),
                           static_cast<_FP16>(60),  static_cast<_FP16>(61),
                           static_cast<_FP16>(62),  static_cast<_FP16>(63),
                           static_cast<_FP16>(64),  static_cast<_FP16>(65),
                           static_cast<_FP16>(66),  static_cast<_FP16>(67),
                           static_cast<_FP16>(68),  static_cast<_FP16>(69),
                           static_cast<_FP16>(70),  static_cast<_FP16>(71),
                           static_cast<_FP16>(72),  static_cast<_FP16>(73),
                           static_cast<_FP16>(74),  static_cast<_FP16>(75),
                           static_cast<_FP16>(76),  static_cast<_FP16>(77),
                           static_cast<_FP16>(78),  static_cast<_FP16>(79),
                           static_cast<_FP16>(160), static_cast<_FP16>(162),
                           static_cast<_FP16>(164), static_cast<_FP16>(166),
                           static_cast<_FP16>(168), static_cast<_FP16>(170),
                           static_cast<_FP16>(172), static_cast<_FP16>(174),
                           static_cast<_FP16>(176), static_cast<_FP16>(178),
                           static_cast<_FP16>(180), static_cast<_FP16>(182),
                           static_cast<_FP16>(184), static_cast<_FP16>(186),
                           static_cast<_FP16>(188), static_cast<_FP16>(190),
                           static_cast<_FP16>(192), static_cast<_FP16>(194),
                           static_cast<_FP16>(196), static_cast<_FP16>(198),
                           static_cast<_FP16>(200), static_cast<_FP16>(202),
                           static_cast<_FP16>(204), static_cast<_FP16>(206),
                           static_cast<_FP16>(208), static_cast<_FP16>(210),
                           static_cast<_FP16>(212), static_cast<_FP16>(214),
                           static_cast<_FP16>(216), static_cast<_FP16>(218),
                           static_cast<_FP16>(220), static_cast<_FP16>(222),
                           static_cast<_FP16>(224), static_cast<_FP16>(226),
                           static_cast<_FP16>(228), static_cast<_FP16>(230),
                           static_cast<_FP16>(232), static_cast<_FP16>(234),
                           static_cast<_FP16>(236), static_cast<_FP16>(238)};
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
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(1),
                           static_cast<_FP16>(4),   static_cast<_FP16>(9),
                           static_cast<_FP16>(0),   static_cast<_FP16>(5),
                           static_cast<_FP16>(12),  static_cast<_FP16>(21),
                           static_cast<_FP16>(0),   static_cast<_FP16>(9),
                           static_cast<_FP16>(20),  static_cast<_FP16>(33),
                           static_cast<_FP16>(0),   static_cast<_FP16>(13),
                           static_cast<_FP16>(28),  static_cast<_FP16>(45),
                           static_cast<_FP16>(0),   static_cast<_FP16>(17),
                           static_cast<_FP16>(36),  static_cast<_FP16>(57),
                           static_cast<_FP16>(80),  static_cast<_FP16>(105),
                           static_cast<_FP16>(132), static_cast<_FP16>(161),
                           static_cast<_FP16>(96),  static_cast<_FP16>(125),
                           static_cast<_FP16>(156), static_cast<_FP16>(189),
                           static_cast<_FP16>(112), static_cast<_FP16>(145),
                           static_cast<_FP16>(180), static_cast<_FP16>(217),
                           static_cast<_FP16>(128), static_cast<_FP16>(165),
                           static_cast<_FP16>(204), static_cast<_FP16>(245),
                           static_cast<_FP16>(144), static_cast<_FP16>(185),
                           static_cast<_FP16>(228), static_cast<_FP16>(273),
                           static_cast<_FP16>(320), static_cast<_FP16>(369),
                           static_cast<_FP16>(420), static_cast<_FP16>(473),
                           static_cast<_FP16>(352), static_cast<_FP16>(405),
                           static_cast<_FP16>(460), static_cast<_FP16>(517),
                           static_cast<_FP16>(384), static_cast<_FP16>(441),
                           static_cast<_FP16>(500), static_cast<_FP16>(561),
                           static_cast<_FP16>(416), static_cast<_FP16>(477),
                           static_cast<_FP16>(540), static_cast<_FP16>(605),
                           static_cast<_FP16>(448), static_cast<_FP16>(513),
                           static_cast<_FP16>(580), static_cast<_FP16>(649)};
    nntrainer::TensorV2 answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
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
