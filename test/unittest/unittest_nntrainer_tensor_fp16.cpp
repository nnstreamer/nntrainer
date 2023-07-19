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
#include <iostream>

TEST(nntrainer_Tensor, Tensor_01_fp16_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Tensor tensor = nntrainer::Tensor(
    1, 2, 3, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  tensor.setZero();
  ASSERT_NE(nullptr, tensor.getData<__fp16>());
  if (tensor.getValue(0, 0, 0, 0) != 0.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_01_nhwc_fp16_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Tensor tensor = nntrainer::Tensor(
    1, 2, 3, nntrainer::Tformat::NHWC, nntrainer::Tdatatype::FP16);
  tensor.setZero();
  ASSERT_NE(nullptr, tensor.getData<__fp16>());
  if (tensor.getValue<__fp16>(0, 0, 0, 0) != 0.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_02_fp16_p) {
  int status = ML_ERROR_NONE;
  int height = 3;
  int width = 10;
  std::vector<std::vector<__fp16>> in;
  for (int i = 0; i < height; ++i) {
    std::vector<__fp16> tv;
    for (int j = 0; j < width; ++j) {
      tv.push_back(i * 2.0 + j);
    }
    in.push_back(tv);
  }

  nntrainer::Tensor tensor = nntrainer::Tensor(in, {ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP16});
  ASSERT_NE(nullptr, tensor.getData<__fp16>());

  if (tensor.getValue<__fp16>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_03_fp16_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  std::vector<std::vector<std::vector<__fp16>>> in;
  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<__fp16>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<__fp16> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::Tensor tensor = nntrainer::Tensor(in, {ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP16});
  ASSERT_NE(nullptr, tensor.getData<__fp16>());

  if (tensor.getValue<__fp16>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_i_01_fp16_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);
  input.print(std::cout);
  
  nntrainer::Tensor original;
  original.copy(input);
    
  status = input.multiply_i(2.0);
  EXPECT_EQ(status, ML_ERROR_NONE);
  input.print(std::cout);

  __fp16 *data = original.getData<__fp16>();
  ASSERT_NE(nullptr, data);
  __fp16 *indata = input.getData<__fp16>();
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

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original;
  original.copy(input);

  status = input.multiply_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);

  __fp16 *data = original.getData<__fp16>();
  ASSERT_NE(nullptr, data);
  __fp16 *indata = input.getData<__fp16>();
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

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor target2(batch, channel, height - 2, width - 1,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  status = input.multiply_i(target2);

  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

// TEST(nntrainer_Tensor, multiply_i_broadcast_01_fp16_p) {
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t =
//       ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
//     nntrainer::Tensor m =
//       ranged(1, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(1), static_cast<__fp16>(4), static_cast<__fp16>(9), static_cast<__fp16>(16), static_cast<__fp16>(25), static_cast<__fp16>(36), static_cast<__fp16>(49), static_cast<__fp16>(64), static_cast<__fp16>(81), static_cast<__fp16>(100), static_cast<__fp16>(121),
//       static_cast<__fp16>(144), static_cast<__fp16>(169), static_cast<__fp16>(196), static_cast<__fp16>(225), static_cast<__fp16>(256), static_cast<__fp16>(289), static_cast<__fp16>(324), static_cast<__fp16>(361), static_cast<__fp16>(400), static_cast<__fp16>(441), static_cast<__fp16>(484), static_cast<__fp16>(529),
//       static_cast<__fp16>(576), static_cast<__fp16>(625), static_cast<__fp16>(676), static_cast<__fp16>(729), static_cast<__fp16>(784), static_cast<__fp16>(841), static_cast<__fp16>(900), static_cast<__fp16>(961), static_cast<__fp16>(1024), static_cast<__fp16>(1089), static_cast<__fp16>(1156), static_cast<__fp16>(1225),
//       static_cast<__fp16>(1296), static_cast<__fp16>(1369), static_cast<__fp16>(1444), static_cast<__fp16>(1521), static_cast<__fp16>(0), static_cast<__fp16>(41), static_cast<__fp16>(84), static_cast<__fp16>(129), static_cast<__fp16>(176), static_cast<__fp16>(225), static_cast<__fp16>(276), static_cast<__fp16>(329),
//       static_cast<__fp16>(384), static_cast<__fp16>(441), static_cast<__fp16>(500), static_cast<__fp16>(561), static_cast<__fp16>(624), static_cast<__fp16>(689), static_cast<__fp16>(756), static_cast<__fp16>(825), static_cast<__fp16>(896), static_cast<__fp16>(969), static_cast<__fp16>(1044), static_cast<__fp16>(1121),
//       static_cast<__fp16>(1200), static_cast<__fp16>(1281), static_cast<__fp16>(1364), static_cast<__fp16>(1449), static_cast<__fp16>(1536), static_cast<__fp16>(1625), static_cast<__fp16>(1716), static_cast<__fp16>(1809), static_cast<__fp16>(1904), static_cast<__fp16>(2001), static_cast<__fp16>(2100), static_cast<__fp16>(2201),
//       static_cast<__fp16>(2304), static_cast<__fp16>(2409), static_cast<__fp16>(2516), static_cast<__fp16>(2625), static_cast<__fp16>(2736), static_cast<__fp16>(2849), static_cast<__fp16>(2964), static_cast<__fp16>(3081), static_cast<__fp16>(0), static_cast<__fp16>(81), static_cast<__fp16>(164), static_cast<__fp16>(249),
//       static_cast<__fp16>(336), static_cast<__fp16>(425), static_cast<__fp16>(516), static_cast<__fp16>(609), static_cast<__fp16>(704), static_cast<__fp16>(801), static_cast<__fp16>(900), static_cast<__fp16>(1001), static_cast<__fp16>(1104), static_cast<__fp16>(1209), static_cast<__fp16>(1316), static_cast<__fp16>(1425),
//       static_cast<__fp16>(1536), static_cast<__fp16>(1649), static_cast<__fp16>(1764), static_cast<__fp16>(1881), static_cast<__fp16>(2000), static_cast<__fp16>(2121), static_cast<__fp16>(2244), static_cast<__fp16>(2369), static_cast<__fp16>(2496), static_cast<__fp16>(2625), static_cast<__fp16>(2756), static_cast<__fp16>(2889),
//       static_cast<__fp16>(3024), static_cast<__fp16>(3161), static_cast<__fp16>(3300), static_cast<__fp16>(3441), static_cast<__fp16>(3584), static_cast<__fp16>(3729), static_cast<__fp16>(3876), static_cast<__fp16>(4025), static_cast<__fp16>(4176), static_cast<__fp16>(4329), static_cast<__fp16>(4484), static_cast<__fp16>(4641)};
//     nntrainer::Tensor answer(ref_dim, answer_data, nntrainer::Tdatatype::FP16);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
//     nntrainer::Tensor m = ranged(3, 1, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(1), static_cast<__fp16>(4), static_cast<__fp16>(9), static_cast<__fp16>(16), static_cast<__fp16>(25), static_cast<__fp16>(36), static_cast<__fp16>(49), static_cast<__fp16>(64), static_cast<__fp16>(81), static_cast<__fp16>(100), static_cast<__fp16>(121),
//       static_cast<__fp16>(144), static_cast<__fp16>(169), static_cast<__fp16>(196), static_cast<__fp16>(225), static_cast<__fp16>(256), static_cast<__fp16>(289), static_cast<__fp16>(324), static_cast<__fp16>(361), static_cast<__fp16>(0), static_cast<__fp16>(21), static_cast<__fp16>(44), static_cast<__fp16>(69),
//       static_cast<__fp16>(96), static_cast<__fp16>(125), static_cast<__fp16>(156), static_cast<__fp16>(189), static_cast<__fp16>(224), static_cast<__fp16>(261), static_cast<__fp16>(300), static_cast<__fp16>(341), static_cast<__fp16>(384), static_cast<__fp16>(429), static_cast<__fp16>(476), static_cast<__fp16>(525),
//       static_cast<__fp16>(576), static_cast<__fp16>(629), static_cast<__fp16>(684), static_cast<__fp16>(741), static_cast<__fp16>(800), static_cast<__fp16>(861), static_cast<__fp16>(924), static_cast<__fp16>(989), static_cast<__fp16>(1056), static_cast<__fp16>(1125), static_cast<__fp16>(1196), static_cast<__fp16>(1269),
//       static_cast<__fp16>(1344), static_cast<__fp16>(1421), static_cast<__fp16>(1500), static_cast<__fp16>(1581), static_cast<__fp16>(1664), static_cast<__fp16>(1749), static_cast<__fp16>(1836), static_cast<__fp16>(1925), static_cast<__fp16>(2016), static_cast<__fp16>(2109), static_cast<__fp16>(2204), static_cast<__fp16>(2301),
//       static_cast<__fp16>(1200), static_cast<__fp16>(1281), static_cast<__fp16>(1364), static_cast<__fp16>(1449), static_cast<__fp16>(1536), static_cast<__fp16>(1625), static_cast<__fp16>(1716), static_cast<__fp16>(1809), static_cast<__fp16>(1904), static_cast<__fp16>(2001), static_cast<__fp16>(2100), static_cast<__fp16>(2201),
//       static_cast<__fp16>(2304), static_cast<__fp16>(2409), static_cast<__fp16>(2516), static_cast<__fp16>(2625), static_cast<__fp16>(2736), static_cast<__fp16>(2849), static_cast<__fp16>(2964), static_cast<__fp16>(3081), static_cast<__fp16>(3200), static_cast<__fp16>(3321), static_cast<__fp16>(3444), static_cast<__fp16>(3569),
//       static_cast<__fp16>(3696), static_cast<__fp16>(3825), static_cast<__fp16>(3956), static_cast<__fp16>(4089), static_cast<__fp16>(4224), static_cast<__fp16>(4361), static_cast<__fp16>(4500), static_cast<__fp16>(4641), static_cast<__fp16>(4784), static_cast<__fp16>(4929), static_cast<__fp16>(5076), static_cast<__fp16>(5225),
//       static_cast<__fp16>(5376), static_cast<__fp16>(5529), static_cast<__fp16>(5684), static_cast<__fp16>(5841), static_cast<__fp16>(4000), static_cast<__fp16>(4141), static_cast<__fp16>(4284), static_cast<__fp16>(4429), static_cast<__fp16>(4576), static_cast<__fp16>(4725), static_cast<__fp16>(4876), static_cast<__fp16>(5029),
//       static_cast<__fp16>(5184), static_cast<__fp16>(5341), static_cast<__fp16>(5500), static_cast<__fp16>(5661), static_cast<__fp16>(5824), static_cast<__fp16>(5989), static_cast<__fp16>(6156), static_cast<__fp16>(6325), static_cast<__fp16>(6496), static_cast<__fp16>(6669), static_cast<__fp16>(6844), static_cast<__fp16>(7021)};
//     nntrainer::Tensor answer(ref_dim, answer_data, nntrainer::Tdatatype::FP16);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 2, 4, 1);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(5), static_cast<__fp16>(6), static_cast<__fp16>(7), static_cast<__fp16>(8), static_cast<__fp16>(9), static_cast<__fp16>(20), static_cast<__fp16>(22),
//       static_cast<__fp16>(24), static_cast<__fp16>(26), static_cast<__fp16>(28), static_cast<__fp16>(45), static_cast<__fp16>(48), static_cast<__fp16>(51), static_cast<__fp16>(54), static_cast<__fp16>(57), static_cast<__fp16>(80), static_cast<__fp16>(84), static_cast<__fp16>(88), static_cast<__fp16>(92),
//       static_cast<__fp16>(96), static_cast<__fp16>(125), static_cast<__fp16>(130), static_cast<__fp16>(135), static_cast<__fp16>(140), static_cast<__fp16>(145), static_cast<__fp16>(180), static_cast<__fp16>(186), static_cast<__fp16>(192), static_cast<__fp16>(198), static_cast<__fp16>(204), static_cast<__fp16>(245),
//       static_cast<__fp16>(252), static_cast<__fp16>(259), static_cast<__fp16>(266), static_cast<__fp16>(273), static_cast<__fp16>(320), static_cast<__fp16>(328), static_cast<__fp16>(336), static_cast<__fp16>(344), static_cast<__fp16>(352), static_cast<__fp16>(405), static_cast<__fp16>(414), static_cast<__fp16>(423),
//       static_cast<__fp16>(432), static_cast<__fp16>(441), static_cast<__fp16>(500), static_cast<__fp16>(510), static_cast<__fp16>(520), static_cast<__fp16>(530), static_cast<__fp16>(540), static_cast<__fp16>(605), static_cast<__fp16>(616), static_cast<__fp16>(627), static_cast<__fp16>(638), static_cast<__fp16>(649),
//       static_cast<__fp16>(720), static_cast<__fp16>(732), static_cast<__fp16>(744), static_cast<__fp16>(756), static_cast<__fp16>(768), static_cast<__fp16>(845), static_cast<__fp16>(858), static_cast<__fp16>(871), static_cast<__fp16>(884), static_cast<__fp16>(897), static_cast<__fp16>(980), static_cast<__fp16>(994),
//       static_cast<__fp16>(1008), static_cast<__fp16>(1022), static_cast<__fp16>(1036), static_cast<__fp16>(1125), static_cast<__fp16>(1140), static_cast<__fp16>(1155), static_cast<__fp16>(1170), static_cast<__fp16>(1185), static_cast<__fp16>(1280), static_cast<__fp16>(1296), static_cast<__fp16>(1312), static_cast<__fp16>(1328),
//       static_cast<__fp16>(1344), static_cast<__fp16>(1445), static_cast<__fp16>(1462), static_cast<__fp16>(1479), static_cast<__fp16>(1496), static_cast<__fp16>(1513), static_cast<__fp16>(1620), static_cast<__fp16>(1638), static_cast<__fp16>(1656), static_cast<__fp16>(1674), static_cast<__fp16>(1692), static_cast<__fp16>(1805),
//       static_cast<__fp16>(1824), static_cast<__fp16>(1843), static_cast<__fp16>(1862), static_cast<__fp16>(1881), static_cast<__fp16>(2000), static_cast<__fp16>(2020), static_cast<__fp16>(2040), static_cast<__fp16>(2060), static_cast<__fp16>(2080), static_cast<__fp16>(2205), static_cast<__fp16>(2226), static_cast<__fp16>(2247),
//       static_cast<__fp16>(2268), static_cast<__fp16>(2289), static_cast<__fp16>(2420), static_cast<__fp16>(2442), static_cast<__fp16>(2464), static_cast<__fp16>(2486), static_cast<__fp16>(2508), static_cast<__fp16>(2645), static_cast<__fp16>(2668), static_cast<__fp16>(2691), static_cast<__fp16>(2714), static_cast<__fp16>(2737)};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 1, 1, 5);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(1), static_cast<__fp16>(4), static_cast<__fp16>(9), static_cast<__fp16>(16), static_cast<__fp16>(0), static_cast<__fp16>(6), static_cast<__fp16>(14), static_cast<__fp16>(24), static_cast<__fp16>(36), static_cast<__fp16>(0), static_cast<__fp16>(11),
//       static_cast<__fp16>(24), static_cast<__fp16>(39), static_cast<__fp16>(56), static_cast<__fp16>(0), static_cast<__fp16>(16), static_cast<__fp16>(34), static_cast<__fp16>(54), static_cast<__fp16>(76), static_cast<__fp16>(0), static_cast<__fp16>(21), static_cast<__fp16>(44), static_cast<__fp16>(69),
//       static_cast<__fp16>(96), static_cast<__fp16>(0), static_cast<__fp16>(26), static_cast<__fp16>(54), static_cast<__fp16>(84), static_cast<__fp16>(116), static_cast<__fp16>(0), static_cast<__fp16>(31), static_cast<__fp16>(64), static_cast<__fp16>(99), static_cast<__fp16>(136), static_cast<__fp16>(0),
//       static_cast<__fp16>(36), static_cast<__fp16>(74), static_cast<__fp16>(114), static_cast<__fp16>(156), static_cast<__fp16>(200), static_cast<__fp16>(246), static_cast<__fp16>(294), static_cast<__fp16>(344), static_cast<__fp16>(396), static_cast<__fp16>(225), static_cast<__fp16>(276), static_cast<__fp16>(329),
//       static_cast<__fp16>(384), static_cast<__fp16>(441), static_cast<__fp16>(250), static_cast<__fp16>(306), static_cast<__fp16>(364), static_cast<__fp16>(424), static_cast<__fp16>(486), static_cast<__fp16>(275), static_cast<__fp16>(336), static_cast<__fp16>(399), static_cast<__fp16>(464), static_cast<__fp16>(531),
//       static_cast<__fp16>(300), static_cast<__fp16>(366), static_cast<__fp16>(434), static_cast<__fp16>(504), static_cast<__fp16>(576), static_cast<__fp16>(325), static_cast<__fp16>(396), static_cast<__fp16>(469), static_cast<__fp16>(544), static_cast<__fp16>(621), static_cast<__fp16>(350), static_cast<__fp16>(426),
//       static_cast<__fp16>(504), static_cast<__fp16>(584), static_cast<__fp16>(666), static_cast<__fp16>(375), static_cast<__fp16>(456), static_cast<__fp16>(539), static_cast<__fp16>(624), static_cast<__fp16>(711), static_cast<__fp16>(800), static_cast<__fp16>(891), static_cast<__fp16>(984), static_cast<__fp16>(1079),
//       static_cast<__fp16>(1176), static_cast<__fp16>(850), static_cast<__fp16>(946), static_cast<__fp16>(1044), static_cast<__fp16>(1144), static_cast<__fp16>(1246), static_cast<__fp16>(900), static_cast<__fp16>(1001), static_cast<__fp16>(1104), static_cast<__fp16>(1209), static_cast<__fp16>(1316), static_cast<__fp16>(950),
//       static_cast<__fp16>(1056), static_cast<__fp16>(1164), static_cast<__fp16>(1274), static_cast<__fp16>(1386), static_cast<__fp16>(1000), static_cast<__fp16>(1111), static_cast<__fp16>(1224), static_cast<__fp16>(1339), static_cast<__fp16>(1456), static_cast<__fp16>(1050), static_cast<__fp16>(1166), static_cast<__fp16>(1284),
//       static_cast<__fp16>(1404), static_cast<__fp16>(1526), static_cast<__fp16>(1100), static_cast<__fp16>(1221), static_cast<__fp16>(1344), static_cast<__fp16>(1469), static_cast<__fp16>(1596), static_cast<__fp16>(1150), static_cast<__fp16>(1276), static_cast<__fp16>(1404), static_cast<__fp16>(1534), static_cast<__fp16>(1666)};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(1, 2, 1, 5);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(1), static_cast<__fp16>(4), static_cast<__fp16>(9), static_cast<__fp16>(16), static_cast<__fp16>(0), static_cast<__fp16>(6), static_cast<__fp16>(14), static_cast<__fp16>(24), static_cast<__fp16>(36), static_cast<__fp16>(0), static_cast<__fp16>(11), static_cast<__fp16>(24), static_cast<__fp16>(39),
//       static_cast<__fp16>(56), static_cast<__fp16>(0), static_cast<__fp16>(16), static_cast<__fp16>(34), static_cast<__fp16>(54), static_cast<__fp16>(76), static_cast<__fp16>(100), static_cast<__fp16>(126), static_cast<__fp16>(154), static_cast<__fp16>(184), static_cast<__fp16>(216), static_cast<__fp16>(125), static_cast<__fp16>(156), static_cast<__fp16>(189),
//       static_cast<__fp16>(224), static_cast<__fp16>(261), static_cast<__fp16>(150), static_cast<__fp16>(186), static_cast<__fp16>(224), static_cast<__fp16>(264), static_cast<__fp16>(306), static_cast<__fp16>(175), static_cast<__fp16>(216), static_cast<__fp16>(259), static_cast<__fp16>(304), static_cast<__fp16>(351), static_cast<__fp16>(0), static_cast<__fp16>(41),
//       static_cast<__fp16>(84), static_cast<__fp16>(129), static_cast<__fp16>(176), static_cast<__fp16>(0), static_cast<__fp16>(46), static_cast<__fp16>(94), static_cast<__fp16>(144), static_cast<__fp16>(196), static_cast<__fp16>(0), static_cast<__fp16>(51), static_cast<__fp16>(104), static_cast<__fp16>(159), static_cast<__fp16>(216), static_cast<__fp16>(0),
//       static_cast<__fp16>(56), static_cast<__fp16>(114), static_cast<__fp16>(174), static_cast<__fp16>(236), static_cast<__fp16>(300), static_cast<__fp16>(366), static_cast<__fp16>(434), static_cast<__fp16>(504), static_cast<__fp16>(576), static_cast<__fp16>(325), static_cast<__fp16>(396), static_cast<__fp16>(469), static_cast<__fp16>(544), static_cast<__fp16>(621),
//       static_cast<__fp16>(350), static_cast<__fp16>(426), static_cast<__fp16>(504), static_cast<__fp16>(584), static_cast<__fp16>(666), static_cast<__fp16>(375), static_cast<__fp16>(456), static_cast<__fp16>(539), static_cast<__fp16>(624), static_cast<__fp16>(711), static_cast<__fp16>(0), static_cast<__fp16>(81), static_cast<__fp16>(164), static_cast<__fp16>(249),
//       static_cast<__fp16>(336), static_cast<__fp16>(0), static_cast<__fp16>(86), static_cast<__fp16>(174), static_cast<__fp16>(264), static_cast<__fp16>(356), static_cast<__fp16>(0), static_cast<__fp16>(91), static_cast<__fp16>(184), static_cast<__fp16>(279), static_cast<__fp16>(376), static_cast<__fp16>(0), static_cast<__fp16>(96), static_cast<__fp16>(194),
//       static_cast<__fp16>(294), static_cast<__fp16>(396), static_cast<__fp16>(500), static_cast<__fp16>(606), static_cast<__fp16>(714), static_cast<__fp16>(824), static_cast<__fp16>(936), static_cast<__fp16>(525), static_cast<__fp16>(636), static_cast<__fp16>(749), static_cast<__fp16>(864), static_cast<__fp16>(981), static_cast<__fp16>(550), static_cast<__fp16>(666),
//       static_cast<__fp16>(784), static_cast<__fp16>(904), static_cast<__fp16>(1026), static_cast<__fp16>(575), static_cast<__fp16>(696), static_cast<__fp16>(819), static_cast<__fp16>(944), static_cast<__fp16>(1071)};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 1, 4, 1);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(5), static_cast<__fp16>(6), static_cast<__fp16>(7), static_cast<__fp16>(8), static_cast<__fp16>(9), static_cast<__fp16>(20), static_cast<__fp16>(22),
//       static_cast<__fp16>(24), static_cast<__fp16>(26), static_cast<__fp16>(28), static_cast<__fp16>(45), static_cast<__fp16>(48), static_cast<__fp16>(51), static_cast<__fp16>(54), static_cast<__fp16>(57), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0),
//       static_cast<__fp16>(0), static_cast<__fp16>(25), static_cast<__fp16>(26), static_cast<__fp16>(27), static_cast<__fp16>(28), static_cast<__fp16>(29), static_cast<__fp16>(60), static_cast<__fp16>(62), static_cast<__fp16>(64), static_cast<__fp16>(66), static_cast<__fp16>(68), static_cast<__fp16>(105),
//       static_cast<__fp16>(108), static_cast<__fp16>(111), static_cast<__fp16>(114), static_cast<__fp16>(117), static_cast<__fp16>(160), static_cast<__fp16>(164), static_cast<__fp16>(168), static_cast<__fp16>(172), static_cast<__fp16>(176), static_cast<__fp16>(225), static_cast<__fp16>(230), static_cast<__fp16>(235),
//       static_cast<__fp16>(240), static_cast<__fp16>(245), static_cast<__fp16>(300), static_cast<__fp16>(306), static_cast<__fp16>(312), static_cast<__fp16>(318), static_cast<__fp16>(324), static_cast<__fp16>(385), static_cast<__fp16>(392), static_cast<__fp16>(399), static_cast<__fp16>(406), static_cast<__fp16>(413),
//       static_cast<__fp16>(240), static_cast<__fp16>(244), static_cast<__fp16>(248), static_cast<__fp16>(252), static_cast<__fp16>(256), static_cast<__fp16>(325), static_cast<__fp16>(330), static_cast<__fp16>(335), static_cast<__fp16>(340), static_cast<__fp16>(345), static_cast<__fp16>(420), static_cast<__fp16>(426),
//       static_cast<__fp16>(432), static_cast<__fp16>(438), static_cast<__fp16>(444), static_cast<__fp16>(525), static_cast<__fp16>(532), static_cast<__fp16>(539), static_cast<__fp16>(546), static_cast<__fp16>(553), static_cast<__fp16>(640), static_cast<__fp16>(648), static_cast<__fp16>(656), static_cast<__fp16>(664),
//       static_cast<__fp16>(672), static_cast<__fp16>(765), static_cast<__fp16>(774), static_cast<__fp16>(783), static_cast<__fp16>(792), static_cast<__fp16>(801), static_cast<__fp16>(900), static_cast<__fp16>(910), static_cast<__fp16>(920), static_cast<__fp16>(930), static_cast<__fp16>(940), static_cast<__fp16>(1045),
//       static_cast<__fp16>(1056), static_cast<__fp16>(1067), static_cast<__fp16>(1078), static_cast<__fp16>(1089), static_cast<__fp16>(800), static_cast<__fp16>(808), static_cast<__fp16>(816), static_cast<__fp16>(824), static_cast<__fp16>(832), static_cast<__fp16>(945), static_cast<__fp16>(954), static_cast<__fp16>(963),
//       static_cast<__fp16>(972), static_cast<__fp16>(981), static_cast<__fp16>(1100), static_cast<__fp16>(1110), static_cast<__fp16>(1120), static_cast<__fp16>(1130), static_cast<__fp16>(1140), static_cast<__fp16>(1265), static_cast<__fp16>(1276), static_cast<__fp16>(1287), static_cast<__fp16>(1298), static_cast<__fp16>(1309)};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(1, 1, 1, 5);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(1), static_cast<__fp16>(4), static_cast<__fp16>(9), static_cast<__fp16>(16), static_cast<__fp16>(0), static_cast<__fp16>(6), static_cast<__fp16>(14), static_cast<__fp16>(24), static_cast<__fp16>(36), static_cast<__fp16>(0), static_cast<__fp16>(11), static_cast<__fp16>(24), static_cast<__fp16>(39), static_cast<__fp16>(56),
//       static_cast<__fp16>(0), static_cast<__fp16>(16), static_cast<__fp16>(34), static_cast<__fp16>(54), static_cast<__fp16>(76), static_cast<__fp16>(0), static_cast<__fp16>(21), static_cast<__fp16>(44), static_cast<__fp16>(69), static_cast<__fp16>(96), static_cast<__fp16>(0), static_cast<__fp16>(26), static_cast<__fp16>(54), static_cast<__fp16>(84), static_cast<__fp16>(116),
//       static_cast<__fp16>(0), static_cast<__fp16>(31), static_cast<__fp16>(64), static_cast<__fp16>(99), static_cast<__fp16>(136), static_cast<__fp16>(0), static_cast<__fp16>(36), static_cast<__fp16>(74), static_cast<__fp16>(114), static_cast<__fp16>(156), static_cast<__fp16>(0), static_cast<__fp16>(41), static_cast<__fp16>(84), static_cast<__fp16>(129), static_cast<__fp16>(176),
//       static_cast<__fp16>(0), static_cast<__fp16>(46), static_cast<__fp16>(94), static_cast<__fp16>(144), static_cast<__fp16>(196), static_cast<__fp16>(0), static_cast<__fp16>(51), static_cast<__fp16>(104), static_cast<__fp16>(159), static_cast<__fp16>(216), static_cast<__fp16>(0), static_cast<__fp16>(56), static_cast<__fp16>(114), static_cast<__fp16>(174), static_cast<__fp16>(236),
//       static_cast<__fp16>(0), static_cast<__fp16>(61), static_cast<__fp16>(124), static_cast<__fp16>(189), static_cast<__fp16>(256), static_cast<__fp16>(0), static_cast<__fp16>(66), static_cast<__fp16>(134), static_cast<__fp16>(204), static_cast<__fp16>(276), static_cast<__fp16>(0), static_cast<__fp16>(71), static_cast<__fp16>(144), static_cast<__fp16>(219), static_cast<__fp16>(296),
//       static_cast<__fp16>(0), static_cast<__fp16>(76), static_cast<__fp16>(154), static_cast<__fp16>(234), static_cast<__fp16>(316), static_cast<__fp16>(0), static_cast<__fp16>(81), static_cast<__fp16>(164), static_cast<__fp16>(249), static_cast<__fp16>(336), static_cast<__fp16>(0), static_cast<__fp16>(86), static_cast<__fp16>(174), static_cast<__fp16>(264), static_cast<__fp16>(356),
//       static_cast<__fp16>(0), static_cast<__fp16>(91), static_cast<__fp16>(184), static_cast<__fp16>(279), static_cast<__fp16>(376), static_cast<__fp16>(0), static_cast<__fp16>(96), static_cast<__fp16>(194), static_cast<__fp16>(294), static_cast<__fp16>(396), static_cast<__fp16>(0), static_cast<__fp16>(101), static_cast<__fp16>(204), static_cast<__fp16>(309), static_cast<__fp16>(416),
//       static_cast<__fp16>(0), static_cast<__fp16>(106), static_cast<__fp16>(214), static_cast<__fp16>(324), static_cast<__fp16>(436), static_cast<__fp16>(0), static_cast<__fp16>(111), static_cast<__fp16>(224), static_cast<__fp16>(339), static_cast<__fp16>(456), static_cast<__fp16>(0), static_cast<__fp16>(116), static_cast<__fp16>(234), static_cast<__fp16>(354), static_cast<__fp16>(476)};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(1, 2, 1, 1);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0),
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(20), static_cast<__fp16>(21), static_cast<__fp16>(22), static_cast<__fp16>(23), static_cast<__fp16>(24), static_cast<__fp16>(25), static_cast<__fp16>(26), static_cast<__fp16>(27),
//       static_cast<__fp16>(28), static_cast<__fp16>(29), static_cast<__fp16>(30), static_cast<__fp16>(31), static_cast<__fp16>(32), static_cast<__fp16>(33), static_cast<__fp16>(34), static_cast<__fp16>(35), static_cast<__fp16>(36), static_cast<__fp16>(37), static_cast<__fp16>(38), static_cast<__fp16>(39), static_cast<__fp16>(0), static_cast<__fp16>(0),
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0),
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(60), static_cast<__fp16>(61), static_cast<__fp16>(62), static_cast<__fp16>(63), static_cast<__fp16>(64), static_cast<__fp16>(65), static_cast<__fp16>(66), static_cast<__fp16>(67), static_cast<__fp16>(68), static_cast<__fp16>(69),
//       static_cast<__fp16>(70), static_cast<__fp16>(71), static_cast<__fp16>(72), static_cast<__fp16>(73), static_cast<__fp16>(74), static_cast<__fp16>(75), static_cast<__fp16>(76), static_cast<__fp16>(77), static_cast<__fp16>(78), static_cast<__fp16>(79), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0),
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0),
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(100), static_cast<__fp16>(101), static_cast<__fp16>(102), static_cast<__fp16>(103), static_cast<__fp16>(104), static_cast<__fp16>(105), static_cast<__fp16>(106), static_cast<__fp16>(107), static_cast<__fp16>(108), static_cast<__fp16>(109), static_cast<__fp16>(110), static_cast<__fp16>(111),
//       static_cast<__fp16>(112), static_cast<__fp16>(113), static_cast<__fp16>(114), static_cast<__fp16>(115), static_cast<__fp16>(116), static_cast<__fp16>(117), static_cast<__fp16>(118), static_cast<__fp16>(119)};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 1, 1, 1);
//     __fp16 answer_data[] = {
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0),
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0),
//       static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(0), static_cast<__fp16>(40), static_cast<__fp16>(41),
//       static_cast<__fp16>(42), static_cast<__fp16>(43), static_cast<__fp16>(44), static_cast<__fp16>(45), static_cast<__fp16>(46), static_cast<__fp16>(47), static_cast<__fp16>(48), static_cast<__fp16>(49), static_cast<__fp16>(50), static_cast<__fp16>(51), static_cast<__fp16>(52), static_cast<__fp16>(53), static_cast<__fp16>(54), static_cast<__fp16>(55),
//       static_cast<__fp16>(56), static_cast<__fp16>(57), static_cast<__fp16>(58), static_cast<__fp16>(59), static_cast<__fp16>(60), static_cast<__fp16>(61), static_cast<__fp16>(62), static_cast<__fp16>(63), static_cast<__fp16>(64), static_cast<__fp16>(65), static_cast<__fp16>(66), static_cast<__fp16>(67), static_cast<__fp16>(68), static_cast<__fp16>(69),
//       static_cast<__fp16>(70), static_cast<__fp16>(71), static_cast<__fp16>(72), static_cast<__fp16>(73), static_cast<__fp16>(74), static_cast<__fp16>(75), static_cast<__fp16>(76), static_cast<__fp16>(77), static_cast<__fp16>(78), static_cast<__fp16>(79), static_cast<__fp16>(160), static_cast<__fp16>(162), static_cast<__fp16>(164), static_cast<__fp16>(166),
//       static_cast<__fp16>(168), static_cast<__fp16>(170), static_cast<__fp16>(172), static_cast<__fp16>(174), static_cast<__fp16>(176), static_cast<__fp16>(178), static_cast<__fp16>(180), static_cast<__fp16>(182), static_cast<__fp16>(184), static_cast<__fp16>(186), static_cast<__fp16>(188), static_cast<__fp16>(190), static_cast<__fp16>(192), static_cast<__fp16>(194),
//       static_cast<__fp16>(196), static_cast<__fp16>(198), static_cast<__fp16>(200), static_cast<__fp16>(202), static_cast<__fp16>(204), static_cast<__fp16>(206), static_cast<__fp16>(208), static_cast<__fp16>(210), static_cast<__fp16>(212), static_cast<__fp16>(214), static_cast<__fp16>(216), static_cast<__fp16>(218), static_cast<__fp16>(220), static_cast<__fp16>(222),
//       static_cast<__fp16>(224), static_cast<__fp16>(226), static_cast<__fp16>(228), static_cast<__fp16>(230), static_cast<__fp16>(232), static_cast<__fp16>(234), static_cast<__fp16>(236), static_cast<__fp16>(238)};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 5, 1, 4);
//     nntrainer::Tensor t = ranged(3, 5, 1, 4);
//     nntrainer::Tensor m = ranged(3, 1, 1, 4);
//     __fp16 answer_data[] = {static_cast<__fp16>(0), static_cast<__fp16>(1), static_cast<__fp16>(4), static_cast<__fp16>(9), static_cast<__fp16>(0), static_cast<__fp16>(5), static_cast<__fp16>(12), static_cast<__fp16>(21), static_cast<__fp16>(0), static_cast<__fp16>(9),
//                         static_cast<__fp16>(20), static_cast<__fp16>(33), static_cast<__fp16>(0), static_cast<__fp16>(13), static_cast<__fp16>(28), static_cast<__fp16>(45), static_cast<__fp16>(0), static_cast<__fp16>(17), static_cast<__fp16>(36), static_cast<__fp16>(57),
//                         static_cast<__fp16>(80), static_cast<__fp16>(105), static_cast<__fp16>(132), static_cast<__fp16>(161), static_cast<__fp16>(96), static_cast<__fp16>(125), static_cast<__fp16>(156), static_cast<__fp16>(189), static_cast<__fp16>(112), static_cast<__fp16>(145),
//                         static_cast<__fp16>(180), static_cast<__fp16>(217), static_cast<__fp16>(128), static_cast<__fp16>(165), static_cast<__fp16>(204), static_cast<__fp16>(245), static_cast<__fp16>(144), static_cast<__fp16>(185), static_cast<__fp16>(228), static_cast<__fp16>(273),
//                         static_cast<__fp16>(320), static_cast<__fp16>(369), static_cast<__fp16>(420), static_cast<__fp16>(473), static_cast<__fp16>(352), static_cast<__fp16>(405), static_cast<__fp16>(460), static_cast<__fp16>(517), static_cast<__fp16>(384), static_cast<__fp16>(441),
//                         static_cast<__fp16>(500), static_cast<__fp16>(561), static_cast<__fp16>(416), static_cast<__fp16>(477), static_cast<__fp16>(540), static_cast<__fp16>(605), static_cast<__fp16>(448), static_cast<__fp16>(513), static_cast<__fp16>(580), static_cast<__fp16>(649)};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.multiply_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
// }

// TEST(nntrainer_Tensor, multiply_i_broadcast_not_supported_01_n) {
//   nntrainer::Tensor target(3, 1, 3, 1);
//   nntrainer::Tensor target2(3, 1, 3, 3);

//   EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, multiply_i_broadcast_not_broadcastable_02_n) {
//   nntrainer::Tensor target(3, 2, 4, 5);
//   nntrainer::Tensor target2(3, 2, 3, 1);

//   EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, multiply_01_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   nntrainer::Tensor result = input.multiply(0.0);
//   if (result.getValue(0, 0, 1, 1) != 0.0)
//     status = ML_ERROR_RESULT_OUT_OF_RANGE;
//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, multiply_02_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor result = input.multiply(input);

//   __fp16 *data = result.getData();
//   ASSERT_NE(nullptr, data);
//   __fp16 *indata = input.getData();
//   ASSERT_NE(nullptr, indata);

//   for (int i = 0; i < batch * height * width; ++i) {
//     if (data[i] != indata[i] * indata[i]) {
//       status = ML_ERROR_RESULT_OUT_OF_RANGE;
//       break;
//     }
//   }

//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, multiply_03_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor test(batch - 1, height - 1, width - 1);

//   EXPECT_THROW({ input.multiply(test); }, std::invalid_argument);
// }

// TEST(nntrainer_Tensor, multiply_04_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(batch, channel, height, 2 * width);
//   nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
//   nntrainer::Tensor test(dim);

//   EXPECT_THROW(shared_input.multiply(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, multiply_05_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   nntrainer::Tensor test(batch, channel, height, 2 * width);
//   nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

//   EXPECT_THROW(input.multiply(shared_test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, multiply_06_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim, false);
//   nntrainer::Tensor test(dim);
//   GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

//   EXPECT_THROW(input.multiply(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, multiply_07_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
//   nntrainer::Tensor test(dim, false);

//   EXPECT_THROW(input.multiply(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, multiply_08_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
//   nntrainer::Tensor test(dim);
//   GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
//   nntrainer::Tensor output(dim, false);

//   EXPECT_THROW(input.multiply(test, output), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, multiply___fp16_01_p) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor expected(batch, channel, height, width);
//   GEN_TEST_INPUT(expected, (i * (batch * height) + j * (width) + k + 1) * 2);

//   nntrainer::Tensor result = input.multiply(2.0);

//   EXPECT_EQ(result, expected);
// }

// TEST(nntrainer_Tensor, divide_i_01_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   nntrainer::Tensor original;
//   original.copy(input);

//   status = input.divide_i((__fp16)2.0);
//   EXPECT_EQ(status, ML_ERROR_NONE);

//   __fp16 *data = original.getData();
//   ASSERT_NE(nullptr, data);
//   __fp16 *indata = input.getData();
//   ASSERT_NE(nullptr, indata);

//   for (int i = 0; i < batch * height * width * channel; ++i) {
//     EXPECT_FLOAT_EQ(data[i], indata[i] + indata[i]);
//   }
// }

// TEST(nntrainer_Tensor, divide_i_02_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   status = input.divide_i(input);
//   EXPECT_EQ(status, ML_ERROR_NONE);
//   __fp16 *indata = input.getData();
//   ASSERT_NE(nullptr, indata);

//   for (int i = 0; i < batch * height * width * channel; ++i) {
//     EXPECT_FLOAT_EQ(indata[i], __fp16(1.0));
//   }
// }

// TEST(nntrainer_Tensor, divide_i_01_n) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   status = input.divide_i((__fp16)0);
//   EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, divide_i_02_n) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   nntrainer::Tensor original(batch, channel, height - 2, width - 1);

//   status = input.divide_i(original);
//   EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, divide_01_p) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor result = input.divide(1.0);

//   __fp16 *previous = input.getData();
//   ASSERT_NE(nullptr, previous);
//   __fp16 *data = result.getData();
//   ASSERT_NE(nullptr, data);

//   for (int i = 0; i < batch * height * width * channel; ++i) {
//     EXPECT_FLOAT_EQ(data[i], previous[i]);
//   }
// }

// TEST(nntrainer_Tensor, divide_02_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   EXPECT_THROW({ input.divide(0.0); }, std::invalid_argument);
// }

// TEST(nntrainer_Tensor, divide_03_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1);

//   EXPECT_THROW({ input.divide(test); }, std::invalid_argument);
// }

// TEST(nntrainer_Tensor, divide_04_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(batch, channel, height, 2 * width);
//   nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
//   nntrainer::Tensor test(dim);

//   EXPECT_THROW(shared_input.divide(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, divide_05_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   nntrainer::Tensor test(batch, channel, height, 2 * width);
//   nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

//   EXPECT_THROW(input.divide(shared_test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, divide_06_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim, false);
//   nntrainer::Tensor test(dim);
//   GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

//   EXPECT_THROW(input.divide(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, divide_07_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
//   nntrainer::Tensor test(dim, false);

//   EXPECT_THROW(input.divide(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, divide_08_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
//   nntrainer::Tensor test(dim);
//   GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
//   nntrainer::Tensor output(dim, false);

//   EXPECT_THROW(input.divide(test, output), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, divide_i_broadcast_01_p) {
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(1, 2, 4, 5);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       1.0,       1.0,       41.0,      21.0,
//       14.333333, 11.0,      9.0,       7.6666665, 6.714286,  6.0,
//       5.4444447, 5.0,       4.6363635, 4.3333335, 4.076923,  3.857143,
//       3.6666667, 3.5,       3.3529413, 3.2222223, 3.1052632, 3.0,
//       2.9047618, 2.8181818, 2.7391305, 2.6666667, 2.6,       2.5384614,
//       2.4814816, 2.4285715, 2.3793104, 2.3333333, 2.2903225, 2.25,
//       2.2121212, 2.1764705, 2.142857,  2.1111112, 2.0810812, 2.0526316,
//       2.025641,  2.0,       81.0,      41.0,      27.666666, 21.0,
//       17.0,      14.333333, 12.428572, 11.0,      9.888889,  9.0,
//       8.272727,  7.6666665, 7.1538463, 6.714286,  6.3333335, 6.0,
//       5.7058825, 5.4444447, 5.2105265, 5.0,       4.8095236, 4.6363635,
//       4.478261,  4.3333335, 4.2,       4.076923,  3.9629629, 3.857143,
//       3.7586207, 3.6666667, 3.580645,  3.5,       3.4242425, 3.3529413,
//       3.2857144, 3.2222223, 3.162162,  3.1052632, 3.0512822, 3.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(3, 1, 4, 5);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       1.0,       1.0,       1.0,       1.0,
//       1.0,       1.0,       21.0,      11.0,      7.6666665, 6.0,
//       5.0,       4.3333335, 3.857143,  3.5,       3.2222223, 3.0,
//       2.8181818, 2.6666667, 2.5384614, 2.4285715, 2.3333333, 2.25,
//       2.1764705, 2.1111112, 2.0526316, 2.0,       1.9523809, 1.9090909,
//       1.8695652, 1.8333334, 1.8,       1.7692307, 1.7407408, 1.7142857,
//       1.6896552, 1.6666666, 1.6451613, 1.625,     1.6060606, 1.5882353,
//       1.5714285, 1.5555556, 1.5405406, 1.5263158, 1.5128205, 1.5,
//       2.9047618, 2.8181818, 2.7391305, 2.6666667, 2.6,       2.5384614,
//       2.4814816, 2.4285715, 2.3793104, 2.3333333, 2.2903225, 2.25,
//       2.2121212, 2.1764705, 2.142857,  2.1111112, 2.0810812, 2.0526316,
//       2.025641,  2.0,       1.9756098, 1.9523809, 1.9302325, 1.9090909,
//       1.8888888, 1.8695652, 1.8510638, 1.8333334, 1.8163265, 1.8,
//       1.7843137, 1.7692307, 1.754717,  1.7407408, 1.7272727, 1.7142857,
//       1.7017543, 1.6896552, 1.6779661, 1.6666666, 2.4634147, 2.4285715,
//       2.3953488, 2.3636363, 2.3333333, 2.3043478, 2.2765958, 2.25,
//       2.2244897, 2.2,       2.1764705, 2.1538463, 2.1320755, 2.1111112,
//       2.090909,  2.0714285, 2.0526316, 2.0344827, 2.0169492, 2.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(3, 2, 4, 1);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,       2.0,       3.0,       4.0,       5.0,       3.0,
//       3.5,       4.0,       4.5,       5.0,       3.6666667, 4.0,
//       4.3333335, 4.6666665, 5.0,       4.0,       4.25,      4.5,
//       4.75,      5.0,       4.2,       4.4,       4.6,       4.8,
//       5.0,       4.3333335, 4.5,       4.6666665, 4.8333335, 5.0,
//       4.428571,  4.571429,  4.714286,  4.857143,  5.0,       4.5,
//       4.625,     4.75,      4.875,     5.0,       4.5555553, 4.6666665,
//       4.7777777, 4.888889,  5.0,       4.6,       4.7,       4.8,
//       4.9,       5.0,       4.6363635, 4.7272725, 4.818182,  4.909091,
//       5.0,       4.6666665, 4.75,      4.8333335, 4.9166665, 5.0,
//       4.6923075, 4.769231,  4.8461537, 4.923077,  5.0,       4.714286,
//       4.785714,  4.857143,  4.928571,  5.0,       4.733333,  4.8,
//       4.866667,  4.9333334, 5.0,       4.75,      4.8125,    4.875,
//       4.9375,    5.0,       4.7647057, 4.8235292, 4.882353,  4.9411764,
//       5.0,       4.7777777, 4.8333335, 4.888889,  4.9444447, 5.0,
//       4.7894735, 4.8421054, 4.894737,  4.9473686, 5.0,       4.8,
//       4.85,      4.9,       4.95,      5.0,       4.8095236, 4.857143,
//       4.904762,  4.952381,  5.0,       4.818182,  4.8636365, 4.909091,
//       4.9545455, 5.0,       4.826087,  4.869565,  4.9130435, 4.9565215,
//       5.0,       4.8333335, 4.875,     4.9166665, 4.9583335, 5.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(3, 1, 1, 5);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,       1.0,       1.0,       1.0,       1.0,       6.0,
//       3.5,       2.6666667, 2.25,      2.0,       11.0,      6.0,
//       4.3333335, 3.5,       3.0,       16.0,      8.5,       6.0,
//       4.75,      4.0,       21.0,      11.0,      7.6666665, 6.0,
//       5.0,       26.0,      13.5,      9.333333,  7.25,      6.0,
//       31.0,      16.0,      11.0,      8.5,       7.0,       36.0,
//       18.5,      12.666667, 9.75,      8.0,       6.8333335, 6.0,
//       5.375,     4.888889,  4.5,       7.6666665, 6.714286,  6.0,
//       5.4444447, 5.0,       8.5,       7.428571,  6.625,     6.0,
//       5.5,       9.333333,  8.142858,  7.25,      6.5555553, 6.0,
//       10.166667, 8.857142,  7.875,     7.111111,  6.5,       11.0,
//       9.571428,  8.5,       7.6666665, 7.0,       11.833333, 10.285714,
//       9.125,     8.222222,  7.5,       12.666667, 11.0,      9.75,
//       8.777778,  8.0,       7.3636365, 6.8333335, 6.3846154, 6.0,
//       5.6666665, 7.818182,  7.25,      6.769231,  6.357143,  6.0,
//       8.272727,  7.6666665, 7.1538463, 6.714286,  6.3333335, 8.727273,
//       8.083333,  7.5384617, 7.071429,  6.6666665, 9.181818,  8.5,
//       7.923077,  7.428571,  7.0,       9.636364,  8.916667,  8.307693,
//       7.785714,  7.3333335, 10.090909, 9.333333,  8.692307,  8.142858,
//       7.6666665, 10.545455, 9.75,      9.076923,  8.5,       8.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(1, 2, 1, 5);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,       1.0,       1.0,       1.0,       1.0,       6.0,
//       3.5,       2.6666667, 2.25,      2.0,       11.0,      6.0,
//       4.3333335, 3.5,       3.0,       16.0,      8.5,       6.0,
//       4.75,      4.0,       3.5,       3.142857,  2.875,     2.6666667,
//       2.5,       4.3333335, 3.857143,  3.5,       3.2222223, 3.0,
//       5.1666665, 4.571429,  4.125,     3.7777777, 3.5,       6.0,
//       5.285714,  4.75,      4.3333335, 4.0,       41.0,      21.0,
//       14.333333, 11.0,      9.0,       46.0,      23.5,      16.0,
//       12.25,     10.0,      51.0,      26.0,      17.666666, 13.5,
//       11.0,      56.0,      28.5,      19.333334, 14.75,     12.0,
//       10.166667, 8.857142,  7.875,     7.111111,  6.5,       11.0,
//       9.571428,  8.5,       7.6666665, 7.0,       11.833333, 10.285714,
//       9.125,     8.222222,  7.5,       12.666667, 11.0,      9.75,
//       8.777778,  8.0,       81.0,      41.0,      27.666666, 21.0,
//       17.0,      86.0,      43.5,      29.333334, 22.25,     18.0,
//       91.0,      46.0,      31.0,      23.5,      19.0,      96.0,
//       48.5,      32.666668, 24.75,     20.0,      16.833334, 14.571428,
//       12.875,    11.555555, 10.5,      17.666666, 15.285714, 13.5,
//       12.111111, 11.0,      18.5,      16.0,      14.125,    12.666667,
//       11.5,      19.333334, 16.714285, 14.75,     13.222222, 12.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(3, 1, 4, 1);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,       2.0,       3.0,       4.0,       5.0,       3.0,
//       3.5,       4.0,       4.5,       5.0,       3.6666667, 4.0,
//       4.3333335, 4.6666665, 5.0,       4.0,       4.25,      4.5,
//       4.75,      5.0,       21.0,      22.0,      23.0,      24.0,
//       25.0,      13.0,      13.5,      14.0,      14.5,      15.0,
//       10.333333, 10.666667, 11.0,      11.333333, 11.666667, 9.0,
//       9.25,      9.5,       9.75,      10.0,      8.2,       8.4,
//       8.6,       8.8,       9.0,       7.6666665, 7.8333335, 8.0,
//       8.166667,  8.333333,  7.285714,  7.428571,  7.571429,  7.714286,
//       7.857143,  7.0,       7.125,     7.25,      7.375,     7.5,
//       12.2,      12.4,      12.6,      12.8,      13.0,      11.0,
//       11.166667, 11.333333, 11.5,      11.666667, 10.142858, 10.285714,
//       10.428572, 10.571428, 10.714286, 9.5,       9.625,     9.75,
//       9.875,     10.0,      9.0,       9.111111,  9.222222,  9.333333,
//       9.444445,  8.6,       8.7,       8.8,       8.9,       9.0,
//       8.272727,  8.363636,  8.454545,  8.545455,  8.636364,  8.0,
//       8.083333,  8.166667,  8.25,      8.333333,  11.222222, 11.333333,
//       11.444445, 11.555555, 11.666667, 10.6,      10.7,      10.8,
//       10.9,      11.0,      10.090909, 10.181818, 10.272727, 10.363636,
//       10.454545, 9.666667,  9.75,      9.833333,  9.916667,  10.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(1, 1, 1, 5);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,   1.0,  1.0,       1.0,  1.0,  6.0,   3.5,  2.6666667, 2.25,  2.0,
//       11.0,  6.0,  4.3333335, 3.5,  3.0,  16.0,  8.5,  6.0,       4.75,  4.0,
//       21.0,  11.0, 7.6666665, 6.0,  5.0,  26.0,  13.5, 9.333333,  7.25,  6.0,
//       31.0,  16.0, 11.0,      8.5,  7.0,  36.0,  18.5, 12.666667, 9.75,  8.0,
//       41.0,  21.0, 14.333333, 11.0, 9.0,  46.0,  23.5, 16.0,      12.25, 10.0,
//       51.0,  26.0, 17.666666, 13.5, 11.0, 56.0,  28.5, 19.333334, 14.75, 12.0,
//       61.0,  31.0, 21.0,      16.0, 13.0, 66.0,  33.5, 22.666666, 17.25, 14.0,
//       71.0,  36.0, 24.333334, 18.5, 15.0, 76.0,  38.5, 26.0,      19.75, 16.0,
//       81.0,  41.0, 27.666666, 21.0, 17.0, 86.0,  43.5, 29.333334, 22.25, 18.0,
//       91.0,  46.0, 31.0,      23.5, 19.0, 96.0,  48.5, 32.666668, 24.75, 20.0,
//       101.0, 51.0, 34.333332, 26.0, 21.0, 106.0, 53.5, 36.0,      27.25, 22.0,
//       111.0, 56.0, 37.666668, 28.5, 23.0, 116.0, 58.5, 39.333332, 29.75, 24.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(1, 2, 1, 1);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,  2.0,  3.0,  4.0,   5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
//       13.0, 14.0, 15.0, 16.0,  17.0, 18.0, 19.0, 20.0, 10.5, 11.0, 11.5, 12.0,
//       12.5, 13.0, 13.5, 14.0,  14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0,
//       18.5, 19.0, 19.5, 20.0,  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
//       49.0, 50.0, 51.0, 52.0,  53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
//       30.5, 31.0, 31.5, 32.0,  32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0,
//       36.5, 37.0, 37.5, 38.0,  38.5, 39.0, 39.5, 40.0, 81.0, 82.0, 83.0, 84.0,
//       85.0, 86.0, 87.0, 88.0,  89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0,
//       97.0, 98.0, 99.0, 100.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0,
//       54.5, 55.0, 55.5, 56.0,  56.5, 57.0, 57.5, 58.0, 58.5, 59.0, 59.5, 60.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(3, 1, 1, 1);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,       2.0,       3.0,  4.0,       5.0,       6.0,
//       7.0,       8.0,       9.0,  10.0,      11.0,      12.0,
//       13.0,      14.0,      15.0, 16.0,      17.0,      18.0,
//       19.0,      20.0,      21.0, 22.0,      23.0,      24.0,
//       25.0,      26.0,      27.0, 28.0,      29.0,      30.0,
//       31.0,      32.0,      33.0, 34.0,      35.0,      36.0,
//       37.0,      38.0,      39.0, 40.0,      20.5,      21.0,
//       21.5,      22.0,      22.5, 23.0,      23.5,      24.0,
//       24.5,      25.0,      25.5, 26.0,      26.5,      27.0,
//       27.5,      28.0,      28.5, 29.0,      29.5,      30.0,
//       30.5,      31.0,      31.5, 32.0,      32.5,      33.0,
//       33.5,      34.0,      34.5, 35.0,      35.5,      36.0,
//       36.5,      37.0,      37.5, 38.0,      38.5,      39.0,
//       39.5,      40.0,      27.0, 27.333334, 27.666666, 28.0,
//       28.333334, 28.666666, 29.0, 29.333334, 29.666666, 30.0,
//       30.333334, 30.666666, 31.0, 31.333334, 31.666666, 32.0,
//       32.333332, 32.666668, 33.0, 33.333332, 33.666668, 34.0,
//       34.333332, 34.666668, 35.0, 35.333332, 35.666668, 36.0,
//       36.333332, 36.666668, 37.0, 37.333332, 37.666668, 38.0,
//       38.333332, 38.666668, 39.0, 39.333332, 39.666668, 40.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 5, 1, 4);
//     nntrainer::Tensor t = ranged(3, 5, 1, 4);
//     t.add_i(1);
//     nntrainer::Tensor m = ranged(3, 1, 1, 4);
//     m.add_i(1);
//     __fp16 answer_data[] = {
//       1.0,       1.0,       1.0,       1.0,       5.0,       3.0,
//       2.3333333, 2.0,       9.0,       5.0,       3.6666667, 3.0,
//       13.0,      7.0,       5.0,       4.0,       17.0,      9.0,
//       6.3333335, 5.0,       4.2,       3.6666667, 3.2857144, 3.0,
//       5.0,       4.3333335, 3.857143,  3.5,       5.8,       5.0,
//       4.428571,  4.0,       6.6,       5.6666665, 5.0,       4.5,
//       7.4,       6.3333335, 5.571429,  5.0,       4.5555553, 4.2,
//       3.909091,  3.6666667, 5.0,       4.6,       4.2727275, 4.0,
//       5.4444447, 5.0,       4.6363635, 4.3333335, 5.888889,  5.4,
//       5.0,       4.6666665, 6.3333335, 5.8,       5.3636365, 5.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.divide_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
// }

// TEST(nntrainer_Tensor, divide_i_broadcast_not_supported_01_n) {
//   nntrainer::Tensor target(3, 1, 3, 1);
//   nntrainer::Tensor target2(3, 1, 3, 3);

//   EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, divide_i_broadcast_not_broadcastable_02_n) {
//   nntrainer::Tensor target(3, 2, 4, 5);
//   nntrainer::Tensor target2(3, 2, 3, 1);

//   EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, add_i_01_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int height = 3;
//   int width = 10;
//   int channel = 1;

//   nntrainer::Tensor target(batch, channel, height, width);
//   GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

//   nntrainer::Tensor original(batch, channel, height, width);
//   original.copy(target);

//   status = target.add_i(2.1);
//   EXPECT_EQ(status, ML_ERROR_NONE);

//   __fp16 *previous = original.getData();
//   ASSERT_NE(nullptr, previous);
//   __fp16 *data = target.getData();
//   ASSERT_NE(nullptr, data);

//   for (int i = 0; i < batch * height * width; ++i) {
//     EXPECT_FLOAT_EQ(data[i], previous[i] + (__fp16)2.1);
//   }
// }

// TEST(nntrainer_Tensor, add_i_02_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int height = 3;
//   int width = 10;
//   int channel = 1;

//   nntrainer::Tensor target(batch, channel, height, width);
//   GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor original(batch, height, width);
//   original.copy(target);

//   status = target.add_i(target, 3.0);
//   EXPECT_EQ(status, ML_ERROR_NONE);

//   __fp16 *previous = original.getData();
//   ASSERT_NE(nullptr, previous);
//   __fp16 *data = target.getData();
//   ASSERT_NE(nullptr, data);

//   for (int i = 0; i < batch * height * width; ++i) {
//     EXPECT_FLOAT_EQ(data[i], previous[i] * 4.0);
//   }
// }

// /**
//  * @brief operand dimension is not right
//  */
// TEST(nntrainer_Tensor, add_i_01_n) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int height = 3;
//   int width = 10;
//   int channel = 1;

//   nntrainer::Tensor target(batch, channel, height, width);
//   GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor target2(batch, height - 2, width - 3);

//   status = target.add_i(target2);
//   EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, add_i_broadcast_01_p) {
//   nntrainer::TensorDim ref_dim{3, 2, 4, 5};
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(1, 2, 4, 5);
//     __fp16 answer_data[] = {
//       0,   2,   4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,  26,
//       28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,  54,
//       56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,  40,  42,
//       44,  46,  48,  50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,
//       72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,
//       100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 80,  82,  84,  86,
//       88,  90,  92,  94,  96,  98,  100, 102, 104, 106, 108, 110, 112, 114,
//       116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142,
//       144, 146, 148, 150, 152, 154, 156, 158};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 1, 4, 5);
//     __fp16 answer_data[] = {
//       0,   2,   4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,  26,
//       28,  30,  32,  34,  36,  38,  20,  22,  24,  26,  28,  30,  32,  34,
//       36,  38,  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62,
//       64,  66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,
//       92,  94,  96,  98,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,
//       100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
//       128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154,
//       156, 158, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162,
//       164, 166, 168, 170, 172, 174, 176, 178};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 2, 4, 1);
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   6,   7,   8,   9,   10,  12,  13,  14,  15,
//       16,  18,  19,  20,  21,  22,  24,  25,  26,  27,  28,  30,  31,  32,
//       33,  34,  36,  37,  38,  39,  40,  42,  43,  44,  45,  46,  48,  49,
//       50,  51,  52,  54,  55,  56,  57,  58,  60,  61,  62,  63,  64,  66,
//       67,  68,  69,  70,  72,  73,  74,  75,  76,  78,  79,  80,  81,  82,
//       84,  85,  86,  87,  88,  90,  91,  92,  93,  94,  96,  97,  98,  99,
//       100, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116,
//       117, 118, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 132, 133,
//       134, 135, 136, 138, 139, 140, 141, 142};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 1, 1, 5);
//     __fp16 answer_data[] = {
//       0,   2,   4,   6,   8,   5,   7,   9,   11,  13,  10,  12,  14,  16,
//       18,  15,  17,  19,  21,  23,  20,  22,  24,  26,  28,  25,  27,  29,
//       31,  33,  30,  32,  34,  36,  38,  35,  37,  39,  41,  43,  45,  47,
//       49,  51,  53,  50,  52,  54,  56,  58,  55,  57,  59,  61,  63,  60,
//       62,  64,  66,  68,  65,  67,  69,  71,  73,  70,  72,  74,  76,  78,
//       75,  77,  79,  81,  83,  80,  82,  84,  86,  88,  90,  92,  94,  96,
//       98,  95,  97,  99,  101, 103, 100, 102, 104, 106, 108, 105, 107, 109,
//       111, 113, 110, 112, 114, 116, 118, 115, 117, 119, 121, 123, 120, 122,
//       124, 126, 128, 125, 127, 129, 131, 133};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(1, 2, 1, 5);
//     __fp16 answer_data[] = {
//       0,   2,   4,   6,   8,   5,   7,   9,   11,  13,  10,  12,  14,  16,
//       18,  15,  17,  19,  21,  23,  25,  27,  29,  31,  33,  30,  32,  34,
//       36,  38,  35,  37,  39,  41,  43,  40,  42,  44,  46,  48,  40,  42,
//       44,  46,  48,  45,  47,  49,  51,  53,  50,  52,  54,  56,  58,  55,
//       57,  59,  61,  63,  65,  67,  69,  71,  73,  70,  72,  74,  76,  78,
//       75,  77,  79,  81,  83,  80,  82,  84,  86,  88,  80,  82,  84,  86,
//       88,  85,  87,  89,  91,  93,  90,  92,  94,  96,  98,  95,  97,  99,
//       101, 103, 105, 107, 109, 111, 113, 110, 112, 114, 116, 118, 115, 117,
//       119, 121, 123, 120, 122, 124, 126, 128};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 1, 4, 1);
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   6,   7,   8,   9,   10,  12,  13,  14,  15,
//       16,  18,  19,  20,  21,  22,  20,  21,  22,  23,  24,  26,  27,  28,
//       29,  30,  32,  33,  34,  35,  36,  38,  39,  40,  41,  42,  44,  45,
//       46,  47,  48,  50,  51,  52,  53,  54,  56,  57,  58,  59,  60,  62,
//       63,  64,  65,  66,  64,  65,  66,  67,  68,  70,  71,  72,  73,  74,
//       76,  77,  78,  79,  80,  82,  83,  84,  85,  86,  88,  89,  90,  91,
//       92,  94,  95,  96,  97,  98,  100, 101, 102, 103, 104, 106, 107, 108,
//       109, 110, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 120, 121,
//       122, 123, 124, 126, 127, 128, 129, 130};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(1, 1, 1, 5);
//     __fp16 answer_data[] = {
//       0,   2,   4,   6,   8,   5,   7,   9,   11,  13,  10,  12,  14,  16,
//       18,  15,  17,  19,  21,  23,  20,  22,  24,  26,  28,  25,  27,  29,
//       31,  33,  30,  32,  34,  36,  38,  35,  37,  39,  41,  43,  40,  42,
//       44,  46,  48,  45,  47,  49,  51,  53,  50,  52,  54,  56,  58,  55,
//       57,  59,  61,  63,  60,  62,  64,  66,  68,  65,  67,  69,  71,  73,
//       70,  72,  74,  76,  78,  75,  77,  79,  81,  83,  80,  82,  84,  86,
//       88,  85,  87,  89,  91,  93,  90,  92,  94,  96,  98,  95,  97,  99,
//       101, 103, 100, 102, 104, 106, 108, 105, 107, 109, 111, 113, 110, 112,
//       114, 116, 118, 115, 117, 119, 121, 123};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(1, 2, 1, 1);
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
//       14,  15,  16,  17,  18,  19,  21,  22,  23,  24,  25,  26,  27,  28,
//       29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  40,  41,
//       42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
//       56,  57,  58,  59,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
//       71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  80,  81,  82,  83,
//       84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
//       98,  99,  101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
//       113, 114, 115, 116, 117, 118, 119, 120};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 1, 1, 1);
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
//       14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
//       28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  41,  42,
//       43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
//       57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
//       71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  82,  83,  84,  85,
//       86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
//       100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
//       114, 115, 116, 117, 118, 119, 120, 121};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(1, 1, 1, 1);
//     m.add_i(1.0);
//     __fp16 answer_data[] = {
//       1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
//       15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,
//       29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
//       43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,
//       57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
//       71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
//       85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,
//       99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
//       113, 114, 115, 116, 117, 118, 119, 120};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 5, 1, 4);
//     nntrainer::Tensor t = ranged(3, 5, 1, 4);
//     nntrainer::Tensor m = ranged(3, 1, 1, 4);
//     __fp16 answer_data[] = {0,  2,  4,  6,  4,  6,  8,  10, 8,  10, 12, 14,
//                            12, 14, 16, 18, 16, 18, 20, 22, 24, 26, 28, 30,
//                            28, 30, 32, 34, 32, 34, 36, 38, 36, 38, 40, 42,
//                            40, 42, 44, 46, 48, 50, 52, 54, 52, 54, 56, 58,
//                            56, 58, 60, 62, 60, 62, 64, 66, 64, 66, 68, 70};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(1, 1, 2, 1);
//     nntrainer::Tensor t = ranged(1, 1, 2, 1);
//     nntrainer::Tensor m = ranged(1, 1, 2, 1);
//     __fp16 answer_data[] = {0.0, 2.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(16, 1, 1, 1);
//     nntrainer::Tensor t = ranged(16, 1, 1, 1);
//     nntrainer::Tensor m = ranged(1, 1, 1, 1);
//     __fp16 answer_data[] = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
//                            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
//     nntrainer::Tensor answer(ref_dim, answer_data);
//     int status = t.add_i(m);
//     EXPECT_EQ(status, ML_ERROR_NONE);
//     EXPECT_EQ(t, answer);
//   }
// }

// TEST(nntrainer_Tensor, add_i_broadcast_not_supported_01_n) {
//   nntrainer::Tensor target(3, 1, 3, 1);
//   nntrainer::Tensor target2(3, 1, 3, 3);

//   EXPECT_EQ(target.add_i(target2), ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, add_i_broadcast_not_broadcastable_02_n) {
//   nntrainer::Tensor target(3, 2, 4, 5);
//   nntrainer::Tensor target2(3, 2, 3, 1);

//   EXPECT_EQ(target.add_i(target2), ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, add_01_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor result = input.add(1.0);

//   __fp16 *data = result.getData();
//   ASSERT_NE(nullptr, data);
//   __fp16 *indata = input.getData();
//   ASSERT_NE(nullptr, indata);

//   for (int i = 0; i < batch * height * width; ++i) {
//     if (data[i] != indata[i] + (__fp16)1.0) {
//       status = ML_ERROR_RESULT_OUT_OF_RANGE;
//       break;
//     }
//   }

//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, add_02_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor result = input.add(input);

//   __fp16 *data = result.getData();
//   ASSERT_NE(nullptr, data);
//   __fp16 *indata = input.getData();
//   ASSERT_NE(nullptr, indata);

//   for (int i = 0; i < batch * height * width; ++i) {
//     if (data[i] != indata[i] + indata[i]) {
//       status = ML_ERROR_RESULT_OUT_OF_RANGE;
//       break;
//     }
//   }

//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, add_03_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1);

//   EXPECT_THROW({ input.add(test); }, std::invalid_argument);
// }

// TEST(nntrainer_Tensor, add_04_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(batch, channel, height, 2 * width);
//   nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
//   nntrainer::Tensor test(dim);

//   EXPECT_THROW(shared_input.add(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, add_05_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   nntrainer::Tensor test(batch, channel, height, 2 * width);
//   nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

//   EXPECT_THROW(input.add(shared_test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, add_06_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim, false);
//   nntrainer::Tensor test(dim);
//   GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

//   EXPECT_THROW(input.add(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, add_07_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
//   nntrainer::Tensor test(dim, false);

//   EXPECT_THROW(input.add(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, add_08_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
//   nntrainer::Tensor test(dim);
//   GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
//   nntrainer::Tensor output(dim, false);

//   EXPECT_THROW(input.add(test, output), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, pow_01_p) {

//   nntrainer::Tensor input = constant(4.0, 3, 2, 4, 5);

//   nntrainer::Tensor actual, expected;

//   actual = input.pow(0.5f);
//   expected = constant(2.0, 3, 2, 4, 5);
//   EXPECT_EQ(actual, expected);

//   actual = input.pow(2.0f);
//   expected = constant(16.0, 3, 2, 4, 5);
//   EXPECT_EQ(actual, expected);

//   actual = input.pow(-0.5f);
//   expected = constant(0.5, 3, 2, 4, 5);
//   EXPECT_EQ(actual, expected);
// }

// TEST(nntrainer_Tensor, erf_01_p) {
//   int batch = 1;
//   int channel = 1;
//   int height = 2;
//   int width = 2;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, k + l * 0.5 + 0.5);
//   nntrainer::Tensor actual = input.erf();
//   nntrainer::Tensor expected(
//     std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//       {{{{0.5205, 0.8427}, {0.966105, 0.995322}}}}));

//   EXPECT_EQ(actual, expected);
// }

// TEST(nntrainer_Tensor, subtract_i_01_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int height = 3;
//   int width = 10;
//   int channel = 1;

//   nntrainer::Tensor target(batch, channel, height, width);
//   GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

//   nntrainer::Tensor original(batch, height, width);
//   original.copy(target);

//   status = target.subtract_i(2.1);
//   EXPECT_EQ(status, ML_ERROR_NONE);

//   __fp16 *previous = original.getData();
//   ASSERT_NE(nullptr, previous);
//   __fp16 *data = target.getData();
//   ASSERT_NE(nullptr, data);

//   for (int i = 0; i < batch * height * width; ++i) {
//     EXPECT_FLOAT_EQ(data[i], previous[i] - (__fp16)2.1);
//   }
// }

// TEST(nntrainer_Tensor, subtract_i_02_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int height = 3;
//   int width = 10;
//   int channel = 1;

//   nntrainer::Tensor target(batch, channel, height, width);
//   GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

//   status = target.subtract_i(target);
//   EXPECT_EQ(status, ML_ERROR_NONE);

//   __fp16 *data = target.getData();
//   ASSERT_NE(nullptr, data);

//   for (int i = 0; i < batch * height * width; ++i) {
//     EXPECT_FLOAT_EQ(data[i], 0);
//   }
// }

// TEST(nntrainer_Tensor, subtract_i_03_n) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int height = 3;
//   int width = 10;
//   int channel = 1;

//   nntrainer::Tensor target(batch, channel, height, width);
//   GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

//   nntrainer::Tensor target2(batch, channel, height - 1, width - 3);

//   status = target.subtract_i(target2);
//   EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
// }

// TEST(nntrainer_Tensor, subtract_01_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor result = input.subtract(1.0);

//   __fp16 *data = result.getData();
//   ASSERT_NE(nullptr, data);
//   __fp16 *indata = input.getData();
//   ASSERT_NE(nullptr, indata);

//   for (int i = 0; i < batch * height * width; ++i) {
//     if (data[i] != indata[i] - 1.0) {
//       status = ML_ERROR_RESULT_OUT_OF_RANGE;
//       break;
//     }
//   }

//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, subtract_02_p) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor result = input.subtract(input);

//   EXPECT_EQ(constant(0.0, batch, channel, height, width), result);
// }

// TEST(nntrainer_Tensor, subtract_03_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1);

//   EXPECT_THROW({ input.subtract(test); }, std::invalid_argument);
// }

// TEST(nntrainer_Tensor, subtract_04_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(batch, channel, height, 2 * width);
//   nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
//   nntrainer::Tensor test(dim);

//   EXPECT_THROW(shared_input.subtract(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, subtract_05_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   nntrainer::Tensor test(batch, channel, height, 2 * width);
//   nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

//   EXPECT_THROW(input.subtract(shared_test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, subtract_06_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim, false);
//   nntrainer::Tensor test(dim);
//   GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

//   EXPECT_THROW(input.subtract(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, subtract_07_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
//   nntrainer::Tensor test(dim, false);

//   EXPECT_THROW(input.subtract(test), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, subtract_08_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::TensorDim dim(batch, channel, height, width);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
//   nntrainer::Tensor test(dim);
//   GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
//   nntrainer::Tensor output(dim, false);

//   EXPECT_THROW(input.subtract(test, output), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, subtract___fp16_01_p) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   nntrainer::Tensor expected(batch, channel, height, width);
//   GEN_TEST_INPUT(expected, i * (batch * height) + j * (width) + k);

//   nntrainer::Tensor result = input.subtract(1.0);

//   EXPECT_EQ(result, expected);
// }

// TEST(nntrainer_Tensor, sum_01_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   EXPECT_THROW({ input.sum(4); }, std::out_of_range);
// }

// TEST(nntrainer_Tensor, sum_02_n) {
//   int batch = 3;
//   int channel = 1;
//   int height = 3;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   EXPECT_THROW({ input.sum(-1); }, std::out_of_range);
// }

// TEST(nntrainer_Tensor, sum_02_p) {
//   int batch = 3;
//   int channel = 2;
//   int height = 2;
//   int width = 10;

//   nntrainer::Tensor ans0(
//     std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//       {{{{39, 42, 45, 48, 51, 54, 57, 60, 63, 66},
//          {69, 72, 75, 78, 81, 84, 87, 90, 93, 96}},
//         {{57, 60, 63, 66, 69, 72, 75, 78, 81, 84},
//          {87, 90, 93, 96, 99, 102, 105, 108, 111, 114}}}}));

//   nntrainer::Tensor ans1(
//     std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//       {{{{8, 10, 12, 14, 16, 18, 20, 22, 24, 26},
//          {28, 30, 32, 34, 36, 38, 40, 42, 44, 46}}},
//        {{{32, 34, 36, 38, 40, 42, 44, 46, 48, 50},
//          {52, 54, 56, 58, 60, 62, 64, 66, 68, 70}}},
//        {{{56, 58, 60, 62, 64, 66, 68, 70, 72, 74},
//          {76, 78, 80, 82, 84, 86, 88, 90, 92, 94}}}}));

//   nntrainer::Tensor ans2(
//     std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//       {{{{12, 14, 16, 18, 20, 22, 24, 26, 28, 30}},
//         {{24, 26, 28, 30, 32, 34, 36, 38, 40, 42}}},
//        {{{36, 38, 40, 42, 44, 46, 48, 50, 52, 54}},
//         {{48, 50, 52, 54, 56, 58, 60, 62, 64, 66}}},
//        {{{60, 62, 64, 66, 68, 70, 72, 74, 76, 78}},
//         {{72, 74, 76, 78, 80, 82, 84, 86, 88, 90}}}}));

//   nntrainer::Tensor ans3(
//     std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//       {{{{55}, {155}}, {{115}, {215}}},
//        {{{175}, {275}}, {{235}, {335}}},
//        {{{295}, {395}}, {{355}, {455}}}}));

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height * channel) + j * (batch * height) +
//                           k * (width) + l + 1);

//   nntrainer::Tensor result0 = input.sum(0);
//   nntrainer::Tensor result1 = input.sum(1);
//   nntrainer::Tensor result2 = input.sum(2);
//   nntrainer::Tensor result3 = input.sum(3);

//   EXPECT_EQ(ans0, result0);
//   EXPECT_EQ(ans1, result1);
//   EXPECT_EQ(ans2, result2);
//   EXPECT_EQ(ans3, result3);
// }

// TEST(nntrainer_Tensor, sum_03_p) {
//   const int batch = 3;
//   const int channel = 2;
//   const int height = 1;
//   const int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (height * channel * width) + j * (height * width) +
//                           k * (width) + l + 1);
//   // Test for alpha == 1 and beta == 0 and dimension of reduced axis == 1
//   {
//     nntrainer::Tensor ans_0_1_0(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{63, 66, 69, 72, 75, 78, 81, 84, 87, 90}},
//           {{93, 96, 99, 102, 105, 108, 111, 114, 117, 120}}}}));

//     nntrainer::Tensor ans_1_1_0(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{12, 14, 16, 18, 20, 22, 24, 26, 28, 30}}},
//          {{{52, 54, 56, 58, 60, 62, 64, 66, 68, 70}}},
//          {{{92, 94, 96, 98, 100, 102, 104, 106, 108, 110}}}}));

//     nntrainer::Tensor ans_2_1_0(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
//           {{11, 12, 13, 14, 15, 16, 17, 18, 19, 20}}},
//          {{{21, 22, 23, 24, 25, 26, 27, 28, 29, 30}},
//           {{31, 32, 33, 34, 35, 36, 37, 38, 39, 40}}},
//          {{{41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
//           {{51, 52, 53, 54, 55, 56, 57, 58, 59, 60}}}}));

//     nntrainer::Tensor ans_3_1_0(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{55}}, {{155}}}, {{{255}}, {{355}}}, {{{455}}, {{555}}}}));

//     nntrainer::Tensor result_0_1_0 = input.sum(0, 1);
//     nntrainer::Tensor result_1_1_0 = input.sum(1, 1);
//     nntrainer::Tensor result_2_1_0 = input.sum(2, 1);
//     nntrainer::Tensor result_3_1_0 = input.sum(3, 1);

//     EXPECT_EQ(ans_0_1_0, result_0_1_0);
//     EXPECT_EQ(ans_1_1_0, result_1_1_0);
//     EXPECT_EQ(ans_2_1_0, result_2_1_0);
//     EXPECT_EQ(ans_3_1_0, result_3_1_0);
//   }

//   // Test for alpha == 1 and beta == 2 and dimension of reduced axis == 1
//   {
//     nntrainer::Tensor ans_0_1_2(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{65, 70, 75, 80, 85, 90, 95, 100, 105, 110}},
//           {{115, 120, 125, 130, 135, 140, 145, 150, 155, 160}}}}));

//     nntrainer::Tensor ans_1_1_2(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{14, 18, 22, 26, 30, 34, 38, 42, 46, 50}}},
//          {{{74, 78, 82, 86, 90, 94, 98, 102, 106, 110}}},
//          {{{134, 138, 142, 146, 150, 154, 158, 162, 166, 170}}}}));

//     nntrainer::Tensor ans_2_1_2(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{3, 6, 9, 12, 15, 18, 21, 24, 27, 30}},
//           {{33, 36, 39, 42, 45, 48, 51, 54, 57, 60}}},
//          {{{63, 66, 69, 72, 75, 78, 81, 84, 87, 90}},
//           {{93, 96, 99, 102, 105, 108, 111, 114, 117, 120}}},
//          {{{123, 126, 129, 132, 135, 138, 141, 144, 147, 150}},
//           {{153, 156, 159, 162, 165, 168, 171, 174, 177, 180}}}}));

//     nntrainer::Tensor ans_3_1_2(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{57}}, {{159}}}, {{{261}}, {{363}}}, {{{465}}, {{567}}}}));

//     nntrainer::Tensor output_0_1_2(1, channel, height, width);
//     {
//       const int batch = 1;
//       GEN_TEST_INPUT(output_0_1_2, i * (channel * height * width) +
//                                      j * (height * width) + k * (width) + l +
//                                      1);
//     }
//     nntrainer::Tensor output_1_1_2(batch, 1, height, width);
//     {
//       const int channel = 1;
//       GEN_TEST_INPUT(output_1_1_2, i * (channel * height * width) +
//                                      j * (height * width) + k * (width) + l +
//                                      1);
//     }
//     nntrainer::Tensor output_2_1_2(batch, channel, 1, width);
//     {
//       const int height = 1;
//       GEN_TEST_INPUT(output_2_1_2, i * (channel * height * width) +
//                                      j * (height * width) + k * (width) + l +
//                                      1);
//     }
//     nntrainer::Tensor output_3_1_2(batch, channel, height, 1);
//     {
//       const int width = 1;
//       GEN_TEST_INPUT(output_3_1_2, i * (channel * height * width) +
//                                      j * (height * width) + k * (width) + l +
//                                      1);
//     }
//     nntrainer::Tensor result_0_1_2 = input.sum(0, output_0_1_2, 1, 2);
//     nntrainer::Tensor result_1_1_2 = input.sum(1, output_1_1_2, 1, 2);
//     nntrainer::Tensor result_2_1_2 = input.sum(2, output_2_1_2, 1, 2);
//     nntrainer::Tensor result_3_1_2 = input.sum(3, output_3_1_2, 1, 2);

//     EXPECT_EQ(ans_0_1_2, result_0_1_2);
//     EXPECT_EQ(ans_1_1_2, result_1_1_2);
//     EXPECT_EQ(ans_2_1_2, result_2_1_2);
//     EXPECT_EQ(ans_3_1_2, result_3_1_2);
//   }

//   // Test for alpha == 2 and beta == 0
//   {
//     nntrainer::Tensor ans_0_2_0(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{126, 132, 138, 144, 150, 156, 162, 168, 174, 180}},
//           {{186, 192, 198, 204, 210, 216, 222, 228, 234, 240}}}}));

//     nntrainer::Tensor ans_1_2_0(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{24, 28, 32, 36, 40, 44, 48, 52, 56, 60}}},
//          {{{104, 108, 112, 116, 120, 124, 128, 132, 136, 140}}},
//          {{{184, 188, 192, 196, 200, 204, 208, 212, 216, 220}}}}));

//     nntrainer::Tensor ans_2_2_0(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}},
//           {{22, 24, 26, 28, 30, 32, 34, 36, 38, 40}}},
//          {{{42, 44, 46, 48, 50, 52, 54, 56, 58, 60}},
//           {{62, 64, 66, 68, 70, 72, 74, 76, 78, 80}}},
//          {{{82, 84, 86, 88, 90, 92, 94, 96, 98, 100}},
//           {{102, 104, 106, 108, 110, 112, 114, 116, 118, 120}}}}));

//     nntrainer::Tensor ans_3_2_0(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{110}}, {{310}}}, {{{510}}, {{710}}}, {{{910}}, {{1110}}}}));

//     nntrainer::Tensor result_0_2_0 = input.sum(0, 2);
//     nntrainer::Tensor result_1_2_0 = input.sum(1, 2);
//     nntrainer::Tensor result_2_2_0 = input.sum(2, 2);
//     nntrainer::Tensor result_3_2_0 = input.sum(3, 2);

//     EXPECT_EQ(ans_0_2_0, result_0_2_0);
//     EXPECT_EQ(ans_1_2_0, result_1_2_0);
//     EXPECT_EQ(ans_2_2_0, result_2_2_0);
//     EXPECT_EQ(ans_3_2_0, result_3_2_0);
//   }

//   // Test for alpha == 2 and beta == 2
//   {
//     nntrainer::Tensor ans_0_2_2(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{128, 136, 144, 152, 160, 168, 176, 184, 192, 200}},
//           {{208, 216, 224, 232, 240, 248, 256, 264, 272, 280}}}}));

//     nntrainer::Tensor ans_1_2_2(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{26, 32, 38, 44, 50, 56, 62, 68, 74, 80}}},
//          {{{126, 132, 138, 144, 150, 156, 162, 168, 174, 180}}},
//          {{{226, 232, 238, 244, 250, 256, 262, 268, 274, 280}}}}));

//     nntrainer::Tensor ans_2_2_2(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{4, 8, 12, 16, 20, 24, 28, 32, 36, 40}},
//           {{44, 48, 52, 56, 60, 64, 68, 72, 76, 80}}},
//          {{{84, 88, 92, 96, 100, 104, 108, 112, 116, 120}},
//           {{124, 128, 132, 136, 140, 144, 148, 152, 156, 160}}},
//          {{{164, 168, 172, 176, 180, 184, 188, 192, 196, 200}},
//           {{204, 208, 212, 216, 220, 224, 228, 232, 236, 240}}}}));

//     nntrainer::Tensor ans_3_2_2(
//       std::vector<std::vector<std::vector<std::vector<__fp16>>>>(
//         {{{{112}}, {{314}}}, {{{516}}, {{718}}}, {{{920}}, {{1122}}}}));

//     nntrainer::Tensor output_0_2_2(1, channel, height, width);
//     {
//       const int batch = 1;
//       GEN_TEST_INPUT(output_0_2_2, i * (channel * height * width) +
//                                      j * (height * width) + k * (width) + l +
//                                      1);
//     }
//     nntrainer::Tensor output_1_2_2(batch, 1, height, width);
//     {
//       const int channel = 1;
//       GEN_TEST_INPUT(output_1_2_2, i * (channel * height * width) +
//                                      j * (height * width) + k * (width) + l +
//                                      1);
//     }
//     nntrainer::Tensor output_2_2_2(batch, channel, 1, width);
//     {
//       const int height = 1;
//       GEN_TEST_INPUT(output_2_2_2, i * (channel * height * width) +
//                                      j * (height * width) + k * (width) + l +
//                                      1);
//     }
//     nntrainer::Tensor output_3_2_2(batch, channel, height, 1);
//     {
//       const int width = 1;
//       GEN_TEST_INPUT(output_3_2_2, i * (channel * height * width) +
//                                      j * (height * width) + k * (width) + l +
//                                      1);
//     }
//     nntrainer::Tensor result_0_2_2 = input.sum(0, output_0_2_2, 2, 2);
//     nntrainer::Tensor result_1_2_2 = input.sum(1, output_1_2_2, 2, 2);
//     nntrainer::Tensor result_2_2_2 = input.sum(2, output_2_2_2, 2, 2);
//     nntrainer::Tensor result_3_2_2 = input.sum(3, output_3_2_2, 2, 2);

//     EXPECT_EQ(ans_0_2_2, result_0_2_2);
//     EXPECT_EQ(ans_1_2_2, result_1_2_2);
//     EXPECT_EQ(ans_2_2_2, result_2_2_2);
//     EXPECT_EQ(ans_3_2_2, result_3_2_2);
//   }
// }

// TEST(nntrainer_Tensor, sum_04_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 2;
//   int height = 2;
//   int width = 10;

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (batch * height * channel) + j * (height * width) +
//                           k * width + l + 1);

//   nntrainer::Tensor result = input.sum_by_batch();
//   if (result.getValue(0, 0, 0, 0) != 820 ||
//       result.getValue(1, 0, 0, 0) != 1300 ||
//       result.getValue(2, 0, 0, 0) != 1780)
//     status = ML_ERROR_RESULT_OUT_OF_RANGE;

//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, multiple_sum_invalid_args_01_n) {
//   nntrainer::Tensor t = constant(1.0, 1, 1, 1, 1);
//   EXPECT_THROW(t.sum(std::vector<unsigned int>()), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, multiple_sum_out_of_range_n) {
//   nntrainer::Tensor t = constant(1.0, 1, 1, 1, 1);
//   EXPECT_THROW(t.sum({7}), std::out_of_range);
// }

// TEST(nntrainer_Tensor, multiple_sum_p) {
//   nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7);
//   nntrainer::Tensor actual, expected;

//   actual = t.sum({0, 1});
//   expected = constant(2 * 3, 1, 1, 5, 7);
//   EXPECT_EQ(actual, expected);

//   actual = t.sum({1, 2, 3});
//   expected = constant(3 * 5 * 7, 2, 1, 1, 1);
//   EXPECT_EQ(actual, expected);

//   actual = t.sum({3, 1});
//   expected = constant(7 * 3, 2, 1, 5, 1);
//   EXPECT_EQ(actual, expected);

//   actual = t.sum({3, 1}, 0.5);
//   expected = constant(7 * 3 * 0.5, 2, 1, 5, 1);
//   EXPECT_EQ(actual, expected);
// }

// TEST(nntrainer_Tensor, average_p) {
//   nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7);

//   nntrainer::Tensor actual, expected;

//   actual = t.average();
//   expected = constant(1.0, 1, 1, 1, 1);
//   EXPECT_EQ(actual, expected);

//   int idx = 0;
//   t = t.apply([&](__fp16 in) { return idx++ % 2; });

//   actual = t.average();
//   expected = constant(0.5, 1, 1, 1, 1);
//   EXPECT_EQ(actual, expected);
// }

// TEST(nntrainer_Tensor, average_axis_p) {
//   nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2);
//   int idx = 0;
//   std::function<__fp16(__fp16)> f = [&](__fp16 in) { return idx++ % 2; };
//   t = t.apply(f);

//   nntrainer::Tensor actual, expected;

//   actual = t.average(0);
//   expected = constant(0, 1, 2, 2, 2).apply(f);
//   EXPECT_EQ(actual, expected);

//   actual = t.average(1);
//   expected = constant(0, 2, 1, 2, 2).apply(f);
//   EXPECT_EQ(actual, expected);

//   actual = t.average(2);
//   expected = constant(0, 2, 2, 1, 2).apply(f);
//   EXPECT_EQ(actual, expected);

//   actual = t.average(3);
//   expected = constant(0.5, 2, 2, 2, 1);
//   EXPECT_EQ(actual, expected);
// }

// TEST(nntrainer_Tensor, average_axis_out_of_range_01_n) {
//   nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2);
//   EXPECT_THROW(t.average(-1), std::out_of_range);
// }

// TEST(nntrainer_Tensor, average_axis_out_of_range_02_n) {
//   nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2);
//   EXPECT_THROW(t.average(7), std::out_of_range);
// }

// TEST(nntrainer_Tensor, average_multiple_axes_p) {
//   nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7);
//   nntrainer::Tensor actual, expected;

//   actual = t.average({0, 1, 2});
//   expected = constant(1.0, 1, 1, 1, 7);
//   EXPECT_EQ(actual, expected);

//   actual = t.average({0, 1, 2, 3});
//   expected = constant(1.0, 1, 1, 1, 1);
//   EXPECT_EQ(actual, expected);

//   actual = t.average({3, 1});
//   expected = constant(1.0, 2, 1, 5, 1);
//   EXPECT_EQ(actual, expected);

//   actual = t.average({3, 1, 1, 1, 3});
//   expected = constant(1.0, 2, 1, 5, 1);
//   EXPECT_EQ(actual, expected);
// }

// TEST(nntrainer_Tensor, average_multiple_axes_01_n) {
//   nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7);
//   EXPECT_THROW(t.average({5, 7}), std::out_of_range);
// }

// TEST(nntrainer_Tensor, dot_01_n) {
//   nntrainer::Tensor input(2, 3, 4, 5);
//   nntrainer::Tensor m(1, 3, 4, 5);
//   EXPECT_THROW(nntrainer::Tensor result = input.dot(m), std::runtime_error);
// }

// TEST(nntrainer_Tensor, dot_02_n) {
//   nntrainer::Tensor input(2, 3, 4, 5);
//   nntrainer::Tensor m(1, 3, 4, 5);
//   EXPECT_THROW(nntrainer::Tensor result = input.dot(m, true),
//                std::runtime_error);
// }

// TEST(nntrainer_Tensor, dot_02_p) {
//   nntrainer::Tensor input(2, 3, 4, 5);
//   nntrainer::Tensor m(1, 3, 4, 5);
//   EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, false, true));
// }

// TEST(nntrainer_Tensor, dot_03_p) {
//   nntrainer::Tensor input(1, 3, 4, 5);
//   nntrainer::Tensor m(1, 3, 4, 5);
//   EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, true));
// }

// TEST(nntrainer_Tensor, dot_04_n) {
//   nntrainer::Tensor input(2, 3, 4, 5);
//   nntrainer::Tensor m(1, 1, 4, 5);
//   EXPECT_THROW(nntrainer::Tensor result = input.dot(m), std::runtime_error);
//   EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, false, true));
// }

// TEST(nntrainer_Tensor, dot_05_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 2;
//   int channel = 3;
//   int height = 4;
//   int width = 5;
//   __fp16 ans[2][3][4][24] = {0};

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (channel * width * height) + j * (height * width) +
//                           k * (width) + l + 1);
//   nntrainer::Tensor weight(batch, channel, height, width);
//   GEN_TEST_INPUT(weight, i * (channel * width * height) + j * (height * width) +
//                            k * (width) + l + 1);
//   weight.reshape({1, 1, 24, 5});

//   nntrainer::Tensor result = input.dot(weight, false, true);

//   for (int b = 0; b < batch; b++) {
//     for (int c = 0; c < channel; c++) {
//       for (int h = 0; h < height; h++) {
//         for (int k = 0; k < batch * channel * height; k++) {
//           ans[b][c][h][k] = 0;
//           for (int w = 0; w < width; w++) {
//             __fp16 val1 = input.getValue(b, c, h, w);
//             __fp16 val2 = weight.getValue(0, 0, k, w);
//             ans[b][c][h][k] += val1 * val2;
//           }
//         }
//       }
//     }
//   }

//   for (unsigned int i = 0; i < result.batch(); ++i) {
//     for (unsigned int c = 0; c < result.channel(); ++c) {
//       for (unsigned int j = 0; j < result.height(); ++j) {
//         for (unsigned int k = 0; k < result.width(); ++k) {
//           __fp16 val1 = ans[i][c][j][k];
//           __fp16 val2 = result.getValue(i, c, j, k);
//           if (val1 != val2) {
//             status = ML_ERROR_RESULT_OUT_OF_RANGE;
//             goto end_dot_01_p;
//           }
//         }
//       }
//     }
//   }
// end_dot_01_p:
//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, dot_06_p) {
//   int status = ML_ERROR_NONE;
//   int batch = 3;
//   int channel = 1;
//   int height = 1;
//   int width = 3;
//   __fp16 ans[3][1][1][3] = {
//     {{{30, 36, 42}}}, {{{66, 81, 96}}}, {{{102, 126, 150}}}};

//   nntrainer::Tensor input(batch, channel, height, width);
//   GEN_TEST_INPUT(input, i * (channel * width * height) + j * (height * width) +
//                           k * (width) + l + 1);

//   nntrainer::Tensor result = input.dot(input);

//   for (unsigned int i = 0; i < result.batch(); ++i) {
//     for (unsigned int j = 0; j < result.height(); ++j) {
//       for (unsigned int k = 0; k < result.width(); ++k) {
//         if (ans[i][0][j][k] != result.getValue(i, 0, j, k)) {
//           status = ML_ERROR_RESULT_OUT_OF_RANGE;
//           goto end_dot_01_p;
//         }
//       }
//     }
//   }
// end_dot_01_p:
//   EXPECT_EQ(status, ML_ERROR_NONE);
// }

// TEST(nntrainer_Tensor, dot_transpose_p) {
//   {
//     __fp16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
//     __fp16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
//     __fp16 answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
//                            92, 113, 134, 155, 128, 158, 188, 218};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
//     __fp16 answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
//                            92, 113, 134, 155, 128, 158, 188, 218};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
//     __fp16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
//     __fp16 answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
//                            92, 113, 134, 155, 128, 158, 188, 218};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
//     __fp16 answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
//                            92, 113, 134, 155, 128, 158, 188, 218};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 1, 4, 2, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
//     __fp16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 1, 4, 2, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
//     __fp16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
//     __fp16 b_data[] = {0, 2, 4, 1, 3, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
//     __fp16 answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
//     __fp16 answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
//     __fp16 b_data[] = {0, 2, 4, 1, 3, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
//     __fp16 answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
//     __fp16 answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 1, 4, 2, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
//     __fp16 b_data[] = {0, 2, 4, 1, 3, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
//     __fp16 answer_data[] = {10, 13, 28, 40};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 1, 4, 2, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
//     __fp16 answer_data[] = {10, 13, 28, 40};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
//     __fp16 b_data[] = {0, 2, 4, 1, 3, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
//     __fp16 answer_data[] = {10, 13, 28, 40};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
//     __fp16 answer_data[] = {10, 13, 28, 40};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
// }

// TEST(nntrainer_Tensor, dot_shortcuts_p) {
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
//     __fp16 answer_data[] = {5};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
//     __fp16 answer_data[] = {5};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
//     __fp16 answer_data[] = {5};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
//     __fp16 answer_data[] = {5};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
//     __fp16 answer_data[] = {5, 14};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 1, 4, 2, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
//     __fp16 answer_data[] = {5, 14};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
//     __fp16 answer_data[] = {5, 14};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 1, 4, 2, 5};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
//     __fp16 answer_data[] = {5, 14};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
//     __fp16 answer_data[] = {5, 14, 23, 32};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1), b_data);
//     __fp16 answer_data[] = {5, 14, 23, 32};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
//     __fp16 answer_data[] = {5, 14, 23, 32};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4), a_data);
//     __fp16 b_data[] = {0, 1, 2};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3), b_data);
//     __fp16 answer_data[] = {5, 14, 23, 32};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
//     __fp16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
//     __fp16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
//     __fp16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
//     __fp16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3), b_data);
//     __fp16 answer_data[] = {20, 23, 26, 29};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
//     __fp16 answer_data[] = {10, 13};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
//     __fp16 b_data[] = {0, 1, 2, 3, 4, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2), b_data);
//     __fp16 answer_data[] = {10, 13};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, false);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3), a_data);
//     __fp16 b_data[] = {0, 2, 4, 1, 3, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
//     __fp16 answer_data[] = {10, 13};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, false, true);
//     EXPECT_EQ(ret, answer);
//   }
//   {
//     __fp16 a_data[] = {0, 1, 2};
//     nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1), a_data);
//     __fp16 b_data[] = {0, 2, 4, 1, 3, 5};
//     nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3), b_data);
//     __fp16 answer_data[] = {10, 13};
//     nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2), answer_data);
//     nntrainer::Tensor ret = a.dot(b, true, true);
//     EXPECT_EQ(ret, answer);
//   }
// }

// TEST(nntrainer_Tensor, transpose_p) {
//   nntrainer::TensorDim ref_dim(3, 2, 4, 5);

//   /// plain transpose
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
//       14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
//       28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
//       42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
//       56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
//       70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
//       84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
//       98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
//       112, 113, 114, 115, 116, 117, 118, 119};
//     nntrainer::Tensor answer({3, 2, 4, 5}, answer_data);
//     nntrainer::Tensor m = t.transpose("0:1:2");
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     __fp16 answer_data[] = {
//       0,   5,   10,  15,  1,   6,   11,  16,  2,   7,   12,  17,  3,   8,
//       13,  18,  4,   9,   14,  19,  20,  25,  30,  35,  21,  26,  31,  36,
//       22,  27,  32,  37,  23,  28,  33,  38,  24,  29,  34,  39,  40,  45,
//       50,  55,  41,  46,  51,  56,  42,  47,  52,  57,  43,  48,  53,  58,
//       44,  49,  54,  59,  60,  65,  70,  75,  61,  66,  71,  76,  62,  67,
//       72,  77,  63,  68,  73,  78,  64,  69,  74,  79,  80,  85,  90,  95,
//       81,  86,  91,  96,  82,  87,  92,  97,  83,  88,  93,  98,  84,  89,
//       94,  99,  100, 105, 110, 115, 101, 106, 111, 116, 102, 107, 112, 117,
//       103, 108, 113, 118, 104, 109, 114, 119};
//     nntrainer::Tensor answer({3, 2, 5, 4}, answer_data);
//     nntrainer::Tensor m = t.transpose("0:2:1");
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   20,  21,  22,  23,  24,  5,   6,   7,   8,
//       9,   25,  26,  27,  28,  29,  10,  11,  12,  13,  14,  30,  31,  32,
//       33,  34,  15,  16,  17,  18,  19,  35,  36,  37,  38,  39,  40,  41,
//       42,  43,  44,  60,  61,  62,  63,  64,  45,  46,  47,  48,  49,  65,
//       66,  67,  68,  69,  50,  51,  52,  53,  54,  70,  71,  72,  73,  74,
//       55,  56,  57,  58,  59,  75,  76,  77,  78,  79,  80,  81,  82,  83,
//       84,  100, 101, 102, 103, 104, 85,  86,  87,  88,  89,  105, 106, 107,
//       108, 109, 90,  91,  92,  93,  94,  110, 111, 112, 113, 114, 95,  96,
//       97,  98,  99,  115, 116, 117, 118, 119};
//     nntrainer::Tensor answer({3, 4, 2, 5}, answer_data);
//     nntrainer::Tensor m = t.transpose("1:0:2");
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     __fp16 answer_data[] = {
//       0,  20,  1,  21,  2,  22,  3,  23,  4,  24,  5,  25,  6,  26,  7,  27,
//       8,  28,  9,  29,  10, 30,  11, 31,  12, 32,  13, 33,  14, 34,  15, 35,
//       16, 36,  17, 37,  18, 38,  19, 39,  40, 60,  41, 61,  42, 62,  43, 63,
//       44, 64,  45, 65,  46, 66,  47, 67,  48, 68,  49, 69,  50, 70,  51, 71,
//       52, 72,  53, 73,  54, 74,  55, 75,  56, 76,  57, 77,  58, 78,  59, 79,
//       80, 100, 81, 101, 82, 102, 83, 103, 84, 104, 85, 105, 86, 106, 87, 107,
//       88, 108, 89, 109, 90, 110, 91, 111, 92, 112, 93, 113, 94, 114, 95, 115,
//       96, 116, 97, 117, 98, 118, 99, 119};
//     nntrainer::Tensor answer({3, 4, 5, 2}, answer_data);
//     nntrainer::Tensor m = t.transpose("1:2:0");
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     __fp16 answer_data[] = {
//       0,  5,  10,  15,  20,  25,  30,  35, 1,  6,   11,  16,  21,  26,  31,
//       36, 2,  7,   12,  17,  22,  27,  32, 37, 3,   8,   13,  18,  23,  28,
//       33, 38, 4,   9,   14,  19,  24,  29, 34, 39,  40,  45,  50,  55,  60,
//       65, 70, 75,  41,  46,  51,  56,  61, 66, 71,  76,  42,  47,  52,  57,
//       62, 67, 72,  77,  43,  48,  53,  58, 63, 68,  73,  78,  44,  49,  54,
//       59, 64, 69,  74,  79,  80,  85,  90, 95, 100, 105, 110, 115, 81,  86,
//       91, 96, 101, 106, 111, 116, 82,  87, 92, 97,  102, 107, 112, 117, 83,
//       88, 93, 98,  103, 108, 113, 118, 84, 89, 94,  99,  104, 109, 114, 119};
//     nntrainer::Tensor answer({3, 5, 2, 4}, answer_data);
//     nntrainer::Tensor m = t.transpose("2:0:1");
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     __fp16 answer_data[] = {
//       0,  20,  5,  25,  10, 30,  15, 35,  1,  21,  6,  26,  11, 31,  16, 36,
//       2,  22,  7,  27,  12, 32,  17, 37,  3,  23,  8,  28,  13, 33,  18, 38,
//       4,  24,  9,  29,  14, 34,  19, 39,  40, 60,  45, 65,  50, 70,  55, 75,
//       41, 61,  46, 66,  51, 71,  56, 76,  42, 62,  47, 67,  52, 72,  57, 77,
//       43, 63,  48, 68,  53, 73,  58, 78,  44, 64,  49, 69,  54, 74,  59, 79,
//       80, 100, 85, 105, 90, 110, 95, 115, 81, 101, 86, 106, 91, 111, 96, 116,
//       82, 102, 87, 107, 92, 112, 97, 117, 83, 103, 88, 108, 93, 113, 98, 118,
//       84, 104, 89, 109, 94, 114, 99, 119};
//     nntrainer::Tensor answer({3, 5, 4, 2}, answer_data);
//     nntrainer::Tensor m = t.transpose("2:1:0");
//     EXPECT_EQ(answer, m);
//   }

//   /// outplace transpose
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 2, 4, 5);
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
//       14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
//       28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
//       42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
//       56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
//       70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
//       84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
//       98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
//       112, 113, 114, 115, 116, 117, 118, 119};
//     nntrainer::Tensor answer({3, 2, 4, 5}, answer_data);
//     t.transpose("0:1:2", m);
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 2, 5, 4);
//     __fp16 answer_data[] = {
//       0,   5,   10,  15,  1,   6,   11,  16,  2,   7,   12,  17,  3,   8,
//       13,  18,  4,   9,   14,  19,  20,  25,  30,  35,  21,  26,  31,  36,
//       22,  27,  32,  37,  23,  28,  33,  38,  24,  29,  34,  39,  40,  45,
//       50,  55,  41,  46,  51,  56,  42,  47,  52,  57,  43,  48,  53,  58,
//       44,  49,  54,  59,  60,  65,  70,  75,  61,  66,  71,  76,  62,  67,
//       72,  77,  63,  68,  73,  78,  64,  69,  74,  79,  80,  85,  90,  95,
//       81,  86,  91,  96,  82,  87,  92,  97,  83,  88,  93,  98,  84,  89,
//       94,  99,  100, 105, 110, 115, 101, 106, 111, 116, 102, 107, 112, 117,
//       103, 108, 113, 118, 104, 109, 114, 119};
//     nntrainer::Tensor answer({3, 2, 5, 4}, answer_data);
//     t.transpose("0:2:1", m);
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 4, 2, 5);
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   20,  21,  22,  23,  24,  5,   6,   7,   8,
//       9,   25,  26,  27,  28,  29,  10,  11,  12,  13,  14,  30,  31,  32,
//       33,  34,  15,  16,  17,  18,  19,  35,  36,  37,  38,  39,  40,  41,
//       42,  43,  44,  60,  61,  62,  63,  64,  45,  46,  47,  48,  49,  65,
//       66,  67,  68,  69,  50,  51,  52,  53,  54,  70,  71,  72,  73,  74,
//       55,  56,  57,  58,  59,  75,  76,  77,  78,  79,  80,  81,  82,  83,
//       84,  100, 101, 102, 103, 104, 85,  86,  87,  88,  89,  105, 106, 107,
//       108, 109, 90,  91,  92,  93,  94,  110, 111, 112, 113, 114, 95,  96,
//       97,  98,  99,  115, 116, 117, 118, 119};
//     nntrainer::Tensor answer({3, 4, 2, 5}, answer_data);
//     t.transpose("1:0:2", m);
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 4, 5, 2);
//     __fp16 answer_data[] = {
//       0,  20,  1,  21,  2,  22,  3,  23,  4,  24,  5,  25,  6,  26,  7,  27,
//       8,  28,  9,  29,  10, 30,  11, 31,  12, 32,  13, 33,  14, 34,  15, 35,
//       16, 36,  17, 37,  18, 38,  19, 39,  40, 60,  41, 61,  42, 62,  43, 63,
//       44, 64,  45, 65,  46, 66,  47, 67,  48, 68,  49, 69,  50, 70,  51, 71,
//       52, 72,  53, 73,  54, 74,  55, 75,  56, 76,  57, 77,  58, 78,  59, 79,
//       80, 100, 81, 101, 82, 102, 83, 103, 84, 104, 85, 105, 86, 106, 87, 107,
//       88, 108, 89, 109, 90, 110, 91, 111, 92, 112, 93, 113, 94, 114, 95, 115,
//       96, 116, 97, 117, 98, 118, 99, 119};
//     nntrainer::Tensor answer({3, 4, 5, 2}, answer_data);
//     t.transpose("1:2:0", m);
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 5, 2, 4);
//     __fp16 answer_data[] = {
//       0,  5,  10,  15,  20,  25,  30,  35, 1,  6,   11,  16,  21,  26,  31,
//       36, 2,  7,   12,  17,  22,  27,  32, 37, 3,   8,   13,  18,  23,  28,
//       33, 38, 4,   9,   14,  19,  24,  29, 34, 39,  40,  45,  50,  55,  60,
//       65, 70, 75,  41,  46,  51,  56,  61, 66, 71,  76,  42,  47,  52,  57,
//       62, 67, 72,  77,  43,  48,  53,  58, 63, 68,  73,  78,  44,  49,  54,
//       59, 64, 69,  74,  79,  80,  85,  90, 95, 100, 105, 110, 115, 81,  86,
//       91, 96, 101, 106, 111, 116, 82,  87, 92, 97,  102, 107, 112, 117, 83,
//       88, 93, 98,  103, 108, 113, 118, 84, 89, 94,  99,  104, 109, 114, 119};
//     nntrainer::Tensor answer({3, 5, 2, 4}, answer_data);
//     t.transpose("2:0:1", m);
//     EXPECT_EQ(answer, m);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     nntrainer::Tensor m = ranged(3, 5, 4, 2);
//     __fp16 answer_data[] = {
//       0,  20,  5,  25,  10, 30,  15, 35,  1,  21,  6,  26,  11, 31,  16, 36,
//       2,  22,  7,  27,  12, 32,  17, 37,  3,  23,  8,  28,  13, 33,  18, 38,
//       4,  24,  9,  29,  14, 34,  19, 39,  40, 60,  45, 65,  50, 70,  55, 75,
//       41, 61,  46, 66,  51, 71,  56, 76,  42, 62,  47, 67,  52, 72,  57, 77,
//       43, 63,  48, 68,  53, 73,  58, 78,  44, 64,  49, 69,  54, 74,  59, 79,
//       80, 100, 85, 105, 90, 110, 95, 115, 81, 101, 86, 106, 91, 111, 96, 116,
//       82, 102, 87, 107, 92, 112, 97, 117, 83, 103, 88, 108, 93, 113, 98, 118,
//       84, 104, 89, 109, 94, 114, 99, 119};
//     nntrainer::Tensor answer({3, 5, 4, 2}, answer_data);
//     t.transpose("2:1:0", m);
//     EXPECT_EQ(answer, m);
//   }
// }

// TEST(nntrainer_Tensor, tranpose_dimension_not_match_n) {
//   nntrainer::Tensor a(3, 2, 4, 5);
//   nntrainer::Tensor b(3, 1, 2, 3);

//   EXPECT_THROW(a.transpose("0:1:2", b), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, set_01_p) {
//   nntrainer::Tensor tensor = nntrainer::Tensor(1, 1, 1, 1);

//   tensor.setZero();
//   EXPECT_EQ(tensor.getValue(0, 0, 0, 0), 0.0);

//   tensor.setRandUniform(-0.5, 0);
//   __fp16 val = tensor.getValue(0, 0, 0, 0);
//   EXPECT_TRUE(val >= -0.5 && val < 0);
// }

// TEST(nntrainer_Tensor, save_read_01_p) {
//   int batch = 3;
//   int channel = 4;
//   int height = 5;
//   int width = 6;
//   nntrainer::Tensor target(3, 4, 5, 6);
//   nntrainer::Tensor readed(3, 4, 5, 6);

//   GEN_TEST_INPUT(target, i * (channel * width * height) + j * (height * width) +
//                            k * (width) + l + 1);

//   std::ofstream save_file("save.bin", std::ios::out | std::ios::binary);
//   target.save(save_file);
//   save_file.close();

//   std::ifstream read_file("save.bin");
//   readed.read(read_file);
//   read_file.close();

//   EXPECT_EQ(target, readed);

//   int status = std::remove("save.bin");

//   ASSERT_EQ(status, 0);
// }

// TEST(nntrainer_Tensor, save_read_01_n) {
//   int batch = 3;
//   int channel = 4;
//   int height = 5;
//   int width = 6;
//   nntrainer::Tensor target(3, 4, 5, 6);
//   nntrainer::Tensor readed(3, 4, 1, 1);

//   GEN_TEST_INPUT(target, i * (channel * width * height) + j * (height * width) +
//                            k * (width) + l + 1);

//   std::ofstream save_file("save.bin", std::ios::out | std::ios::binary);
//   target.save(save_file);
//   save_file.close();

//   std::ifstream read_file("save.bin");
//   readed.read(read_file);
//   read_file.close();

//   EXPECT_NE(target, readed);

//   int status = std::remove("save.bin");

//   ASSERT_EQ(status, 0);
// }

// TEST(nntrainer_Tensor, copy_and_shares_variable_p) {
//   nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6);
//   nntrainer::Tensor B = A.clone();
//   nntrainer::Tensor C = A;

//   C.setValue(1, 1, 1, 1, 2.0f);

//   EXPECT_EQ(A, C);
//   EXPECT_NE(B, C);

//   C.reshape(nntrainer::TensorDim(3, 4, 6, 5));
//   EXPECT_EQ(A.getDim(), B.getDim());
//   EXPECT_NE(A.getDim(), C.getDim());
// }

// TEST(nntrainer_Tensor, reshape_n_01) {
//   nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6);

//   EXPECT_THROW(A.reshape(nntrainer::TensorDim(9, 9, 9, 9)),
//                std::invalid_argument);
// }

// TEST(nntrainer_Tensor, reshape_n_02) {
//   nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6);
//   nntrainer::TensorDim A_dim = A.getDim();

//   /** Changing the dim of a tensor only affects local copy of the dim */
//   A_dim.setTensorDim(1, 100);
//   EXPECT_EQ(A_dim.getTensorDim(1), 100u);

//   nntrainer::TensorDim A_dim_2 = A.getDim();
//   EXPECT_EQ(A_dim_2.getTensorDim(1), 4u);
// }

// TEST(nntrainer_Tensor, copy_and_reshape_n) {
//   nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6);
//   nntrainer::Tensor B = A;
//   nntrainer::Tensor C = A.clone();

//   EXPECT_THROW(B.reshape(nntrainer::TensorDim(9, 9, 9, 9)),
//                std::invalid_argument);
// }

// /// @note this test case demonstrates it is dangerous to use sharedConstTensor
// /// to const correct the inner data.
// TEST(nntrainer_Tensor, constructor_from_shared_const_ptr_shares_variable_n) {
//   nntrainer::sharedConstTensor A =
//     MAKE_SHARED_TENSOR(constant(1.0f, 3, 4, 5, 6));

//   nntrainer::Tensor B = *A;
//   nntrainer::Tensor C = A->clone();

//   B.setValue(2, 3, 4, 5, 2.0f);
//   EXPECT_EQ(*A, B);
//   EXPECT_NE(*A, C);

//   C.reshape(nntrainer::TensorDim(3, 4, 6, 5));
//   EXPECT_EQ(A->getDim(), B.getDim());
//   EXPECT_NE(A->getDim(), C.getDim());
// }

// TEST(nntrainer_Tensor, print_small_size) {
//   nntrainer::Tensor target = constant(1.0, 3, 1, 2, 3);

//   std::stringstream ss, expected;
//   ss << target;

//   expected << '<' << typeid(target).name() << " at " << &target << ">\n"
//            << "data addr: " << target.getData() << '\n'
//            << "Shape: 3:1:2:3\n"
//            << "         1          1          1 \n"
//            << "         1          1          1 \n"
//            << "\n"
//            << "-------\n"
//            << "         1          1          1 \n"
//            << "         1          1          1 \n"
//            << "\n"
//            << "-------\n"
//            << "         1          1          1 \n"
//            << "         1          1          1 \n"
//            << "\n"
//            << "-------\n";

//   EXPECT_EQ(ss.str(), expected.str());
// }

// // TEST(nntrainer_Tensor, print_large_size) {
// //   nntrainer::Tensor target = constant(1.2, 3, 10, 10, 10);

// //   std::stringstream ss, expected;

// //   expected << '<' << typeid(target).name() << " at " << &target << ">\n"
// //            << "data addr: " << target.getData() << '\n'
// //            << "Shape: 3:10:10:10\n"
// //            << "[1.2 1.2 1.2 ... 1.2 1.2 1.2]\n";
// //   ss << target;

// //   EXPECT_EQ(ss.str(), expected.str());
// // }

// TEST(nntrainer_Tensor, DISABLED_equation_test_01_p) {
//   nntrainer::Tensor a, b, c;
//   nntrainer::Tensor ret1, ret2;

//   a = randUniform(4, 6, 7, 3, -100, 100);
//   b = randUniform(4, 6, 7, 3, -100, 100);
//   c = randUniform(4, 6, 7, 3, -100, 100);

//   ret1 = a.subtract(b).multiply(c);
//   ret2 = a.multiply(c).subtract(b.multiply(c));

//   __fp16 *data1 = ret1.getData();
//   __fp16 *data2 = ret2.getData();
//   EXPECT_EQ(ret1, ret2);

//   for (unsigned int i = 0; i < ret1.size(); ++i) {
//     EXPECT_FLOAT_EQ(data1[i], data2[i]);
//   }
// }

// TEST(nntrainer_Tensor, fill_p) {
//   /// same dimension, buffer size
//   {
//     nntrainer::Tensor target(3, 2, 4, 5);
//     nntrainer::Tensor original = randUniform(3, 2, 4, 5, -1.0f, 1.0f);
//     target.fill(original, false);

//     EXPECT_EQ(target, original);
//   }

//   /// same dimension, buffer size is different (not tested)
//   {
//     /// there is no way to make non contiguous tensor publicily yet
//     EXPECT_TRUE(true);
//   }

//   /// uninitialized with initialized flag is true
//   {
//     nntrainer::Tensor target;
//     nntrainer::Tensor original = randUniform(3, 2, 4, 5, -1.0f, 1.0f);
//     target.fill(original, true);

//     EXPECT_EQ(target, original);
//   }
// }

// TEST(nntrainer_Tensor, fill_uninitialized_n) {
//   nntrainer::Tensor target;
//   nntrainer::Tensor original = randUniform(3, 1, 2, 3, -1.0f, 1.0f);
//   EXPECT_THROW(target.fill(original, false), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, fill_different_dimension_n) {
//   nntrainer::Tensor target(3, 1, 3, 2);
//   nntrainer::Tensor original = randUniform(3, 1, 2, 3, -1.0f, 1.0f);
//   EXPECT_THROW(target.fill(original, false), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, DISABLED_fill_non_contiguous_n) {
//   /// there is no way to make non contiguous tensor publicily yet
//   EXPECT_TRUE(false);
// }

// TEST(nntrainer_Tensor, DISABLED_fill_different_buffer_size_n) {
//   /// there is no way to make same dimension, diffrent buffersized tensor
//   /// publicily yet
//   EXPECT_TRUE(false);
// }

// TEST(nntrainer_Tensor, empty_01) {
//   nntrainer::Tensor t;

//   EXPECT_TRUE(t.empty());
// }

// TEST(nntrainer_Tensor, empty_02) {
//   nntrainer::Tensor t({1, 2, 3, 4}, false);

//   EXPECT_FALSE(t.empty());
// }

// TEST(nntrainer_Tensor, empty_03) {
//   nntrainer::Tensor t({1, 2, 3, 4}, true);

//   EXPECT_FALSE(t.empty());
// }

// TEST(nntrainer_Tensor, allocate_01_n) {
//   nntrainer::Tensor t;
//   EXPECT_FALSE(t.isAllocated());

//   t.allocate();
//   EXPECT_FALSE(t.isAllocated());
// }

// TEST(nntrainer_Tensor, allocate_02_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, false);
//   EXPECT_FALSE(t.isAllocated());

//   t.allocate();
//   EXPECT_TRUE(t.isAllocated());
// }

// TEST(nntrainer_Tensor, allocate_03_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, true);
//   EXPECT_TRUE(t.isAllocated());

//   t.allocate();
//   EXPECT_TRUE(t.isAllocated());
// }

// TEST(nntrainer_Tensor, initialize_01_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, true, nntrainer::Tensor::Initializer::ONES);

//   nntrainer::Tensor golden(1, 2, 3, 4);
//   golden.setValue(1);

//   EXPECT_EQ(golden, t);
// }

// TEST(nntrainer_Tensor, initialize_02_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, true);

//   nntrainer::Tensor golden(1, 2, 3, 4);
//   golden.setValue(1);

//   EXPECT_NE(golden, t);

//   t.initialize(nntrainer::Tensor::Initializer::ONES);
//   EXPECT_EQ(golden, t);
// }

// TEST(nntrainer_Tensor, initialize_03_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, false,
//                       nntrainer::Tensor::Initializer::ONES);
//   t.allocate();

//   nntrainer::Tensor golden(1, 2, 3, 4);
//   golden.setValue(1);

//   EXPECT_EQ(golden, t);
// }

// TEST(nntrainer_Tensor, initialize_04_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, false);
//   t.initialize(nntrainer::Tensor::Initializer::ONES);
//   t.allocate();

//   nntrainer::Tensor golden(1, 2, 3, 4);
//   golden.setValue(1);

//   EXPECT_EQ(golden, t);
// }

// TEST(nntrainer_Tensor, initialize_05_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, false);
//   t.allocate();

//   nntrainer::Tensor golden(1, 2, 3, 4);
//   golden.setValue(1.f);

//   /**
//    * Ideally, it should be NE, but it can be equal due to no initialization
//    * EXPECT_NE(golden, t);
//    */

//   t.initialize(nntrainer::Tensor::Initializer::ONES);
//   EXPECT_EQ(golden, t);
// }

// TEST(nntrainer_Tensor, initialize_06_n) {
//   nntrainer::Tensor t({1, 2, 3, 4}, true, nntrainer::Tensor::Initializer::ONES);
//   nntrainer::Tensor golden({1, 2, 3, 4}, true,
//                            nntrainer::Tensor::Initializer::ZEROS);

//   EXPECT_NE(golden, t);

//   golden.initialize(nntrainer::Tensor::Initializer::ONES);
//   EXPECT_EQ(golden, t);
// }

// TEST(nntrainer_Tensor, initialize_07_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, true, nntrainer::Tensor::Initializer::ONES);

//   nntrainer::Tensor golden(1, 2, 3, 4);
//   golden.setValue(1);

//   EXPECT_EQ(golden, t);

//   t.setValue(0, 0, 0, 0, 0);
//   t.setValue(0, 0, 0, t.size() - 1, 0);
//   EXPECT_NE(golden, t);

//   t.initialize();
//   EXPECT_EQ(golden, t);
// }

// TEST(nntrainer_Tensor, initialize_08_p) {
//   nntrainer::Tensor t({1, 2, 3, 4}, true, nntrainer::Tensor::Initializer::ONES);

//   nntrainer::Tensor golden(1, 2, 3, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
//   golden.setValue(1);
//   EXPECT_EQ(golden, t);

//   t.initialize(nntrainer::Tensor::Initializer::HE_NORMAL);
//   EXPECT_NE(golden, t);


//   t.initialize();
//   EXPECT_NE(golden, t);

//   t.initialize(nntrainer::Tensor::Initializer::ONES);
//   EXPECT_EQ(golden, t);

//   t.initialize();
//   EXPECT_EQ(golden, t);
// }

// TEST(nntrainer_Tensor, split_01_p) {
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(3);
//     {
//       __fp16 answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
//                              10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
//                              20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
//                              30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
//       answer.emplace_back(ml::train::TensorDim{1, 2, 4, 5}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
//                              50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
//                              60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
//                              70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
//       answer.emplace_back(ml::train::TensorDim{1, 2, 4, 5}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
//                              90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
//                              100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
//                              110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
//       answer.emplace_back(ml::train::TensorDim{1, 2, 4, 5}, answer_data);
//     }
//     EXPECT_EQ(t.split(3, 0), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(2);
//     {
//       __fp16 answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
//                              12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43,
//                              44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
//                              56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87,
//                              88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
//       answer.emplace_back(ml::train::TensorDim{3, 1, 4, 5}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
//                              30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
//                              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
//                              70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
//                              100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
//                              110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
//       answer.emplace_back(ml::train::TensorDim{3, 1, 4, 5}, answer_data);
//     }
//     EXPECT_EQ(t.split(2, 1), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(2);
//     {
//       __fp16 answer_data[] = {
//         0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   20,  21,  22,  23,  24,
//         25, 26, 27, 28, 29, 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
//         60, 61, 62, 63, 64, 65,  66,  67,  68,  69,  80,  81,  82,  83,  84,
//         85, 86, 87, 88, 89, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 2, 5}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {
//         10, 11, 12, 13, 14, 15,  16,  17,  18,  19,  30,  31,  32,  33,  34,
//         35, 36, 37, 38, 39, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
//         70, 71, 72, 73, 74, 75,  76,  77,  78,  79,  90,  91,  92,  93,  94,
//         95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 2, 5}, answer_data);
//     }
//     EXPECT_EQ(t.split(2, 2), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(5);
//     {
//       __fp16 answer_data[] = {0,  5,  10, 15, 20,  25,  30,  35,
//                              40, 45, 50, 55, 60,  65,  70,  75,
//                              80, 85, 90, 95, 100, 105, 110, 115};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 1}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {1,  6,  11, 16, 21,  26,  31,  36,
//                              41, 46, 51, 56, 61,  66,  71,  76,
//                              81, 86, 91, 96, 101, 106, 111, 116};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 1}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {2,  7,  12, 17, 22,  27,  32,  37,
//                              42, 47, 52, 57, 62,  67,  72,  77,
//                              82, 87, 92, 97, 102, 107, 112, 117};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 1}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {3,  8,  13, 18, 23,  28,  33,  38,
//                              43, 48, 53, 58, 63,  68,  73,  78,
//                              83, 88, 93, 98, 103, 108, 113, 118};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 1}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
//                              44, 49, 54, 59, 64,  69,  74,  79,
//                              84, 89, 94, 99, 104, 109, 114, 119};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 1}, answer_data);
//     }
//     EXPECT_EQ(t.split(5, 3), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(1, 1, 4, 6);
//     nntrainer::Tensor t = ranged(1, 1, 4, 6);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(2);
//     {
//       __fp16 answer_data[] = {0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20};
//       answer.emplace_back(ml::train::TensorDim{1, 1, 4, 3}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23};
//       answer.emplace_back(ml::train::TensorDim{1, 1, 4, 3}, answer_data);
//     }
//     EXPECT_EQ(t.split(2, 3), answer);
//   }
// }

// TEST(nntrainer_Tensor, split_02_n) {
//   nntrainer::Tensor t(1, 1, 1, 1);
//   EXPECT_THROW(t.split(0, 0), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, split_03_n) {
//   nntrainer::Tensor t(3, 1, 1, 1);
//   EXPECT_THROW(t.split(2, 0), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, split_04_p) {
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(2);
//     {
//       __fp16 answer_data[] = {
//         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
//         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
//         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
//         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
//         64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
//       answer.emplace_back(ml::train::TensorDim{2, 2, 4, 5}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
//                              90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
//                              100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
//                              110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
//       answer.emplace_back(ml::train::TensorDim{1, 2, 4, 5}, answer_data);
//     }
//     EXPECT_EQ(t.split({2, 1}, 0), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(2);
//     {
//       __fp16 answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
//                              12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43,
//                              44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
//                              56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87,
//                              88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
//       answer.emplace_back(ml::train::TensorDim{3, 1, 4, 5}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
//                              30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
//                              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
//                              70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
//                              100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
//                              110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
//       answer.emplace_back(ml::train::TensorDim{3, 1, 4, 5}, answer_data);
//     }
//     EXPECT_EQ(t.split({1, 1}, 1), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(2);
//     {
//       __fp16 answer_data[] = {
//         0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   20,  21,  22,  23,  24,
//         25, 26, 27, 28, 29, 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
//         60, 61, 62, 63, 64, 65,  66,  67,  68,  69,  80,  81,  82,  83,  84,
//         85, 86, 87, 88, 89, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 2, 5}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {
//         10, 11, 12, 13, 14, 15,  16,  17,  18,  19,  30,  31,  32,  33,  34,
//         35, 36, 37, 38, 39, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
//         70, 71, 72, 73, 74, 75,  76,  77,  78,  79,  90,  91,  92,  93,  94,
//         95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 2, 5}, answer_data);
//     }
//     EXPECT_EQ(t.split({2, 2}, 2), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(3);
//     {
//       __fp16 answer_data[] = {0,  5,  10, 15, 20,  25,  30,  35,
//                              40, 45, 50, 55, 60,  65,  70,  75,
//                              80, 85, 90, 95, 100, 105, 110, 115};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 1}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {
//         1,   2,   3,   6,   7,   8,   11,  12,  13,  16,  17,  18, 21, 22, 23,
//         26,  27,  28,  31,  32,  33,  36,  37,  38,  41,  42,  43, 46, 47, 48,
//         51,  52,  53,  56,  57,  58,  61,  62,  63,  66,  67,  68, 71, 72, 73,
//         76,  77,  78,  81,  82,  83,  86,  87,  88,  91,  92,  93, 96, 97, 98,
//         101, 102, 103, 106, 107, 108, 111, 112, 113, 116, 117, 118};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 3}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
//                              44, 49, 54, 59, 64,  69,  74,  79,
//                              84, 89, 94, 99, 104, 109, 114, 119};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 1}, answer_data);
//     }
//     EXPECT_EQ(t.split({1, 3, 1}, 3), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(3);
//     {
//       __fp16 answer_data[] = {
//         0,  1,  5,  6,  10, 11, 15, 16, 20,  21,  25,  26,  30,  31,  35,  36,
//         40, 41, 45, 46, 50, 51, 55, 56, 60,  61,  65,  66,  70,  71,  75,  76,
//         80, 81, 85, 86, 90, 91, 95, 96, 100, 101, 105, 106, 110, 111, 115, 116};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 2}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {
//         2,  3,  7,  8,  12, 13, 17, 18, 22,  23,  27,  28,  32,  33,  37,  38,
//         42, 43, 47, 48, 52, 53, 57, 58, 62,  63,  67,  68,  72,  73,  77,  78,
//         82, 83, 87, 88, 92, 93, 97, 98, 102, 103, 107, 108, 112, 113, 117, 118};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 2}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
//                              44, 49, 54, 59, 64,  69,  74,  79,
//                              84, 89, 94, 99, 104, 109, 114, 119};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 1}, answer_data);
//     }
//     EXPECT_EQ(t.split({2, 2, 1}, 3), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(3, 2, 4, 5);
//     nntrainer::Tensor t = ranged(3, 2, 4, 5);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(2);
//     {
//       __fp16 answer_data[] = {
//         0,  1,  5,  6,  10, 11, 15, 16, 20,  21,  25,  26,  30,  31,  35,  36,
//         40, 41, 45, 46, 50, 51, 55, 56, 60,  61,  65,  66,  70,  71,  75,  76,
//         80, 81, 85, 86, 90, 91, 95, 96, 100, 101, 105, 106, 110, 111, 115, 116};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 2}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {
//         2,   3,   4,   7,   8,   9,   12,  13,  14,  17,  18,  19, 22, 23, 24,
//         27,  28,  29,  32,  33,  34,  37,  38,  39,  42,  43,  44, 47, 48, 49,
//         52,  53,  54,  57,  58,  59,  62,  63,  64,  67,  68,  69, 72, 73, 74,
//         77,  78,  79,  82,  83,  84,  87,  88,  89,  92,  93,  94, 97, 98, 99,
//         102, 103, 104, 107, 108, 109, 112, 113, 114, 117, 118, 119};
//       answer.emplace_back(ml::train::TensorDim{3, 2, 4, 3}, answer_data);
//     }
//     EXPECT_EQ(t.split({2, 3}, 3), answer);
//   }
//   {
//     nntrainer::TensorDim ref_dim(1, 1, 4, 6);
//     nntrainer::Tensor t = ranged(1, 1, 4, 6);
//     std::vector<nntrainer::Tensor> answer;
//     answer.reserve(3);
//     {
//       __fp16 answer_data[] = {0, 6, 12, 18};
//       answer.emplace_back(ml::train::TensorDim{1, 1, 4, 1}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21};
//       answer.emplace_back(ml::train::TensorDim{1, 1, 4, 3}, answer_data);
//     }
//     {
//       __fp16 answer_data[] = {4, 5, 10, 11, 16, 17, 22, 23};
//       answer.emplace_back(ml::train::TensorDim{1, 1, 4, 2}, answer_data);
//     }
//     EXPECT_EQ(t.split({1, 3, 2}, 3), answer);
//   }
// }

// TEST(nntrainer_Tensor, split_05_n) {
//   nntrainer::Tensor t(3, 1, 1, 1);
//   EXPECT_THROW(t.split({1, 1}, 0), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, split_06_n) {
//   nntrainer::Tensor t(3, 1, 1, 1);
//   EXPECT_THROW(t.split({2, 0, 1}, 0), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, split_07_n) {
//   nntrainer::Tensor t(3, 1, 1, 1);
//   EXPECT_THROW(t.split({}, 0), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, cat_01_p) {
//   {
//     std::vector<nntrainer::Tensor> inputs;
//     inputs.reserve(2);
//     inputs.emplace_back(ranged(2, 1, 1, 2));
//     inputs.emplace_back(ranged(2, 2, 1, 2));
//     __fp16 answer_data[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 6, 7};
//     nntrainer::Tensor answer(ml::train::TensorDim{2, 3, 1, 2}, answer_data);
//     EXPECT_EQ(nntrainer::Tensor::cat(inputs, 1), answer);
//   }
//   {
//     std::vector<nntrainer::Tensor> inputs;
//     inputs.reserve(2);
//     inputs.emplace_back(ranged(3, 2, 4, 5));
//     inputs.emplace_back(ranged(2, 2, 4, 5));
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
//       15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
//       30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
//       45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
//       60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
//       75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
//       90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
//       105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
//       0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
//       15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
//       30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
//       45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
//       60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
//       75,  76,  77,  78,  79};
//     nntrainer::Tensor answer(ml::train::TensorDim{5, 2, 4, 5}, answer_data);
//     EXPECT_EQ(nntrainer::Tensor::cat(inputs, 0), answer);
//   }
//   {
//     std::vector<nntrainer::Tensor> inputs;
//     inputs.reserve(2);
//     inputs.emplace_back(ranged(3, 3, 4, 5));
//     inputs.emplace_back(ranged(3, 2, 4, 5));
//     __fp16 answer_data[] = {
//       0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
//       14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
//       28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
//       42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
//       56,  57,  58,  59,  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
//       10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
//       24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,
//       38,  39,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
//       72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,
//       86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
//       100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
//       114, 115, 116, 117, 118, 119, 40,  41,  42,  43,  44,  45,  46,  47,
//       48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
//       62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
//       76,  77,  78,  79,  120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
//       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
//       144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
//       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
//       172, 173, 174, 175, 176, 177, 178, 179, 80,  81,  82,  83,  84,  85,
//       86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
//       100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
//       114, 115, 116, 117, 118, 119};
//     nntrainer::Tensor answer(ml::train::TensorDim{3, 5, 4, 5}, answer_data);
//     EXPECT_EQ(nntrainer::Tensor::cat(inputs, 1), answer);
//   }
//   {
//     std::vector<nntrainer::Tensor> inputs;
//     inputs.reserve(2);
//     inputs.emplace_back(ranged(3, 2, 1, 5));
//     inputs.emplace_back(ranged(3, 2, 2, 5));
//     __fp16 answer_data[] = {
//       0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,
//       8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 20,
//       21, 22, 23, 24, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 30, 31, 32, 33,
//       34, 35, 36, 37, 38, 39, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44, 45, 46,
//       47, 48, 49, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
//     nntrainer::Tensor answer(ml::train::TensorDim{3, 2, 3, 5}, answer_data);
//     EXPECT_EQ(nntrainer::Tensor::cat(inputs, 2), answer);
//   }
//   {
//     std::vector<nntrainer::Tensor> inputs;
//     inputs.reserve(3);
//     inputs.emplace_back(ranged(3, 2, 4, 1));
//     inputs.emplace_back(ranged(3, 2, 4, 3));
//     inputs.emplace_back(ranged(3, 2, 4, 2));
//     __fp16 answer_data[] = {
//       0,  0,  1,  2,  0,  1,  1,  3,  4,  5,  2,  3,  2,  6,  7,  8,  4,  5,
//       3,  9,  10, 11, 6,  7,  4,  12, 13, 14, 8,  9,  5,  15, 16, 17, 10, 11,
//       6,  18, 19, 20, 12, 13, 7,  21, 22, 23, 14, 15, 8,  24, 25, 26, 16, 17,
//       9,  27, 28, 29, 18, 19, 10, 30, 31, 32, 20, 21, 11, 33, 34, 35, 22, 23,
//       12, 36, 37, 38, 24, 25, 13, 39, 40, 41, 26, 27, 14, 42, 43, 44, 28, 29,
//       15, 45, 46, 47, 30, 31, 16, 48, 49, 50, 32, 33, 17, 51, 52, 53, 34, 35,
//       18, 54, 55, 56, 36, 37, 19, 57, 58, 59, 38, 39, 20, 60, 61, 62, 40, 41,
//       21, 63, 64, 65, 42, 43, 22, 66, 67, 68, 44, 45, 23, 69, 70, 71, 46, 47};
//     nntrainer::Tensor answer(ml::train::TensorDim{3, 2, 4, 6}, answer_data);
//     EXPECT_EQ(nntrainer::Tensor::cat(inputs, 3), answer);
//   }
// }

// TEST(nntrainer_Tensor, cat_02_n) {
//   {
//     std::vector<nntrainer::Tensor> inputs;
//     inputs.reserve(2);
//     inputs.emplace_back(nntrainer::Tensor(2, 1, 1, 2));
//     inputs.emplace_back(nntrainer::Tensor(2, 2, 1, 2));
//     EXPECT_THROW(nntrainer::Tensor::cat(inputs, 2), std::invalid_argument);
//   }
// }

// TEST(nntrainer_Tensor, zoneout_mask_01_n) {
//   const __fp16 zoneout_rate = 0.3f;
//   nntrainer::Tensor t(10, 10, 10, 10);
//   nntrainer::Tensor opposite(20, 20, 20, 20);
//   EXPECT_THROW(t.zoneout_mask(opposite, zoneout_rate), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, zoneout_mask_02_p) {
//   const __fp16 zoneout_rate = 0.3f;
//   nntrainer::Tensor t(10, 10, 10, 10);
//   nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
//   constexpr __fp16 epsilon = 1e-3;

//   EXPECT_EQ(t.size(), opposite.size());

//   auto is_near = [epsilon](__fp16 val1, __fp16 val2) {
//     return val2 - epsilon < val1 && val1 < val2 + epsilon;
//   };

//   for (unsigned int i = 0; i < opposite.size(); ++i) {
//     if (is_near(opposite.getValue(i), 0.0f)) {
//       EXPECT_NEAR(t.getValue(i), 1.0f, epsilon);
//     } else if (is_near(opposite.getValue(i), 1.0f)) {
//       EXPECT_NEAR(t.getValue(i), 0.0f, epsilon);
//     } else {
//       FAIL() << "This should not be happen";
//     }
//   }
// }

// TEST(nntrainer_Tensor, zoneout_mask_03_p) {
//   const __fp16 zoneout_rate = 0.3f;
//   nntrainer::Tensor t(10, 10, 100, 100);
//   nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
//   constexpr __fp16 epsilon = 1e-3;

//   auto is_near = [epsilon](__fp16 val1, __fp16 val2) {
//     return val2 - epsilon < val1 && val1 < val2 + epsilon;
//   };
//   auto percentage = [](unsigned int dividend, unsigned int divisor) {
//     return (__fp16)dividend / (__fp16)divisor;
//   };

//   {
//     unsigned int zeros = 0;
//     unsigned int ones = 0;
//     for (unsigned int i = 0; i < opposite.size(); ++i) {
//       if (is_near(opposite.getValue(i), 0.0f)) {
//         ++zeros;
//       } else if (is_near(opposite.getValue(i), 1.0f)) {
//         ++ones;
//       } else {
//         FAIL() << "This should not be happen";
//       }
//     }
//     EXPECT_NEAR(percentage(zeros, opposite.size()), 1.0f - zoneout_rate,
//                 epsilon);

//     // main test
//     EXPECT_NEAR(percentage(ones, opposite.size()), zoneout_rate, epsilon);
//   }

//   {
//     unsigned int zeros = 0;
//     unsigned int ones = 0;
//     for (unsigned int i = 0; i < t.size(); ++i) {
//       if (is_near(t.getValue(i), 0.0f)) {
//         ++zeros;
//       } else if (is_near(t.getValue(i), 1.0f)) {
//         ++ones;
//       } else {
//         FAIL() << "This should not be happen";
//       }
//     }
//     EXPECT_NEAR(percentage(zeros, t.size()), zoneout_rate, epsilon);

//     // main test
//     EXPECT_NEAR(percentage(ones, t.size()), 1.0f - zoneout_rate, epsilon);
//   }
// }

// TEST(nntrainer_Tensor, zoneout_mask_04_n) {
//   const __fp16 zoneout_rate = 0.3f;
//   nntrainer::Tensor t(10, 10, 100, 100);
//   nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
//   constexpr __fp16 epsilon = 1e-3;

//   auto is_near = [epsilon](__fp16 val1, __fp16 val2) {
//     return val2 - epsilon < val1 && val1 < val2 + epsilon;
//   };
//   auto percentage = [](unsigned int dividend, unsigned int divisor) {
//     return (__fp16)dividend / (__fp16)divisor;
//   };

//   {
//     unsigned int zeros = 0;
//     unsigned int ones = 0;
//     for (unsigned int i = 0; i < opposite.size(); ++i) {
//       if (is_near(opposite.getValue(i), 0.0f)) {
//         ++zeros;
//       } else if (is_near(opposite.getValue(i), 1.0f)) {
//         ++ones;
//       } else {
//         FAIL() << "This should not be happen";
//       }
//     }
//     EXPECT_FALSE(
//       is_near(percentage(ones, opposite.size()), 1.0f - zoneout_rate));
//   }

//   {
//     unsigned int zeros = 0;
//     unsigned int ones = 0;
//     for (unsigned int i = 0; i < t.size(); ++i) {
//       if (is_near(t.getValue(i), 0.0f)) {
//         ++zeros;
//       } else if (is_near(t.getValue(i), 1.0f)) {
//         ++ones;
//       } else {
//         FAIL() << "This should not be happen";
//       }
//     }
//     EXPECT_FALSE(is_near(percentage(ones, t.size()), zoneout_rate));
//   }
// }

// TEST(nntrainer_Tensor, TensorMap_p) {
//   __fp16 dat[] = {1, 2, 3};

//   {
//     nntrainer::Tensor a = nntrainer::Tensor::Map(dat, 3 * sizeof(__fp16), {3});
//     /// check if a.getData() has same address with dat
//     EXPECT_EQ(dat, a.getData());
//     {
//       /// check if b.getData() has same address with data
//       nntrainer::Tensor b = a;
//       EXPECT_EQ(dat, b.getData());
//     }
//   }
//   /// check if dat is accessible after destruction of all the tensor
//   EXPECT_FLOAT_EQ(dat[2], 3);
// }

// TEST(nntrainer_Tensor, TensorWrap_01_n) {
//   __fp16 dat[] = {1, 2, 3};
//   EXPECT_THROW(nntrainer::Tensor::Map(dat, 3, nntrainer::TensorDim({})),
//                std::invalid_argument);
// }

// TEST(nntrainer_Tensor, TensorWrap_02_n) {
//   __fp16 dat[] = {1, 2, 3};
//   EXPECT_THROW(nntrainer::Tensor::Map(dat, 3, {4}), std::invalid_argument);
// }

// TEST(nntrainer_Tensor, TensorPaddedValue_p) {
//   nntrainer::Tensor a = ranged(1, 1, 3, 3);
//   __fp16 default_padded = -1;

//   for (int i = 0; i < 5; ++i) {
//     for (int j = 0; j < 5; ++j) {
//       __fp16 expected = default_padded;
//       if (1 <= i && i <= 3 && 1 <= j && j <= 3) {
//         expected = (i - 1) * 3 + (j - 1);
//       }
//       __fp16 actual = a.getValuePaddedVirtual<__fp16>(0, 0, i, j, 1, 1, default_padded);
//       EXPECT_FLOAT_EQ(actual, expected);
//     }
//   }
// }

GTEST_API_ int main(int argc, char **argv) {
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
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
