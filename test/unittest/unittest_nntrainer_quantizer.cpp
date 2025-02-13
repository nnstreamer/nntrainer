// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file        unittest_nntrainer_quantizer.cpp
 * @date        16 December 2024
 * @brief       Unit test utility for quantizer.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <fstream>
#include <nntrainer_error.h>
#include <quantizer.h>
#include <tensor.h>

/**
 * @brief Quantize to FP32 Tensor (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_01_n) {
  nntrainer::Tensor input(3, 2, 4, 5);
  input.setRandNormal(1.235f, 0.04f);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_THROW(quantizer->quantize(input, nntrainer::Tdatatype::FP32),
               std::invalid_argument);
}

/**
 * @brief Dequantize to Quantized Tensor (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_02_n) {
  nntrainer::Tensor input(3, 3, 24, 24);
  input.setRandNormal(3.812f, 0.15f);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  nntrainer::Tensor quantized_tensor =
    quantizer->quantize(input, nntrainer::Tdatatype::QINT8);

  EXPECT_THROW(quantizer->dequantize(input, nntrainer::Tdatatype::QINT8),
               std::invalid_argument);
}

/**
 * @brief Quantize Quantized Tensor (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_03_n) {
  nntrainer::Tensor input(3, 3, 24, 24, nntrainer::Tformat::NCHW,
                          nntrainer::Tdatatype::QINT8);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_THROW(quantizer->quantize(input, nntrainer::Tdatatype::QINT4),
               std::invalid_argument);
}

/**
 * @brief Output tensor size not matching (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_04_n) {
  nntrainer::Tensor input(3, 2, 4, 5);
  input.setRandNormal(1.235f, 0.04f);

  nntrainer::Tensor output(3, 1, 1, 1, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::QINT8);

  float scale = 0.00235f;

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_THROW(quantizer->quantize(input, output, &scale),
               std::invalid_argument);
}

/**
 * @brief Zero point not given for unsigned int (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_05_n) {
  nntrainer::Tensor input(3, 2, 4, 5);
  input.setRandNormal(1.235f, 0.04f);

  nntrainer::Tensor uint8_output(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::UINT8);

  nntrainer::Tensor uint16_output(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                  nntrainer::Tdatatype::UINT16);

  float scale = 0.00235f;

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_THROW(quantizer->quantize(input, uint8_output, &scale),
               std::invalid_argument);

  EXPECT_THROW(quantizer->quantize(input, uint16_output, &scale),
               std::invalid_argument);
}

TEST(nntrainer_Quantizer, per_tensor_affine_01_p) {
  float input_data[] = {-0.16924214, -0.10338581, 0.31561565,  -0.00533330,
                        0.44809300,  -0.15348488, 0.14003623,  -0.07908171,
                        -0.21415669, -0.35267806, 0.46354777,  -0.35009885,
                        -0.07760239, -0.28348053, -0.37242615, 0.30941701};
  nntrainer::Tensor input({1, 1, 4, 4}, input_data);

  std::vector<int8_t> qdata = {-47, -28, 87,  -1,  123, -42, 39,   -22,
                               -59, -97, 127, -96, -21, -78, -102, 85};
  float qscale = 0.00363567f;
  int8_t *scale_array = reinterpret_cast<int8_t *>(&qscale);
  for (unsigned int i = 0; i < 4; ++i) {
    qdata.push_back(scale_array[i]);
  }
  nntrainer::Tensor quant_answer(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8},
    qdata.data());

  float output_data[] = {-0.17087643, -0.10179872, 0.31630316,  -0.00363567,
                         0.44718724,  -0.15269808, 0.14179108,  -0.07998471,
                         -0.21450445, -0.35265985, 0.46172991,  -0.34902418,
                         -0.07634904, -0.28358215, -0.37083820, 0.30903184};
  nntrainer::Tensor float_answer({1, 1, 4, 4}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  nntrainer::Tensor quantized_tensor =
    quantizer->quantize(input, nntrainer::Tdatatype::QINT8);
  ASSERT_EQ(quantized_tensor, quant_answer);

  // Perform Dequantization
  nntrainer::Tensor output =
    quantizer->dequantize(quantized_tensor, nntrainer::Tdatatype::FP32);
  ASSERT_EQ(output, float_answer);
}

TEST(nntrainer_Quantizer, per_tensor_affine_02_p) {
  float input_data[] = {
    -0.29562217, 0.02348283,  0.04334664,  0.03752254,  0.17764580,
    0.04449826,  0.15144463,  -0.15716791, -0.07842141, 0.34517670,
    0.16458672,  -0.09487095, -0.28020513, 0.32698259,  -0.24903688,
    -0.33132783, 0.13940062,  0.18400775,  -0.26359966, 0.30900121,
    0.08309542,  -0.09066082, 0.08950174,  -0.29709017, -0.26397359,
    -0.16240828, -0.18758762, -0.31878781, 0.06728745,  -0.04749811,
    0.16789703,  0.02212419,  0.10671097,  -0.28938687, 0.16250020,
    -0.09017495, 0.24699482,  -0.26789218, 0.16414545,  0.22879964,
    -0.15821624, -0.23149055, 0.26526868,  -0.11006282, -0.20480227,
    0.29863110,  0.24005184,  -0.09062263, 0.22294718,  0.32583672,
    -0.10362835, 0.03243832,  0.24707781,  0.27685603,  0.03360258,
    -0.00209959, 0.27976128,  -0.24468939, -0.19273037, -0.25921509,
    -0.20489319, 0.33036807,  0.27226517,  -0.25207010};
  nntrainer::Tensor input({1, 1, 8, 8}, input_data);

  std::vector<int8_t> qdata = {
    -109, 9,    16,   14,  66,  16,  56,  -58,  -29, 127, 61,   -35, -104,
    121,  -92,  -122, 51,  68,  -97, 114, 31,   -33, 33,  -110, -98, -60,
    -69,  -118, 25,   -18, 62,  8,   39,  -107, 60,  -33, 91,   -99, 61,
    85,   -58,  -86,  98,  -41, -76, 110, 89,   -33, 82,  120,  -38, 12,
    91,   102,  12,   -1,  103, -90, -71, -96,  -76, 122, 101,  -93};
  float qscale = 0.00270727f;
  int8_t *scale_array = reinterpret_cast<int8_t *>(&qscale);
  for (unsigned int i = 0; i < 4; ++i) {
    qdata.push_back(scale_array[i]);
  }
  nntrainer::Tensor quant_answer(
    {1, 1, 8, 8, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8},
    qdata.data());

  float output_data[] = {
    -0.29509223, 0.02436541,  0.04331629,  0.03790175,  0.17867969,
    0.04331629,  0.15160701,  -0.15702155, -0.07851078, 0.34382305,
    0.16514336,  -0.09475438, -0.28155589, 0.32757944,  -0.24906866,
    -0.33028671, 0.13807067,  0.18409424,  -0.26260501, 0.30862856,
    0.08392531,  -0.08933984, 0.08933984,  -0.29779950, -0.26531228,
    -0.16243608, -0.18680149, -0.31945765, 0.06768170,  -0.04873083,
    0.16785063,  0.02165814,  0.10558346,  -0.28967768, 0.16243608,
    -0.08933984, 0.24636140,  -0.26801956, 0.16514336,  0.23011778,
    -0.15702155, -0.23282506, 0.26531228,  -0.11099799, -0.20575237,
    0.29779950,  0.24094686,  -0.08933984, 0.22199598,  0.32487217,
    -0.10287619, 0.03248722,  0.24636140,  0.27614135,  0.03248722,
    -0.00270727, 0.27884862,  -0.24365413, -0.19221604, -0.25989774,
    -0.20575237, 0.33028671,  0.27343407,  -0.25177592};
  nntrainer::Tensor float_answer({1, 1, 8, 8}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  nntrainer::Tensor quantized_tensor =
    quantizer->quantize(input, nntrainer::Tdatatype::QINT8);
  ASSERT_EQ(quantized_tensor, quant_answer);

  // Perform Dequantization
  nntrainer::Tensor output =
    quantizer->dequantize(quantized_tensor, nntrainer::Tdatatype::FP32);
  ASSERT_EQ(output, float_answer);
}

TEST(nntrainer_Quantizer, per_tensor_affine_03_p) {
  float input_data[] = {-0.16924214, -0.10338581, 0.31561565,  -0.00533330,
                        0.44809300,  -0.15348488, 0.14003623,  -0.07908171,
                        -0.21415669, -0.35267806, 0.46354777,  -0.35009885,
                        -0.07760239, -0.28348053, -0.37242615, 0.30941701};
  nntrainer::Tensor input({1, 1, 4, 4}, input_data);

  std::vector<int16_t> qdata = {-47, -28, 87,  -1,  123, -42, 39,   -22,
                                -59, -97, 127, -96, -21, -78, -102, 85};
  float qscale = 0.00363567f;
  int16_t *scale_array = reinterpret_cast<int16_t *>(&qscale);
  for (unsigned int i = 0; i < 2; ++i) {
    qdata.push_back(scale_array[i]);
  }
  nntrainer::Tensor quant_answer(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT16},
    qdata.data());

  nntrainer::Tensor quantized_tensor(1, 1, 4, 4, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::QINT16);

  float output_data[] = {-0.17087643, -0.10179872, 0.31630316,  -0.00363567,
                         0.44718724,  -0.15269808, 0.14179108,  -0.07998471,
                         -0.21450445, -0.35265985, 0.46172991,  -0.34902418,
                         -0.07634904, -0.28358215, -0.37083820, 0.30903184};
  nntrainer::Tensor float_answer({1, 1, 4, 4}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  quantizer->quantize(input, quantized_tensor, &qscale);
  ASSERT_EQ(quantized_tensor, quant_answer);

  // Perform Dequantization
  nntrainer::Tensor output =
    quantizer->dequantize(quantized_tensor, nntrainer::Tdatatype::FP32);
  ASSERT_EQ(output, float_answer);
}

/**
 * @brief Quantize / Dequantize UInt8Tensor
 */
TEST(nntrainer_Quantizer, per_tensor_affine_04_p) {
  float input_data[] = {2.31893802,  4.46305752,  -0.75207627, -2.51219273,
                        -0.59212941, -3.74816823, 0.58360142,  0.86855388,
                        2.07299328,  2.69872355,  -1.41879117, 2.31787777,
                        -0.29471058, -0.72146493, 1.81435537,  -1.59683037};
  nntrainer::Tensor input({1, 1, 4, 4}, input_data);

  std::vector<uint8_t> qdata = {194, 255, 107, 56,  111, 21,  145, 153,
                                187, 205, 87,  194, 120, 107, 180, 82};
  float qscale = 0.03500437;
  unsigned int zero_point = 128;

  uint8_t *scale_array = reinterpret_cast<uint8_t *>(&qscale);
  for (unsigned int i = 0; i < sizeof(float) / sizeof(uint8_t); ++i) {
    qdata.push_back(scale_array[i]);
  }

  uint8_t *zp_array = reinterpret_cast<uint8_t *>(&zero_point);
  for (unsigned int i = 0; i < sizeof(unsigned int) / sizeof(uint8_t); ++i) {
    qdata.push_back(zp_array[i]);
  }

  nntrainer::Tensor quant_answer(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT8},
    qdata.data());

  nntrainer::Tensor quantized_tensor(1, 1, 4, 4, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::UINT8);

  float output_data[] = {2.31028867,  4.44555569,  -0.73509187, -2.52031493,
                         -0.59507436, -3.74546790, 0.59507436,  0.87510931,
                         2.06525803,  2.69533682,  -1.43517935, 2.31028867,
                         -0.28003499, -0.73509187, 1.82022738,  -1.61020124};
  nntrainer::Tensor float_answer({1, 1, 4, 4}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  quantizer->quantize(input, quantized_tensor, &qscale, &zero_point);
  ASSERT_EQ(quantized_tensor, quant_answer);

  // Perform Dequantization
  nntrainer::Tensor output =
    quantizer->dequantize(quantized_tensor, nntrainer::Tdatatype::FP32);
  ASSERT_EQ(output, float_answer);
}

/**
 * @brief Tensor quantization to QINT8 and QUINT8
 *
 * @note This test quantizes a float tensor to int8 and uint8, then compares the
 * dequantized output to check if they are the same.
 */
TEST(nntrainer_Quantizer, per_tensor_affine_05_p) {
  // float input tensor
  float input_data[] = {-0.69094253, 3.77131414,  -4.77607393, 1.86816788,
                        -2.97529221, 3.99959946,  -1.44690418, 2.54158640,
                        -0.79941863, -3.75069141, -1.38934612, -0.23342809,
                        -4.05783129, 1.41701365,  -0.84545374, 2.20419312};
  nntrainer::Tensor input({1, 1, 4, 4}, input_data);

  // quantization params
  float scale = 0.03136941;
  unsigned int zero_point = 128;

  // Construct unsigned 8-int tensor (answer)
  std::vector<uint8_t> data_u8 = {106, 248, 0,  188, 33, 255, 82,  209,
                                  103, 8,   84, 121, 0,  173, 101, 198};

  uint8_t *scales_u8 = reinterpret_cast<uint8_t *>(&scale);
  for (unsigned int i = 0; i < sizeof(float) / sizeof(uint8_t); ++i) {
    data_u8.push_back(scales_u8[i]);
  }

  uint8_t *zps_u8 = reinterpret_cast<uint8_t *>(&zero_point);
  for (unsigned int i = 0; i < sizeof(unsigned int) / sizeof(uint8_t); ++i) {
    data_u8.push_back(zps_u8[i]);
  }

  nntrainer::Tensor q_answer_u8(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT8},
    data_u8.data());

  // Construct signed 8-int tensor (answer)
  std::vector<int8_t> data_s8 = {-22, 120,  -128, 60, -95,  127, -46, 81,
                                 -25, -120, -44,  -7, -128, 45,  -27, 70};

  int8_t *scales_s8 = reinterpret_cast<int8_t *>(&scale);
  for (unsigned int i = 0; i < sizeof(float) / sizeof(int8_t); ++i) {
    data_s8.push_back(scales_s8[i]);
  }

  int8_t *zps_s8 = reinterpret_cast<int8_t *>(&zero_point);
  for (unsigned int i = 0; i < sizeof(unsigned int) / sizeof(int8_t); ++i) {
    data_s8.push_back(zps_s8[i]);
  }

  nntrainer::Tensor q_answer_s8(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8},
    data_s8.data());

  // Output quantized tensors
  nntrainer::Tensor q_tensor_u8(1, 1, 4, 4, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::UINT8);

  nntrainer::Tensor q_tensor_s8(1, 1, 4, 4, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::QINT8);

  // Construct output float tensor (answer)
  float output_data[] = {-0.69012696, 3.76432872,  -4.01528406, 1.88216436,
                         -2.98009372, 3.98391461,  -1.44299269, 2.54092193,
                         -0.78423518, -3.76432872, -1.38025391, -0.21958585,
                         -4.01528406, 1.41162336,  -0.84697396, 2.19585848};
  nntrainer::Tensor float_answer({1, 1, 4, 4}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  quantizer->quantize(input, q_tensor_u8, &scale, &zero_point);
  ASSERT_EQ(q_tensor_u8, q_answer_u8);

  quantizer->quantize(input, q_tensor_s8, &scale, &zero_point);
  ASSERT_EQ(q_tensor_s8, q_answer_s8);

  // Perform Dequantization
  nntrainer::Tensor output_s8 =
    quantizer->dequantize(q_tensor_s8, nntrainer::Tdatatype::FP32);

  nntrainer::Tensor output_u8 =
    quantizer->dequantize(q_tensor_u8, nntrainer::Tdatatype::FP32);

  ASSERT_EQ(output_s8, float_answer);
  ASSERT_EQ(output_u8, float_answer);
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
