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
