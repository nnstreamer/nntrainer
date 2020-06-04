/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        unittest_util_func.cpp
 * @date        10 April 2020
 * @brief       Unit test for util_func.cpp.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

#define tolerance 10e-5

#define GEN_TEST_INPUT(input, eqation_i_j_k_l) \
  do {                                         \
    for (int i = 0; i < batch; ++i) {          \
      for (int j = 0; j < channel; ++j) {      \
        for (int k = 0; k < height; ++k) {     \
          for (int l = 0; l < width; ++l) {    \
            float val = eqation_i_j_k_l;       \
            input.setValue(i, j, k, l, val);   \
          }                                    \
        }                                      \
      }                                        \
    }                                          \
  } while (0)

/**
 * @brief Data Buffer
 */
TEST(nntrainer_util_func, softmax_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float results[10] = {7.80134161e-05, 2.12062451e-04, 5.76445508e-04,
                       1.56694135e-03, 4.25938820e-03, 1.15782175e-02,
                       3.14728583e-02, 8.55520989e-02, 2.32554716e-01,
                       6.32149258e-01};

  nntrainer::Tensor T(batch, channel, height, width);
  nntrainer::Tensor Results(batch, channel, height, width);

  GEN_TEST_INPUT(T, (i * (width) + l + 1));

  Results = T.apply(nntrainer::softmax);
  float *data = Results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], results[i % width], tolerance);
  }
}

TEST(nntrainer_util_func, softmax_prime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float results[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (i * (width) + k + 1));

  nntrainer::Tensor softmax_result = input.apply(nntrainer::softmax);

  float *data = softmax_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor softmax_prime_result =
    softmax_result.apply(nntrainer::softmaxPrime);
  data = softmax_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], results[i % width], tolerance);
  }
}

TEST(nntrainer_util_func, random_01_p) {
  int status = ML_ERROR_INVALID_PARAMETER;
  srand(time(NULL));
  float x = nntrainer::random(0.0);
  if (-1.0 < x && x < 1.0)
    status = ML_ERROR_NONE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, sqrtFloat_01_p) {
  int status = ML_ERROR_INVALID_PARAMETER;

  float x = 9871.0;
  float sx = nntrainer::sqrtFloat(x);

  if (fabs(sx * sx - x) < tolerance)
    status = ML_ERROR_NONE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, logFloat_01_p) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (width) + k + 1);

  nntrainer::Tensor Results = input.apply(nntrainer::logFloat);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], (float)log(indata[i]), tolerance);
  }
}

TEST(nntrainer_util_func, sigmoid_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {0.4013123, 0.4255575, 0.450166,  0.4750208, 0.5,
                      0.5249792, 0.549834,  0.5744425, 0.5986877, 0.6224593,
                      0.3100255, 0.3543437, 0.4013123, 0.450166,  0.5,
                      0.549834,  0.5986877, 0.6456563, 0.6899745, 0.7310586,
                      0.2314752, 0.2890505, 0.3543437, 0.4255575, 0.5,
                      0.5744425, 0.6456563, 0.7109495, 0.7685248, 0.8175745};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor Results = input.apply(nntrainer::sigmoid);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_util_func, sigmoidPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    2.40198421e-01, 2.39014374e-01, 2.37750225e-01, 2.36411103e-01,
    2.35003712e-01, 2.33536203e-01, 2.32017974e-01, 2.30459419e-01,
    2.28871631e-01, 2.27266092e-01, 2.44087699e-01, 2.42313881e-01,
    2.40198421e-01, 2.37750225e-01, 2.35003712e-01, 2.32017974e-01,
    2.28871631e-01, 2.25654341e-01, 2.22456864e-01, 2.19361864e-01,
    2.46680881e-01, 2.44849977e-01, 2.42313881e-01, 2.39014374e-01,
    2.35003712e-01, 2.30459419e-01, 2.25654341e-01, 2.2089191e-01,
    2.16437141e-01, 2.12472086e-01};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor sigmoid_result = input.apply(nntrainer::sigmoid);
  float *data = sigmoid_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result =
    sigmoid_result.apply(nntrainer::sigmoidePrime);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_util_func, tanhFloat_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    -3.79948962e-01, -2.91312612e-01, -1.9737532e-01,  -9.96679946e-02,
    0e+00,           9.96679946e-02,  1.9737532e-01,   2.91312612e-01,
    3.79948962e-01,  4.62117157e-01,  -6.6403677e-01,  -5.37049567e-01,
    -3.79948962e-01, -1.9737532e-01,  0e+00,           1.9737532e-01,
    3.79948962e-01,  5.37049567e-01,  6.6403677e-01,   7.61594156e-01,
    -8.33654607e-01, -7.1629787e-01,  -5.37049567e-01, -2.91312612e-01,
    0e+00,           2.91312612e-01,  5.37049567e-01,  7.1629787e-01,
    8.33654607e-01,  9.05148254e-01};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor Results = input.apply(nntrainer::tanhFloat);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_util_func, tanhFloatPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0.8684754, 0.919717,  0.9620329, 0.9901317, 1,
                      0.9901317, 0.9620329, 0.919717,  0.8684754, 0.8135417,
                      0.6623883, 0.7591631, 0.8684754, 0.9620329, 1,
                      0.9620329, 0.8684754, 0.7591631, 0.6623883, 0.5878168,
                      0.5342845, 0.6222535, 0.7591631, 0.919717,  1,
                      0.919717,  0.7591631, 0.6222535, 0.5342845, 0.4833332};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor tanh_result = input.apply(nntrainer::tanhFloat);
  float *data = tanh_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result = tanh_result.apply(nntrainer::tanhPrime);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_util_func, relu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5,
                      0, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1,
                      0, 0, 0, 0, 0, 0.3, 0.6, 0.9, 1.2, 1.5};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor Results = input.apply(nntrainer::relu);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_util_func, reluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor relu_result = input.apply(nntrainer::relu);
  float *data = relu_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result = relu_result.apply(nntrainer::reluPrime);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
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
    ml_loge("Failed to init gtest\n");
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    ml_loge("Failed to run test.\n");
  }

  return result;
}
