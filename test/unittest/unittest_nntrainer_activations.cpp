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
 * @file        unittest_nntrainer_activations.cpp
 * @date        21 July 2020
 * @brief       Unit test for static functions in activation_layer.cpp.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <activation_layer.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <nntrainer_test_util.h>

TEST(nntrainer_activation, softmax_01_p) {
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

  Results = T.apply(nntrainer::ActiFunc::softmax<float>, Results);
  float *data = Results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], results[i % width], tolerance);
  }
}

TEST(nntrainer_activation, softmax_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor T(batch, channel, height, width);
  nntrainer::Tensor Results(batch, channel, height, 2 * width);
  nntrainer::Tensor Result = Results.getSharedDataTensor(
    nntrainer::TensorDim(batch, channel, height, width), 0, false);

  EXPECT_THROW(T.apply(nntrainer::ActiFunc::softmax<float>, Result),
               std::invalid_argument);
}

TEST(nntrainer_activation, softmax_prime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float results[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (i * (width) + k + 1));

  nntrainer::Tensor softmax_result;
  softmax_result =
    input.apply(nntrainer::ActiFunc::softmax<float>, softmax_result);

  float *data = softmax_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor softmax_prime_result;
  softmax_prime_result =
    nntrainer::ActiFunc::softmaxPrime(softmax_result, softmax_prime_result);

  data = softmax_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], results[i % width], tolerance);
  }
}

TEST(nntrainer_activation, softmax_prime_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (i * (width) + k + 1));

  nntrainer::Tensor softmax_result;
  softmax_result =
    input.apply(nntrainer::ActiFunc::softmax<float>, softmax_result);

  float *data = softmax_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor softmax_prime_result(batch, channel, height, 2 * width);
  nntrainer::Tensor softmax_prime_result_shared =
    softmax_prime_result.getSharedDataTensor(
      nntrainer::TensorDim(batch, channel, height, width), 0, false);
  EXPECT_THROW(nntrainer::ActiFunc::softmaxPrime(softmax_result,
                                                 softmax_prime_result_shared),
               std::invalid_argument);
}

TEST(nntrainer_activation, softplus_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.51301527, 0.55435520, 0.59813887, 0.64439666, 0.69314718, 0.74439669,
    0.79813886, 0.85435522, 0.91301525, 0.97407699, 0.37110066, 0.43748793,
    0.51301527, 0.59813887, 0.69314718, 0.79813886, 0.91301525, 1.03748798,
    1.17110062, 1.31326163, 0.26328245, 0.34115386, 0.43748793, 0.55435526,
    0.69314718, 0.85435528, 1.03748798, 1.24115384, 1.46328247, 1.70141327};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor softplus_result =
    input.apply<float>(nntrainer::ActiFunc::softplus<float>);

  float *data = softplus_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, softplusPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.40131232, 0.42555746, 0.45016599, 0.47502083, 0.50000000, 0.52497917,
    0.54983401, 0.57444251, 0.59868765, 0.62245935, 0.31002554, 0.35434368,
    0.40131232, 0.45016599, 0.50000000, 0.54983401, 0.59868771, 0.64565635,
    0.68997449, 0.73105860, 0.23147520, 0.28905049, 0.35434368, 0.42555746,
    0.50000000, 0.57444257, 0.64565635, 0.71094948, 0.76852477, 0.81757450};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor softplus_prime_result =
    input.apply<float>(nntrainer::ActiFunc::softplusPrime<float>);

  float *data = softplus_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, sigmoid_01_p) {
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

  nntrainer::Tensor Results =
    input.apply<float>(nntrainer::ActiFunc::sigmoid<float>);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, DISABLED_sigmoidPrime_01_p) {
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

  nntrainer::Tensor sigmoid_result =
    input.apply<float>(nntrainer::ActiFunc::sigmoid<float>);
  float *data = sigmoid_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result =
    sigmoid_result.apply<float>(nntrainer::ActiFunc::sigmoidPrime<float>);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, tanhFloat_01_p) {
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

  nntrainer::Tensor Results =
    input.apply<float>(nntrainer::ActiFunc::tanhFloat<float>);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, DISABLED_tanhFloatPrime_01_p) {
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

  nntrainer::Tensor tanh_result =
    input.apply<float>(nntrainer::ActiFunc::tanhFloat<float>);
  float *data = tanh_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result =
    tanh_result.apply<float>(nntrainer::ActiFunc::tanhPrime<float>);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, relu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5,
                      0, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1,
                      0, 0, 0, 0, 0, 0.3, 0.6, 0.9, 1.2, 1.5};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor Results =
    input.apply<float>(nntrainer::ActiFunc::relu<float>);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, reluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor relu_result =
    input.apply<float>(nntrainer::ActiFunc::relu<float>);
  float *data = relu_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result =
    relu_result.apply<float>(nntrainer::ActiFunc::reluPrime<float>);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, swish_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    -0.16052495, -0.12766725, -0.0900332,  -0.04750208, 0,
    0.05249792,  0.10996679,  0.17233276,  0.23947506,  0.31122968,
    -0.24802041, -0.21260624, -0.16052495, -0.0900332,  0,
    0.10996679,  0.23947506,  0.3873938,   0.5519796,   0.73105854,
    -0.27777028, -0.26014543, -0.21260624, -0.12766725, 0,
    0.17233276,  0.3873938,   0.6398545,   0.9222298,   1.2263616};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor results(batch, channel, height, width);
  results = nntrainer::ActiFunc::swish(input, results);

  float *data = results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, swishPrime_01_p) {

  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    0.30520803, 0.35221997, 0.40066269, 0.45008320, 0.50000000, 0.54991674,
    0.59933728, 0.64778000, 0.69479191, 0.73996115, 0.13889773, 0.21707317,
    0.30520803, 0.40066269, 0.50000000, 0.59933728, 0.69479191, 0.78292680,
    0.86110222, 0.92767054, 0.01800188, 0.10410020, 0.21707317, 0.35221997,
    0.50000000, 0.64778000, 0.78292680, 0.89589977, 0.98199815, 1.04129410};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor results(batch, channel, height, width);
  nntrainer::ActiFunc::swish(input, results);

  nntrainer::Tensor prime_results(batch, channel, height, width);
  nntrainer::ActiFunc::swishPrime(input, results, prime_results);

  float *data = prime_results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, gelu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    -0.13783135, -0.11462659, -0.08414805, -0.04601721, 0,
    0.05398279,  0.11585195,  0.18537343,  0.26216868,  0.34573120,
    -0.16948429, -0.16455182, -0.13783135, -0.08414805, 0,
    0.11585195,  0.26216868,  0.43544820,  0.63051575,  0.84134471,
    -0.13808367, -0.16565408, -0.16455182, -0.11462659, 0,
    0.18537343,  0.43544820,  0.73434591,  1.06191635,  1.39978909};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor results(batch, channel, height, width);
  results = nntrainer::ActiFunc::gelu(input, results);

  float *data = results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, geluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.19727029, 0.26767227, 0.34253171,  0.42047682,  0.5,         0.57952315,
    0.65746832, 0.73232776, 0.80272973,  0.86749506,  -0.01989788, 0.07431830,
    0.19727029, 0.34253171, 0.5,         0.65746832,  0.80272973,  0.92568171,
    1.01989794, 1.08331537, -0.11795351, -0.05541664, 0.07431830,  0.26767227,
    0.5,        0.73232776, 0.92568171,  1.05541658,  1.11795354,  1.12746918};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor results(batch, channel, height, width);
  nntrainer::ActiFunc::gelu(input, results);

  nntrainer::Tensor prime_results(batch, channel, height, width);
  nntrainer::ActiFunc::geluPrime(input, results, prime_results);

  float *data = prime_results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, elu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {-0.32967997, -0.25918180, -0.18126923, -0.09516257, 0.0,
                      0.1,         0.2,         0.3,         0.4,         0.5,
                      -0.55067104, -0.45118839, -0.32967997, -0.18126923, 0.0,
                      0.2,         0.4,         0.6,         0.8,         1.0,
                      -0.69880581, -0.59343034, -0.45118839, -0.25918180, 0.0,
                      0.3,         0.6,         0.9,         1.2,         1.5};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor elu_result =
    input.apply<float>(nntrainer::ActiFunc::elu<float>);
  float *data = elu_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, eluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0.67032003, 0.74081820, 0.81873077, 0.90483743, 1.0,
                      1.0,        1.0,        1.0,        1.0,        1.0,
                      0.44932896, 0.54881161, 0.67032003, 0.81873077, 1.0,
                      1.0,        1.0,        1.0,        1.0,        1.0,
                      0.30119419, 0.40656966, 0.54881161, 0.74081820, 1.0,
                      1.0,        1.0,        1.0,        1.0,        1.0};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor elu_prime_result =
    input.apply<float>(nntrainer::ActiFunc::eluPrime<float>);
  float *data = elu_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, selu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    -0.57961011, -0.45566735, -0.31868932, -0.16730525, 0.00000000,
    0.10507011,  0.21014021,  0.31521031,  0.42028043,  0.52535051,
    -0.96813440, -0.79323399, -0.57961011, -0.31868932, 0.00000000,
    0.21014021,  0.42028043,  0.63042063,  0.84056085,  1.05070102,
    -1.22856998, -1.04330945, -0.79323399, -0.45566735, 0.00000000,
    0.31521031,  0.63042063,  0.94563091,  1.26084125,  1.57605147};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor selu_result =
    input.apply<float>(nntrainer::ActiFunc::selu<float>);
  float *data = selu_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, seluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    1.17848921, 1.30243194, 1.43940997, 1.59079409, 1.75809932, 1.05070102,
    1.05070102, 1.05070102, 1.05070102, 1.05070102, 0.78996491, 0.96486533,
    1.17848921, 1.43940997, 1.75809932, 1.05070102, 1.05070102, 1.05070102,
    1.05070102, 1.05070102, 0.52952927, 0.71478987, 0.96486533, 1.30243194,
    1.75809932, 1.05070102, 1.05070102, 1.05070102, 1.05070102, 1.05070102};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor selu_prime_result =
    input.apply<float>(nntrainer::ActiFunc::seluPrime<float>);
  float *data = selu_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, mish_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    -0.18891649, -0.15113318, -0.10714479, -0.05678857, 0.00000000,
    0.06317942,  0.13259898,  0.20800139,  0.28903052,  0.37524524,
    -0.28396326, -0.24693604, -0.18891649, -0.10714479, 0.00000000,
    0.13259898,  0.28903052,  0.46613652,  0.65969974,  0.86509836,
    -0.30883580, -0.29565641, -0.24693604, -0.15113318, 0.00000000,
    0.20800139,  0.46613652,  0.76120591,  1.07794595,  1.40337825};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor mish_result =
    input.apply<float>(nntrainer::ActiFunc::mish<float>);
  float *data = mish_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, mishPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.34757277, 0.40851089, 0.47153026,  0.53570276, 0.60000002, 0.66333681,
    0.72462445, 0.78282732, 0.83701748,  0.88642442, 0.13818233, 0.23496543,
    0.34757277, 0.47153026, 0.60000002,  0.72462445, 0.83701748, 0.93047082,
    1.00125492, 1.04903626, -0.00200878, 0.09643580, 0.23496543, 0.40851089,
    0.60000002, 0.78282732, 0.93047082,  1.02791822, 1.07635069, 1.08848798};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor mish_prime_result =
    input.apply<float>(nntrainer::ActiFunc::mishPrime<float>);
  float *data = mish_prime_result.getData();
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
