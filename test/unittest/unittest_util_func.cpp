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
 * @author      Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug         No known bugs
 */


#include <gtest/gtest.h>
#include <nntrainer_error.h>
#include <neuralnet.h>
#include <nntrainer_log.h>
#include <util_func.h>

#define tolerance 10e-5

#define GEN_TEST_INPUT(input, eqation_i_j_k) \
  do {                                       \
    for (int i = 0; i < batch; ++i) {        \
      for (int j = 0; j < height; ++j) {     \
        for (int k = 0; k < width; ++k) {    \
          float val = eqation_i_j_k;         \
          input.setValue(i, j, k, val);      \
        }                                    \
      }                                      \
    }                                        \
  } while (0)

/**
 * @brief Data Buffer
 */
TEST(nntrainer_util_func, softmax_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 1;
  int width = 10;
  float results[10] = {7.80134161e-05, 2.12062451e-04, 5.76445508e-04,
                       1.56694135e-03, 4.25938820e-03, 1.15782175e-02,
                       3.14728583e-02, 8.55520989e-02, 2.32554716e-01,
                       6.32149258e-01};

  nntrainer::Tensor T(batch, height, width);
  nntrainer::Tensor Results(batch, height, width);

  GEN_TEST_INPUT(T, (i * (width) + k + 1));

  Results = T.apply(nntrainer::softmax);
  float *data = Results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    if ((data[i] - results[i % width]) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  EXPECT_EQ(status, ML_ERROR_NONE);
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

  if ((sx * sx - x) * (sx * sx - x) < tolerance)
    status = ML_ERROR_NONE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, logFloat_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, i * (width) + k + 1);

  nntrainer::Tensor Results = input.apply(nntrainer::logFloat);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if ((data[i] - (float)log(indata[i])) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, sigmoid_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, (k - 4) * 0.1);

  nntrainer::Tensor Results = input.apply(nntrainer::sigmoid);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if ((data[i] - (float)(1 / (1 + exp(-indata[i])))) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, tanhFloat_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, (k - 4) * 0.1);

  nntrainer::Tensor Results = input.apply(nntrainer::tanhFloat);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if ((data[i] - (float)(tanh(indata[i]))) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_util_func, relu_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, height, width);
  GEN_TEST_INPUT(input, (k - 4) * 0.1);

  nntrainer::Tensor Results = input.apply(nntrainer::relu);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    float r = (indata[i] <= 0.0) ? 0.0 : indata[i];
    if ((data[i] - r) > tolerance) {
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
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

