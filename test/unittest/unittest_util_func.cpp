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

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <nntrainer_logger.h>
#include <nntrainer_test_util.h>
#include <util_func.h>

TEST(nntrainer_util_func, sqrtFloat_01_p) {
  float x = 9871.0;
  float sx = nntrainer::sqrtFloat(x);

  EXPECT_NEAR(sx * sx, x, tolerance * 10);
}

TEST(nntrainer_util_func, absFloat_01_p) {
  float x = -9871.0;
  float sx = nntrainer::absFloat(x);

  EXPECT_NEAR(sx, -1 * x, tolerance);
}

TEST(nntrainer_util_func, logFloat_01_p) {
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (width) + k + 1);

  nntrainer::Tensor Results = input.apply<float>(nntrainer::logFloat<float>);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], (float)log(indata[i]), tolerance);
  }
}

TEST(nntrainer_util_func, exp_util_01_p) {
  float x = 1;
  float ex = nntrainer::exp_util(x);
  float ans = 2.718281828459;

  EXPECT_NEAR(ex, ans, tolerance);
}

TEST(nntrainer_util_func, checkedRead_n) {
  std::ifstream file("not existing file");
  char array[5];

  EXPECT_THROW(nntrainer::checkedRead(file, array, 5), std::runtime_error);
}

TEST(nntrainer_util_func, checkedWrite_n) {
  std::ofstream file("!@/not good file");
  char array[5] = "abcd";

  EXPECT_THROW(nntrainer::checkedWrite(file, array, 5), std::runtime_error);
}

TEST(nntrainer_util_func, readString_n) {
  std::ifstream file("not existing file");

  EXPECT_THROW(nntrainer::readString(file), std::runtime_error);
}

TEST(nntrainer_util_func, writeString_n) {
  std::ofstream file("!@/not good file");
  std::string str = "abcd";

  EXPECT_THROW(nntrainer::writeString(file, str), std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_no_error_p) {
  EXPECT_NO_THROW(nntrainer::throw_status(ML_ERROR_NONE));
}

TEST(nntrainer_util_func, throw_status_invalid_argument_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_INVALID_PARAMETER),
               std::invalid_argument);
}

TEST(nntrainer_util_func, throw_status_try_again_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_TRY_AGAIN), std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_not_supported_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_NOT_SUPPORTED),
               std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_out_of_memory_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_OUT_OF_MEMORY), std::bad_alloc);
}

TEST(nntrainer_util_func, throw_status_timed_out_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_TIMED_OUT), std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_permission_denied_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_PERMISSION_DENIED),
               std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_unknown_error_n) {
  EXPECT_THROW(nntrainer::throw_status(ML_ERROR_UNKNOWN), std::runtime_error);
}

TEST(nntrainer_util_func, throw_status_default_n) {
  EXPECT_THROW(nntrainer::throw_status(-12345), std::runtime_error);
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
