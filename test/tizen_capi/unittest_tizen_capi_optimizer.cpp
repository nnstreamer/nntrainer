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
 * @file        unittest_tizen_capi_optimizer.cpp
 * @date        12 May 2020
 * @brief       Unit test utility for optimizer.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include "nntrainer_test_util.h"
#include <nntrainer.h>

/**
 * @brief Neural Network Optimizer Create / Delete Test (possitive test)
 */
TEST(nntrainer_capi_nnopt, create_delete_01_p) {
  ml_nnopt_h handle;
  int status;
  status = ml_nnoptimizer_create(&handle, "sgd");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Create / Delete Test (possitive test )
 */
TEST(nntrainer_capi_nnopt, create_delete_02_p) {
  ml_nnopt_h handle;
  int status;
  status = ml_nnoptimizer_create(&handle, "adam");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Create / Delete Test (possitive test )
 */
TEST(nntrainer_capi_nnopt, create_delete_03_n) {
  ml_nnopt_h handle = NULL;
  int status;
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Optimizer Create / Delete Test (possitive test)
 */
TEST(nntrainer_capi_nnopt, create_delete_04_n) {
  ml_nnopt_h handle;
  int status;
  status = ml_nnoptimizer_create(&handle, "adaam");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  testing::InitGoogleTest(&argc, argv);

  result = RUN_ALL_TESTS();

  return result;
}
