/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file        unittest_tizen_capi.cc
 * @date        03 April 2020
 * @brief       Unit test utility.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>
#include <nntrainer.h>

/**
 * @brief Neural Network Model Contruct / Destruct Test (possitive test )
 * @return 0 success, -EADDRNOTAVIL if falied, -EFAULT if failed, -EVALID if failed.
 */
TEST(nntrainer_nnmodel_construct_deconstruct, nntrainer_01_p) {
  ml_nnmodel_h handle;
  int status;
  status = ml_nnmodel_construct(&handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Destruct Test (negative test )
 * @return 0 success, -EINVAL if failed.
 */
TEST(nntrainer_nnmodel_construct_deconstruct, nntrainer_02_n) {
  ml_nnmodel_h handle = NULL;
  int status;
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Construct wit Configuration File Test
 * @return 0 success, -EINVAL if failed.
 */
TEST(nntrainer_nnmodel_construct_deconstruct, nntrainer_04_n) {
  ml_nnmodel_h handle;
  const char *model_conf = "/test/cannot_find.ini";
  int status;
  status = ml_nnmodel_construct_with_conf(model_conf, &handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Construct wit Configuration File Test
 * @return 0 success, -EINVAL if failed.
 */
TEST(nntrainer_nnmodel_construct_deconstruct, nntrainer_05_p) {
  ml_nnmodel_h handle = NULL;
  const char *model_conf = "../test/tizen_capi/test_conf.ini";
  int status;
  status = ml_nnmodel_construct_with_conf(model_conf, &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
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
