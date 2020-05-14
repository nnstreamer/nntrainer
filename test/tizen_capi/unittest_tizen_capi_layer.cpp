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
 * @file        unittest_tizen_capi_layer.cc
 * @date        03 April 2020
 * @brief       Unit test utility for layer.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include "nntrainer_test_util.h"
#include <nntrainer.h>

/**
 * @brief Neural Network Layer Create / Delete Test (possitive test)
 */
TEST(nntrainer_capi_nnlayer, create_delete_01_p) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Create / Delete Test (possitive test)
 */
TEST(nntrainer_capi_nnlayer, create_delete_02_p) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Create / Delete Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, create_delete_03_n) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Layer Set Property Test (positive test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_01_p) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "input_shape=32:1:1:6270", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(handle, "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(handle, "standardization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (positive test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_02_p) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "unit=10", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(handle, "bias_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(handle, "activation =sigmoid", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_03_n) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "activation= sigmoid", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_04_n) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "input_shape=0:0:0:1", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_05_n) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "epsilon =0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_06_n) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "epsilon =0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_07_n) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "activation=0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (positive test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_08_p) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "weight_decay=l2norm",
                                   "weight_decay_lambda=0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negitive test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_09_n) {
  ml_nnlayer_h handle;
  int status;
  status = ml_nnlayer_create(&handle, ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_set_property(handle, "weight_decay=asdfasd",
                                   "weight_decay_lambda=0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnlayer_delete(handle);
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
