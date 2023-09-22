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
#include <gtest/gtest.h>

#include <nntrainer.h>
#include <nntrainer_internal.h>
#include <nntrainer_test_util.h>

/**
 * @brief Neural Network Optimizer Create / Delete Test (positive test)
 */
TEST(nntrainer_capi_nnopt, create_delete_01_p) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_SGD);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Create / Delete Test (positive test )
 */
TEST(nntrainer_capi_nnopt, create_delete_02_p) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Create / Delete Test (negative test )
 */
TEST(nntrainer_capi_nnopt, create_delete_03_n) {
  ml_train_optimizer_h handle = NULL;
  int status;
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Optimizer Create / Delete Test (positive test)
 */
TEST(nntrainer_capi_nnopt, create_delete_04_n) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Optimizer set Property Test (positive test)
 */
TEST(nntrainer_capi_nnopt, setProperty_01_p) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_optimizer_set_property(handle, "beta1=0.002", "beta2=0.001", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Set Property Test (positive test)
 */
TEST(nntrainer_capi_nnopt, setProperty_02_p) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_set_property(
    handle, "learning_rate=0.0001 | decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnopt, setProperty_03_n) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_optimizer_set_property(handle, "beta1=true", "beta2=0.001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnopt, setProperty_04_n) {
  ml_train_optimizer_h handle = NULL;
  int status;

  status = ml_train_optimizer_set_property(
    handle, "learning_rate=0.0001 | decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Optimizer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnopt, setProperty_05_n) {
  ml_train_optimizer_h handle = NULL;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_set_property(handle, "unknown=unknown", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Optimizer Set Property Test (positive test)
 */
TEST(nntrainer_capi_nnopt, setProperty_with_single_param_01_p) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_set_property_with_single_param(
    handle, "beta1=0.002 | beta2=0.001 | epsilon=1e-7");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnopt, setProperty_with_single_param_02_n) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_set_property_with_single_param(
    handle, "beta1=0.002, beta2=0.001, epsilon=1e-7");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Optimizer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnopt, setProperty_with_single_param_03_n) {
  ml_train_optimizer_h handle;
  int status;
  status = ml_train_optimizer_create(&handle, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_optimizer_set_property_with_single_param(
    handle, "beta1=0.002 ! beta2=0.001 ! epsilon=1e-7");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_optimizer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Set learning rate scheduler to Neural Network Optimizer Test (positive
 * test)
 */
TEST(nntrainer_capi_nnopt, set_lr_scheduler_01_p) {
  ml_train_optimizer_h opt_handle;
  ml_train_lr_scheduler_h lr_sched_handle;
  int status;
  status = ml_train_optimizer_create(&opt_handle, ML_TRAIN_OPTIMIZER_TYPE_SGD);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_lr_scheduler_create(&lr_sched_handle,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_CONSTANT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_lr_scheduler(opt_handle, lr_sched_handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_destroy(opt_handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  /** ignore tizen feature check while running the testcases */
  set_feature_state(ML_FEATURE, SUPPORTED);
  set_feature_state(ML_FEATURE_INFERENCE, SUPPORTED);
  set_feature_state(ML_FEATURE_TRAINING, SUPPORTED);

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  /** reset tizen feature check state */
  set_feature_state(ML_FEATURE, NOT_CHECKED_YET);
  set_feature_state(ML_FEATURE_INFERENCE, NOT_CHECKED_YET);
  set_feature_state(ML_FEATURE_TRAINING, NOT_CHECKED_YET);

  return result;
}
