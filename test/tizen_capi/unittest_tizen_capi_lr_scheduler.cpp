/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file        unittest_tizen_capi_lr_scheduler.cpp
 * @date        13 April 2023
 * @brief       Unit test utility for learning rate scheduler.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <nntrainer.h>
#include <nntrainer_internal.h>
#include <nntrainer_test_util.h>

/**
 * @brief Learning rate scheduler Create / Destruct Test (positive test)
 */
TEST(nntrainer_capi_lr_scheduler, create_destruct_01_p) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_CONSTANT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler Create / Destruct Test (positive test)
 */
TEST(nntrainer_capi_lr_scheduler, create_destruct_02_p) {
  ml_train_lr_scheduler_h handle;
  int status;
  status = ml_train_lr_scheduler_create(&handle,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler Create / Destruct Test (positive test)
 */
TEST(nntrainer_capi_lr_scheduler, create_destruct_03_p) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_STEP);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler Create / Destruct Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, create_destruct_04_n) {
  ml_train_lr_scheduler_h handle = NULL;
  int status;
  status = ml_train_lr_scheduler_destroy(&handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Learning rate scheduler Create / Destruct Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, create_destruct_05_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Learning rate scheduler Create / Destruct Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, create_destruct_06_n) {
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

  status = ml_train_lr_scheduler_destroy(lr_sched_handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_optimizer_destroy(opt_handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler Create / Destruct Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, create_destruct_07_n) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_optimizer_h optimizer;
  ml_train_lr_scheduler_h lr_scheduler;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(optimizer, "beta1=0.002",
                                           "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_lr_scheduler_create(&lr_scheduler,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_lr_scheduler_set_property(
    lr_scheduler, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_lr_scheduler(optimizer, lr_scheduler);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_lr_scheduler_destroy(lr_scheduler);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (positive test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_01_p) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_CONSTANT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_lr_scheduler_set_property(handle, "learning_rate=0.001", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (positive test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_02_p) {
  ml_train_lr_scheduler_h handle;
  int status;
  status = ml_train_lr_scheduler_create(&handle,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property(
    handle, "learning_rate=0.001", "decay_rate=0.9", "decay_steps=2", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (positive test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_03_p) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_STEP);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property(
    handle, "learning_rate=0.01, 0.001", "iteration=100", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_04_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_CONSTANT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property(handle, "learning_rate=0.001",
                                              "iteration=10", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_05_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_CONSTANT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property(handle, "learning_rate=0.001",
                                              "decay_rate=0.9", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_06_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_CONSTANT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property(handle, "learning_rate=0.001",
                                              "decay_steps=2", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_07_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status = ml_train_lr_scheduler_create(&handle,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property(handle, "learning_rate=0.001",
                                              "iteration=100", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_08_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_STEP);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property(handle, "learning_rate=0.001",
                                              "decay_rate=0.9", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_09_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status =
    ml_train_lr_scheduler_create(&handle, ML_TRAIN_LR_SCHEDULER_TYPE_STEP);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property(handle, "learning_rate=0.001",
                                              "decay_steps=2", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (positive test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_with_single_param_01_p) {
  ml_train_lr_scheduler_h handle;
  int status;
  status = ml_train_lr_scheduler_create(&handle,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property_with_single_param(
    handle, "learning_rate=0.001 | decay_rate=0.9 | decay_steps=2");
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_with_single_param_02_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status = ml_train_lr_scheduler_create(&handle,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property_with_single_param(
    handle, "learning_rate=0.001, decay_rate=0.9, decay_steps=2");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_lr_scheduler_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Learning rate scheduler set Property Test (negative test)
 */
TEST(nntrainer_capi_lr_scheduler, setProperty_with_single_param_03_n) {
  ml_train_lr_scheduler_h handle;
  int status;
  status = ml_train_lr_scheduler_create(&handle,
                                        ML_TRAIN_LR_SCHEDULER_TYPE_EXPONENTIAL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_lr_scheduler_set_property_with_single_param(
    handle, "learning_rate=0.001 ! decay_rate=0.9 ! decay_steps=2");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_lr_scheduler_destroy(handle);
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
    std::cerr << "Error duing IniGoogleTest" << std::endl;
    return 0;
  }

  /** ignore tizen feature check while running the testcases */
  set_feature_state(ML_FEATURE, SUPPORTED);
  set_feature_state(ML_FEATURE_INFERENCE, SUPPORTED);
  set_feature_state(ML_FEATURE_TRAINING, SUPPORTED);

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  /** reset tizen feature check state */
  set_feature_state(ML_FEATURE, NOT_CHECKED_YET);
  set_feature_state(ML_FEATURE_INFERENCE, NOT_CHECKED_YET);
  set_feature_state(ML_FEATURE_TRAINING, NOT_CHECKED_YET);

  return result;
}
