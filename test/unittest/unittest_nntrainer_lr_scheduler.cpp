// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   unittest_nntrainer_lr_scheduler.cpp
 * @date   09 December 2021
 * @brief  This is unittests for learning rate schedulers
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <fstream>
#include <gtest/gtest.h>

#include <lr_scheduler.h>
#include <lr_scheduler_constant.h>
#include <lr_scheduler_exponential.h>
#include <nntrainer_error.h>

#include "nntrainer_test_util.h"

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_01_p) {
  EXPECT_NO_THROW(nntrainer::createLearningRateScheduler<
                  nntrainer::ConstantLearningRateScheduler>({}));
}

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_02_p) {
  auto &ac = nntrainer::AppContext::Global();
  EXPECT_NO_THROW(
    ac.createObject<nntrainer::LearningRateScheduler>("constant"));
}

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_03_n) {
  auto &ac = nntrainer::AppContext::Global();
  EXPECT_ANY_THROW(ac.createObject<nntrainer::LearningRateScheduler>("random"));
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_constant, prop_01_n) {
  auto &ac = nntrainer::AppContext::Global();
  auto lr = ac.createObject<nntrainer::LearningRateScheduler>("constant");

  /** fails as learning rate property is not set */
  EXPECT_ANY_THROW(lr->getLearningRate(0));
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_constant, prop_02_n) {
  auto &ac = nntrainer::AppContext::Global();
  auto lr = ac.createObject<nntrainer::LearningRateScheduler>("constant");

  EXPECT_ANY_THROW(lr->setProperty({"random=1.0"}));
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_constant, prop_03_p) {
  auto &ac = nntrainer::AppContext::Global();
  auto lr = ac.createObject<nntrainer::LearningRateScheduler>("constant");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));

  EXPECT_NO_THROW(lr->getLearningRate(0));
  EXPECT_FLOAT_EQ(lr->getLearningRate(0), lr->getLearningRate(100));
  EXPECT_FLOAT_EQ(lr->getLearningRate(10), 1.0f);
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_exponential, prop_01_n) {
  auto &ac = nntrainer::AppContext::Global();
  auto lr = ac.createObject<nntrainer::LearningRateScheduler>("exponential");

  EXPECT_ANY_THROW(lr->getLearningRate(0));
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_exponential, prop_02_p) {
  auto &ac = nntrainer::AppContext::Global();
  auto lr = ac.createObject<nntrainer::LearningRateScheduler>("exponential");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));
  EXPECT_ANY_THROW(lr->getLearningRate(0));

  EXPECT_NO_THROW(lr->setProperty({"decay_steps=1"}));
  EXPECT_ANY_THROW(lr->getLearningRate(0));

  EXPECT_NO_THROW(lr->setProperty({"decay_rate=0.9"}));
  EXPECT_NO_THROW(lr->getLearningRate(0));

  EXPECT_FLOAT_EQ(lr->getLearningRate(0), 1.0f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(1), 1.0 * std::pow(0.9f, 1));
  EXPECT_FLOAT_EQ(lr->getLearningRate(3), 1.0 * std::pow(0.9f, 3));
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
