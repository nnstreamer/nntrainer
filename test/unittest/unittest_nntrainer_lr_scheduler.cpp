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

static std::unique_ptr<nntrainer::LearningRateScheduler>
createLRS(const std::string &type) {
  auto &ac = nntrainer::AppContext::Global();
  auto lrs = ac.createObject<ml::train::LearningRateScheduler>(type);
  auto lrs_ptr = static_cast<nntrainer::LearningRateScheduler *>(lrs.release());
  return std::unique_ptr<nntrainer::LearningRateScheduler>(lrs_ptr);
}

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
  EXPECT_NO_THROW(createLRS("constant"));
}

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_03_n) {
  EXPECT_ANY_THROW(createLRS("random"));
}

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_04_n) {
  EXPECT_THROW(nntrainer::createLearningRateScheduler<
                 nntrainer::ConstantLearningRateScheduler>({"unknown"}),
               std::invalid_argument);
}

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_05_n) {
  EXPECT_THROW(nntrainer::createLearningRateScheduler<
                 nntrainer::ConstantLearningRateScheduler>({"lr=0.1"}),
               std::invalid_argument);
}

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_06_n) {
  EXPECT_THROW(
    nntrainer::createLearningRateScheduler<
      nntrainer::ConstantLearningRateScheduler>({"learning_rate:0.1"}),
    std::invalid_argument);
}

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_07_n) {
  EXPECT_THROW(
    nntrainer::createLearningRateScheduler<
      nntrainer::ConstantLearningRateScheduler>({"learning_rate(0.1)"}),
    std::invalid_argument);
}

/**
 * @brief test constructing lr scheduler
 *
 */
TEST(lr_constant, ctor_initializer_08_n) {
  EXPECT_THROW(nntrainer::createLearningRateScheduler<
                 nntrainer::ConstantLearningRateScheduler>({"0.1"}),
               std::invalid_argument);
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_constant, prop_01_n) {
  auto lr = createLRS("constant");

  /** fails as learning rate property is not set */
  EXPECT_ANY_THROW(lr->getLearningRate(0));
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_constant, prop_02_n) {
  auto lr = createLRS("constant");
  EXPECT_THROW(lr->setProperty({"unknown=unknown"}), std::invalid_argument);
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_constant, prop_03_p) {
  auto lr = createLRS("constant");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));

  EXPECT_NO_THROW(lr->getLearningRate(0));
  EXPECT_FLOAT_EQ(lr->getLearningRate(0), lr->getLearningRate(100));
  EXPECT_FLOAT_EQ(lr->getLearningRate(10), 1.0f);
}

/**
 * @brief test set property with wrong format
 *
 */
TEST(lr_constant, prop_04_n) {
  auto lr = createLRS("constant");
  EXPECT_THROW(lr->setProperty({"learning_rate:0.1"}), std::invalid_argument);
}

/**
 * @brief test set property with wrong format
 *
 */
TEST(lr_constant, prop_05_n) {
  auto lr = createLRS("constant");
  EXPECT_THROW(lr->setProperty({"learning_rate(0.1)"}), std::invalid_argument);
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_constant, final_01_n) {
  auto lr = createLRS("constant");

  /** fails as learning rate property is not set */
  EXPECT_ANY_THROW(lr->finalize());
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_constant, final_02_p) {
  auto lr = createLRS("constant");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));
  EXPECT_NO_THROW(lr->finalize());
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_exponential, get_learing_rate_01_n) {
  auto lr = createLRS("exponential");
  EXPECT_ANY_THROW(lr->getLearningRate(0));
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_exponential, get_learning_rate_02_p) {
  auto lr = createLRS("exponential");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));
  EXPECT_ANY_THROW(lr->finalize());
  EXPECT_ANY_THROW(lr->getLearningRate(0));

  EXPECT_NO_THROW(lr->setProperty({"decay_steps=1"}));
  EXPECT_ANY_THROW(lr->finalize());
  EXPECT_ANY_THROW(lr->getLearningRate(0));

  EXPECT_NO_THROW(lr->setProperty({"decay_rate=0.9"}));
  EXPECT_NO_THROW(lr->finalize());
  EXPECT_NO_THROW(lr->getLearningRate(0));

  EXPECT_FLOAT_EQ(lr->getLearningRate(0), 1.0f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(1), 1.0 * std::pow(0.9f, 1));
  EXPECT_FLOAT_EQ(lr->getLearningRate(3), 1.0 * std::pow(0.9f, 3));
}

/**
 * @brief test set property
 *
 */
TEST(lr_exponential, prop_01_n) {
  auto lr = createLRS("exponential");
  EXPECT_THROW(lr->setProperty({"decay_steps=0"}), std::invalid_argument);
}

/**
 * @brief test set property
 *
 */
TEST(lr_exponential, prop_02_n) {
  auto lr = createLRS("exponential");
  EXPECT_THROW(lr->setProperty({"unknown=unknown"}), std::invalid_argument);
}

/**
 * @brief test set property with wrong format
 *
 */
TEST(lr_exponential, prop_03_n) {
  auto lr = createLRS("exponential");
  EXPECT_THROW(lr->setProperty({"learning_rate:0.1"}), std::invalid_argument);
}
/**
 * @brief test finalize
 *
 */
TEST(lr_exponential, finalize_01_n) {
  auto lr = createLRS("exponential");
  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));
  EXPECT_NO_THROW(lr->setProperty({"decay_rate=0.9"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_exponential, finalize_02_n) {
  auto lr = createLRS("exponential");
  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));
  EXPECT_NO_THROW(lr->setProperty({"decay_steps=1"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_exponential, finalize_03_n) {
  auto lr = createLRS("exponential");
  EXPECT_NO_THROW(lr->setProperty({"decay_steps=1"}));
  EXPECT_NO_THROW(lr->setProperty({"decay_rate=0.9"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test set property
 *
 */
TEST(lr_step, prop_01_n) {
  auto lr = createLRS("step");
  EXPECT_THROW(lr->setProperty({"unknown=unknown"}), std::invalid_argument);
}
/**
 * @brief test set property with wrong format
 *
 */
TEST(lr_step, prop_02_n) {
  auto lr = createLRS("step");
  EXPECT_THROW(lr->setProperty({"learning_rate:0.1"}), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_step, finalize_01_n) {
  auto lr = createLRS("step");

  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_step, finalize_02_n) {
  auto lr = createLRS("step");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_step, finalize_03_n) {
  auto lr = createLRS("step");

  EXPECT_NO_THROW(lr->setProperty({"iteration=1"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_step, finalize_04_n) {
  auto lr = createLRS("step");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0, 0.1"}));
  EXPECT_NO_THROW(lr->setProperty({"iteration=1,2"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_step, finalize_05_n) {
  auto lr = createLRS("step");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0"}));
  EXPECT_NO_THROW(lr->setProperty({"iteration=1,2"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_step, finalize_06_n) {
  auto lr = createLRS("step");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0, 0.1, 0.01"}));
  EXPECT_NO_THROW(lr->setProperty({"iteration=1,1"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test finalize
 *
 */
TEST(lr_step, finalize_07_n) {
  auto lr = createLRS("step");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0, 0.1, 0.01"}));
  EXPECT_NO_THROW(lr->setProperty({"iteration=2,1"}));
  EXPECT_THROW(lr->finalize(), std::invalid_argument);
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_step, get_learning_rate_01_p) {
  auto lr = createLRS("step");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0, 0.1"}));
  EXPECT_ANY_THROW(lr->finalize());

  EXPECT_NO_THROW(lr->setProperty({"iteration=1"}));
  EXPECT_NO_THROW(lr->finalize());
  EXPECT_NO_THROW(lr->getLearningRate(0));

  EXPECT_FLOAT_EQ(lr->getLearningRate(0), 1.0f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(1), 1.0f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(3), 0.1f);
}

/**
 * @brief test set and get learning rate
 *
 */
TEST(lr_step, get_learning_rate_02_p) {
  auto lr = createLRS("step");

  EXPECT_NO_THROW(lr->setProperty({"learning_rate=1.0, 0.1, 0.01, 0.001"}));

  EXPECT_NO_THROW(lr->setProperty({"iteration=10,20,30"}));
  EXPECT_NO_THROW(lr->finalize());

  /** step 0 */
  EXPECT_FLOAT_EQ(lr->getLearningRate(0), 1.0f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(5), 1.0f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(10), 1.0f)
  /** step 1 */;
  EXPECT_FLOAT_EQ(lr->getLearningRate(11), 0.1f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(15), 0.1f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(20), 0.1f);
  /** step 2 */
  EXPECT_FLOAT_EQ(lr->getLearningRate(21), 0.01f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(25), 0.01f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(30), 0.01f);
  /** step 3 */
  EXPECT_FLOAT_EQ(lr->getLearningRate(31), 0.001f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(35), 0.001f);
  EXPECT_FLOAT_EQ(lr->getLearningRate(1000), 0.001f);
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
