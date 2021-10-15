// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   models_golden_test.cpp
 * @date   15 Oct 2020
 * @brief  Model parameterized golden test
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <models_golden_test.h>
#include <models_test_utils.h>

#include <gtest/gtest.h>

/**
 * @brief check given ini is failing/suceeding at unoptimized running
 */
TEST_P(nntrainerModelTest, model_test) {
  if (!shouldCompare()) {
    std::cout << "[ SKIPPED  ] option not enabled \n";
    return;
  }
  /** Check model with all optimizations off */

  GraphWatcher g_unopt(getIniName(), false);
  g_unopt.compareFor(getGoldenName(), getLabelDim(), getIteration());

  /// add stub test for tcm
  EXPECT_EQ(std::get<0>(GetParam()), std::get<0>(GetParam()));
}

/**
 * @brief check given ini is failing/suceeding at optimized running
 */
TEST_P(nntrainerModelTest, model_test_optimized) {
  if (!shouldCompare()) {
    std::cout << "[ SKIPPED  ] option not enabled \n";
    return;
  }
  /** Check model with all optimizations on */

  GraphWatcher g_opt(getIniName(), true);
  g_opt.compareFor(getGoldenName(), getLabelDim(), getIteration());

  /// add stub test for tcm
  EXPECT_EQ(std::get<0>(GetParam()), std::get<0>(GetParam()));
}

/**
 * @brief check given ini is failing/suceeding at validation
 */
TEST_P(nntrainerModelTest, model_test_validate) {
  /** Check model with all optimizations on */
  GraphWatcher g_opt(getIniName(), true);
  g_opt.validateFor(getLabelDim());

  /// add stub test for tcm
  EXPECT_EQ(std::get<0>(GetParam()), std::get<0>(GetParam()));
}

TEST_P(nntrainerModelTest, model_test_save_load_compare) {
  if (!shouldSaveLoadIniTest() || !shouldCompare()) {
    std::cout << "[ SKIPPED  ] option not enabled \n";
    return;
  }

  auto nn = nntrainer::NeuralNetwork();
  EXPECT_NO_THROW(nn.loadFromConfig(getIniName()));
  EXPECT_NO_THROW(nn.compile());
  EXPECT_NO_THROW(nn.initialize());

  auto saved_ini_name = getName() + "_saved.ini";
  if (remove(saved_ini_name.c_str())) {
    /// do nothing
  }
  EXPECT_NO_THROW(
    nn.save(saved_ini_name, ml::train::ModelFormat::MODEL_FORMAT_INI));

  GraphWatcher g(saved_ini_name, false);
  g.compareFor(getGoldenName(), getLabelDim(), getIteration());
  if (remove(saved_ini_name.c_str())) {
    std::cerr << "remove ini " << saved_ini_name
              << "failed, reason: " << strerror(errno);
  }
}

TEST_P(nntrainerModelTest, model_test_save_load_verify) {
  if (!shouldSaveLoadIniTest()) {
    std::cout << "[ SKIPPED  ] option not enabled \n";
    return;
  }

  auto nn = nntrainer::NeuralNetwork();

  EXPECT_NO_THROW(nn.loadFromConfig(getIniName()));
  EXPECT_NO_THROW(nn.compile());
  EXPECT_NO_THROW(nn.initialize());

  auto saved_ini_name = getName() + "_saved.ini";
  if (remove(saved_ini_name.c_str())) {
    /// do nothing
  }
  nn.save(saved_ini_name, ml::train::ModelFormat::MODEL_FORMAT_INI);

  GraphWatcher g(saved_ini_name, true);
  g.validateFor(getLabelDim());
  if (remove(saved_ini_name.c_str())) {
    std::cerr << "remove ini " << saved_ini_name
              << "failed, reason: " << strerror(errno);
  }
}

std::tuple<const nntrainer::IniWrapper, const nntrainer::TensorDim,
           const unsigned int, ModelTestOption>
mkModelTc(const nntrainer::IniWrapper &ini, const std::string &label_dim,
          const unsigned int iteration, ModelTestOption options) {
  return std::tuple<const nntrainer::IniWrapper, const nntrainer::TensorDim,
                    const unsigned int, ModelTestOption>(
    ini, nntrainer::TensorDim(label_dim), iteration, options);
}
