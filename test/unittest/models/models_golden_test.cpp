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
#include <neuralnet.h>

/**
 * @brief check given ini is failing/suceeding at unoptimized running
 */
TEST_P(nntrainerModelTest, model_test) {
  if (!shouldCompare()) {
    std::cout << "[ SKIPPED  ] option not enabled \n";
    return;
  }
  /** Check model with all optimizations off */

  GraphWatcher g_unopt(createModel(), false);
  g_unopt.compareFor(getGoldenName(), getLabelDim(), getIteration());

  /// add stub test for tcm
  EXPECT_TRUE(true);
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

  GraphWatcher g_opt(createModel(), true);
  g_opt.compareFor(getGoldenName(), getLabelDim(), getIteration());

  /// add stub test for tcm
  EXPECT_TRUE(true);
}

/**
 * @brief check given ini is failing/suceeding at validation
 */
TEST_P(nntrainerModelTest, model_test_validate) {
  /** Check model with all optimizations on */
  GraphWatcher g_opt(createModel(), true);
  g_opt.validateFor(getLabelDim());

  /// add stub test for tcm
  EXPECT_TRUE(true);
}

TEST_P(nntrainerModelTest, model_test_save_load_compare) {
  if (!shouldSaveLoadIniTest() || !shouldCompare()) {
    std::cout << "[ SKIPPED  ] option not enabled \n";
    return;
  }

  auto nn = createModel();
  EXPECT_NO_THROW(nn->compile());
  EXPECT_NO_THROW(nn->initialize());

  auto saved_ini_name = getName() + "_saved.ini";
  if (remove(saved_ini_name.c_str())) {
    /// do nothing
  }
  EXPECT_NO_THROW(
    nn->save(saved_ini_name, ml::train::ModelFormat::MODEL_FORMAT_INI));

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

  auto nn = createModel();
  EXPECT_NO_THROW(nn->compile());
  EXPECT_NO_THROW(nn->initialize());

  auto saved_ini_name = getName() + "_saved.ini";
  if (remove(saved_ini_name.c_str())) {
    /// do nothing
  }
  nn->save(saved_ini_name, ml::train::ModelFormat::MODEL_FORMAT_INI);

  GraphWatcher g(saved_ini_name, true);
  g.validateFor(getLabelDim());
  if (remove(saved_ini_name.c_str())) {
    std::cerr << "remove ini " << saved_ini_name
              << "failed, reason: " << strerror(errno);
  }
}

ModelGoldenTestParamType mkModelIniTc(const nntrainer::IniWrapper &ini,
                                      const std::string &label_dim,
                                      const unsigned int iteration,
                                      ModelTestOption options) {
  auto generator = [ini]() mutable {
    std::unique_ptr<nntrainer::NeuralNetwork> nn(
      new nntrainer::NeuralNetwork());
    ini.save_ini();
    nn->load(ini.getIniName(), ml::train::ModelFormat::MODEL_FORMAT_INI);
    ini.erase_ini();
    return nn;
  };

  return ModelGoldenTestParamType(generator, ini.getName(),
                                  nntrainer::TensorDim(label_dim), iteration,
                                  options);
}

ModelGoldenTestParamType
mkModelTc(std::function<std::unique_ptr<nntrainer::NeuralNetwork>()> generator,
          const std::string &name, const std::string &label_dim,
          const unsigned int iteration, ModelTestOption options) {
  return ModelGoldenTestParamType(generator, name, label_dim, iteration,
                                  options);
}
