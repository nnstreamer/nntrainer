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

void nntrainerModelTest::compare(
  bool opt,
  std::function<std::unique_ptr<nntrainer::NeuralNetwork>()> creator) {

  auto net = creator ? creator() : createModel();
  GraphWatcher g(std::move(net), opt);
  if (options & (ModelTestOption::USE_V2)) {
    g.compareFor_V2(getGoldenName_V2());
  } else {
    g.compareFor(getGoldenName(), getLabelDim(), getIteration());
  }
}

void nntrainerModelTest::validate(
  bool opt,
  std::function<std::unique_ptr<nntrainer::NeuralNetwork>()> creator) {

  auto net = creator ? creator() : createModel();
  GraphWatcher g(std::move(net), opt);
  if (options & (ModelTestOption::USE_V2)) {
    g.validateFor_V2();
  } else {
    g.validateFor(getLabelDim());
  }
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

  auto creator = [&saved_ini_name]() {
    std::unique_ptr<nntrainer::NeuralNetwork> nn(
      new nntrainer::NeuralNetwork());
    nn->load(saved_ini_name, ml::train::ModelFormat::MODEL_FORMAT_INI);
    if (remove(saved_ini_name.c_str())) {
      const size_t error_buflen = 100;
      char error_buf[error_buflen];
      std::cerr << "remove ini " << saved_ini_name << "failed, reason: "
                << strerror_r(errno, error_buf, error_buflen);
    }
    return nn;
  };

  compare(false, creator);
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

  auto creator = [&saved_ini_name]() {
    std::unique_ptr<nntrainer::NeuralNetwork> nn(
      new nntrainer::NeuralNetwork());
    nn->load(saved_ini_name, ml::train::ModelFormat::MODEL_FORMAT_INI);
    if (remove(saved_ini_name.c_str())) {
      const size_t error_buflen = 100;
      char error_buf[error_buflen];
      std::cerr << "remove ini " << saved_ini_name << "failed, reason: "
                << strerror_r(errno, error_buf, error_buflen);
    }
    return nn;
  };

  validate(false, creator);
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

ModelGoldenTestParamType mkModelTc_V2(
  std::function<std::unique_ptr<nntrainer::NeuralNetwork>()> generator,
  const std::string &name, ModelTestOption options) {
  /** iteration and label_dim is not used */
  return ModelGoldenTestParamType(generator, name, DIM_UNUSED, 1, options);
}
