// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   models_golden_test.h
 * @date   15 Oct 2020
 * @brief  Model parameterized golden test
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __MODELS_GOLDEN_TEST_H__
#define __MODELS_GOLDEN_TEST_H__

#include <gtest/gtest.h>

#include <tuple>

#include <ini_wrapper.h>
#include <tensor_dim.h>

/**
 * @brief Test Option for the unittest models
 *
 */
typedef enum {
  COMPARE = 1 << 0,           /**< Set this to compare the numbers */
  SAVE_AND_LOAD_INI = 1 << 1, /**< Set this to check if saving and constructing
                                 a new model works okay (without weights) */
  NO_THROW_RUN = 0, /**< no comparison, only validate execution without throw */
  ALL = COMPARE | SAVE_AND_LOAD_INI /**< Set every option */
} ModelTestOption;

/**
 * @brief nntrainerModelTest fixture for parametrized test
 *
 * @param nntrainer::IniWrapper ini data
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 */
class nntrainerModelTest
  : public ::testing::TestWithParam<
      std::tuple<const nntrainer::IniWrapper /**< Model Architecture */,
                 const nntrainer::TensorDim /**< InputDimension */,
                 const unsigned int /**< Number of Iterations */,
                 ModelTestOption /**< Options which test to run */>> {

protected:
  nntrainerModelTest() :
    iteration(0),
    name(""),
    options(ModelTestOption::NO_THROW_RUN) {}
  virtual void SetUp() {
    auto param = GetParam();

    ini = std::get<0>(param);
    /// remove the test number after double __
    name = ini.getName();
    name = name.substr(0, name.find("__"));

    label_dim = std::get<1>(param);
    iteration = std::get<2>(param);
    options = std::get<3>(param);
    ini.save_ini();
  }

  virtual void TearDown() { ini.erase_ini(); }

  std::string getIniName() { return ini.getIniName(); }
  std::string getName() { return name; }
  std::string getGoldenName() { return name + ".info"; }
  int getIteration() { return iteration; };
  nntrainer::TensorDim getLabelDim() { return label_dim; }

  bool shouldCompare() {
    return (options & ModelTestOption::COMPARE) == ModelTestOption::COMPARE;
  }
  bool shouldSaveLoadIniTest() {
    return options & ModelTestOption::SAVE_AND_LOAD_INI;
  }

private:
  nntrainer::TensorDim label_dim;
  int iteration;
  std::string name;
  nntrainer::IniWrapper ini;
  ModelTestOption options;
};

/**
 * @brief helper function to make model testcase
 *
 * @param nntrainer::IniWrapper::Sections ini data
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 * @param options options
 */
std::tuple<const nntrainer::IniWrapper, const nntrainer::TensorDim,
           const unsigned int, ModelTestOption>
mkModelTc(const nntrainer::IniWrapper &ini, const std::string &label_dim,
          const unsigned int iteration,
          ModelTestOption options = ModelTestOption::ALL);

#endif // __MODELS_GOLDEN_TEST_H__
