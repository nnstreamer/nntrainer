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

#include <functional>
#include <ini_wrapper.h>
#include <tensor_dim.h>

inline constexpr const char *DIM_UNUSED = "1:1:1";
namespace nntrainer {
class NeuralNetwork;
}

/**
 * @brief Test Option for the unittest models
 *
 */
typedef enum {
  NO_THROW_RUN =
    1 << 0, /**< no comparison, only validate execution without throw */
  COMPARE_RUN = 1 << 1,       /**< Set this to compare the numbers */
  SAVE_AND_LOAD_INI = 1 << 2, /**< Set this to check if saving and constructing
                                 a new model works okay (without weights) */
  USE_V2 = 1 << 3,            /**< use v2 model format */
  COMPARE = COMPARE_RUN | NO_THROW_RUN, /**< Set this to comp are the numbers */

  COMPARE_RUN_V2 = COMPARE_RUN | USE_V2,         /**< compare run v2 */
  NO_THROW_RUN_V2 = NO_THROW_RUN | USE_V2,       /**< no throw run with v2 */
  COMPARE_V2 = COMPARE | USE_V2,                 /**< compare v2 */
  SAVE_AND_LOAD_V2 = SAVE_AND_LOAD_INI | USE_V2, /**< save and load with v2 */

  ALL = COMPARE | SAVE_AND_LOAD_INI, /**< Set every option */
  ALL_V2 = ALL | USE_V2              /**< Set every option with v2 */
} ModelTestOption;

using ModelGoldenTestParamType =
  std::tuple<std::function<std::unique_ptr<nntrainer::NeuralNetwork>()>
             /**<NeuralNetwork Creator */,
             const std::string /**< Golden file name */,
             const nntrainer::TensorDim /**< InputDimension */,
             const unsigned int /**< Number of Iterations */,
             ModelTestOption /**< Options which test to run */>;
/**
 * @brief nntrainerModelTest fixture for parametrized test
 *
 * @param nntrainer::IniWrapper ini data
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 */
class nntrainerModelTest
  : public ::testing::TestWithParam<ModelGoldenTestParamType> {

protected:
  /**
   * @brief Construct a new nntrainer Model Test object
   *
   */
  nntrainerModelTest() :
    iteration(0),
    name(""),
    options(ModelTestOption::NO_THROW_RUN) {}

  /**
   * @brief Destroy the nntrainer Model Test object
   *
   */
  virtual ~nntrainerModelTest() {}

  /**
   * @brief Set the Up object
   *
   */
  virtual void SetUp() {
    auto &param = GetParam();
    nn_creator = std::get<0>(param);

    /// remove the test number after double __
    name = std::get<1>(param);

    label_dim = std::get<2>(param);
    iteration = std::get<3>(param);
    options = std::get<4>(param);
  }

  /**
   * @brief Tear down the test case
   *
   */
  virtual void TearDown() {}

  /**
   * @brief Create a Model object
   *
   * @return std::unique_ptr<nntrainer::NeuralNetwork> created model
   */
  std::unique_ptr<nntrainer::NeuralNetwork> createModel() {
    return nn_creator();
  }

  /**
   * @brief Get the Name object
   *
   * @return std::string name
   */
  std::string getName() { return name; }

  /**
   * @brief Get the number of iteration
   *
   * @return int integer
   */
  int getIteration() { return iteration; };

  /**
   * @brief Get the Label Dim object
   *
   * @return nntrainer::TensorDim label dimension
   */
  nntrainer::TensorDim getLabelDim() { return label_dim; }

  /**
   * @brief query if compare test should be conducted
   *
   * @return bool true if test should be done
   */
  bool shouldCompare() { return options & (ModelTestOption::COMPARE_RUN); }
  /**
   * @brief query if compare test should be conducted
   *
   * @return bool true if test should be done
   */
  bool shouldValidate() { return options & (ModelTestOption::NO_THROW_RUN); }

  /**
   * @brief query if saveload ini test should be done
   *
   * @return bool true if test should be done
   */
  bool shouldSaveLoadIniTest() {
    return options & ModelTestOption::SAVE_AND_LOAD_INI;
  }

  /**
   * @brief compare for the value
   *
   * @param opt set if compare while optimized
   * @param creator creator to use instead of default creator
   */
  void compare(bool opt,
               std::function<std::unique_ptr<nntrainer::NeuralNetwork>()>
                 creator = nullptr);

  /**
   * @brief validate for the value
   *
   * @param opt set if validate while optimized
   * @param creator creator to use instead of default creator
   */
  void validate(bool opt,
                std::function<std::unique_ptr<nntrainer::NeuralNetwork>()>
                  creator = nullptr);

private:
  /**
   * @brief Get the Golden Name object
   *
   * @return std::string
   */
  std::string getGoldenName() {
    return name.substr(0, name.find("__")) + ".info";
  }

  /**
   * @brief Get the GoldenName V2 object
   *
   * @return std::string
   */
  std::string getGoldenName_V2() {
    return name.substr(0, name.find("__")) + ".nnmodelgolden";
  }

  std::function<std::unique_ptr<nntrainer::NeuralNetwork>()> nn_creator;
  nntrainer::TensorDim label_dim;
  int iteration;
  std::string name;
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
ModelGoldenTestParamType
mkModelIniTc(const nntrainer::IniWrapper &ini, const std::string &label_dim,
             const unsigned int iteration,
             ModelTestOption options = ModelTestOption::ALL);

/**
 * @brief helper function to generate tcs
 *
 * @param generator generator
 * @param name name
 * @param label_dim label dimenion
 * @param iteration iteration
 * @param options options
 * @return ModelGoldenTestParamType
 */
ModelGoldenTestParamType
mkModelTc(std::function<std::unique_ptr<nntrainer::NeuralNetwork>()> generator,
          const std::string &name, const std::string &label_dim,
          const unsigned int iteration,
          ModelTestOption options = ModelTestOption::ALL);

/**
 * @brief helper function to generate tcs
 *
 * @param generator generator
 * @param name name
 * @param options options
 * @return ModelGoldenTestParamType
 */
ModelGoldenTestParamType mkModelTc_V2(
  std::function<std::unique_ptr<nntrainer::NeuralNetwork>()> generator,
  const std::string &name, ModelTestOption options = ModelTestOption::ALL);

#endif // __MODELS_GOLDEN_TEST_H__
