// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_nntrainer_graph.cpp
 * @date 29 Oct 2020
 * @brief NNTrainer graph test.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <neuralnet.h>

#include "nntrainer_test_util.h"

namespace initest {
typedef enum {
  LOAD = 1 << 0, /**< should fail at load */
  INIT = 1 << 1, /**< should fail at init */
} IniFailAt;
};

class nntrainerGraphTest
  : public ::testing::TestWithParam<
      std::tuple<const char *, const IniTestWrapper::Sections, int>> {

protected:
  virtual void SetUp() {
    name = std::string(std::get<0>(GetParam()));
    std::cout << "starting test case : " << name << std::endl << std::endl;

    auto sections = std::get<1>(GetParam());

    ini = IniTestWrapper(name, sections);

    failAt = std::get<2>(GetParam());
    ini.save_ini();
  }

  virtual void TearDown() { ini.erase_ini(); }

  std::string getIniName() { return ini.getIniName(); }

  bool failAtLoad() { return failAt & initest::IniFailAt::LOAD; }

  bool failAtInit() { return failAt & initest::IniFailAt::INIT; }

  nntrainer::NeuralNetwork NN;

private:
  int failAt;
  std::string name;
  IniTestWrapper ini;
};

/**
 * @brief check given ini is failing/suceeding at load
 */
TEST_P(nntrainerGraphTest, loadConfig) {
  std::cout << std::get<0>(GetParam()) << std::endl;
  int status = NN.loadFromConfig(getIniName());

  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }
}

static IniSection nw_base("model", "Type = NeuralNetwork | "
                                   "batch_size = 32 | "
                                   "epsilon = 1e-7 | "
                                   "loss = cross");

static IniSection nw_sgd = nw_base + "Optimizer = sgd |"
                                     "Learning_rate = 1";

static IniSection input0("inputlayer0", "Type = input |"
                                        "Input_Shape = 1:1:62720 |"
                                        "bias_initializer = zeros |"
                                        "Normalization = true |"
                                        "Activation = sigmoid");

static IniSection input1("inputlayer1", "Type = input |"
                                        "Input_Shape = 1:1:62720 |"
                                        "bias_initializer = zeros |"
                                        "Normalization = true |"
                                        "Activation = sigmoid");

static IniSection conv2d("conv2d", "Type = conv2d |"
                                   "input_layers=inputlayer0, inputlayer1 |"
                                   "bias_initializer = zeros |"
                                   "Activation = sigmoid |"
                                   "filters = 6 |"
                                   "kernel_size = 5,5 |"
                                   "stride = 1,1 |"
                                   "padding = 0,0 |");

static int SUCCESS = 0;

using I = IniSection;

/**
 * @brief make ini test case from given parameter
 */
std::tuple<const char *, const IniTestWrapper::Sections, int>
mkIniTc(const char *name, const IniTestWrapper::Sections vec, int flag) {
  return std::make_tuple(name, vec, flag);
}

INSTANTIATE_TEST_CASE_P(
  nntrainerIniAutoTests, nntrainerGraphTest,
  ::testing::Values(mkIniTc("basic_p", {nw_sgd, input0, input1, conv2d},
                            SUCCESS)));

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  return result;
}
