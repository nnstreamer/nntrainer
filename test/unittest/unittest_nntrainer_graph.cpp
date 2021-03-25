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

  int batch = 16;
  int channel = 3;
  int height = 32;
  int width = 32;

  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.compile();

  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.initialize();
  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.allocate();
  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  NN.forwarding({MAKE_SHARED_TENSOR(input)});

  nntrainer::Tensor output(batch, 1, 1, 10);

  output.setZero();

  for (int i = 0; i < batch; ++i)
    output.setValue(i, 0, 0, 3, 1.0);

  NN.forwarding({MAKE_SHARED_TENSOR(input)}, {MAKE_SHARED_TENSOR(output)});
  NN.backwarding({MAKE_SHARED_TENSOR(output)}, 1);
}

static IniSection nw_base("model", "Type = NeuralNetwork | "
                                   "batch_size = 16 | "
                                   "epsilon = 1e-7 | "
                                   "loss = cross");

static IniSection nw_sgd = nw_base + "Optimizer = sgd |"
                                     "Learning_rate = 1";

static IniSection input("inputlayer", "Type = input |"
                                      "Input_Shape = 3:32:32");

static IniSection conv2d8("conv2d8", "Type = conv2d |"
                                     "input_layers=inputlayer |"
                                     "bias_initializer = zeros |"
                                     "Activation = relu |"
                                     "filters = 32 |"
                                     "kernel_size = 3,3 |"
                                     "stride = 1,1 |"
                                     "padding = 0,0");

static IniSection conv2d9("conv2d9", "Type = conv2d |"
                                     "input_layers=conv2d8 |"
                                     "bias_initializer = zeros |"
                                     "Activation = relu |"
                                     "filters = 64 |"
                                     "kernel_size = 3,3 |"
                                     "stride = 1,1 |"
                                     "padding = 0,0");

static IniSection pooling2("pooling2", "Type = pooling2d |"
                                       "input_layers = conv2d9 |"
                                       "pool_size = 3, 3 |"
                                       "stride = 3, 3 |"
                                       "padding = 0, 0 |"
                                       "pooling=max");

static IniSection out0("out0", "Type = output |"
                               "input_layers = pooling2");

static IniSection conv2d10("conv2d10", "Type = conv2d |"
                                       "input_layers=out0 |"
                                       "bias_initializer = zeros |"
                                       "Activation = relu |"
                                       "filters = 64 |"
                                       "kernel_size = 3,3 |"
                                       "stride = 1,1 |"
                                       "padding = 1,1");

static IniSection conv2d11("conv2d11", "Type = conv2d |"
                                       "input_layers=conv2d10 |"
                                       "bias_initializer = zeros |"
                                       "Activation = relu |"
                                       "filters = 64 |"
                                       "kernel_size = 3,3 |"
                                       "stride = 1,1 |"
                                       "padding = 1,1");

static IniSection addition0("addition0", "Type=addition |"
                                         "input_layers = conv2d11, out0 ");

static IniSection out1("out1", "Type = output |"
                               "input_layers = addition0");

static IniSection conv2d12("conv2d12", "Type = conv2d |"
                                       "input_layers=out1 |"
                                       "bias_initializer = zeros |"
                                       "Activation = relu |"
                                       "filters = 64 |"
                                       "kernel_size = 3,3 |"
                                       "stride = 1,1 |"
                                       "padding = 1,1");

static IniSection conv2d13("conv2d13", "Type = conv2d |"
                                       "input_layers=conv2d12 |"
                                       "bias_initializer = zeros |"
                                       "Activation = relu |"
                                       "filters = 64 |"
                                       "kernel_size = 3,3 |"
                                       "stride = 1,1 |"
                                       "padding = 1,1");

static IniSection addition1("addition1", "Type=addition |"
                                         "input_layers = conv2d13, out1 ");

static IniSection conv2d14("conv2d14", "Type = conv2d |"
                                       "input_layers=addition1 |"
                                       "bias_initializer = zeros |"
                                       "Activation = relu |"
                                       "filters = 64 |"
                                       "kernel_size = 3,3 |"
                                       "stride = 1,1 |"
                                       "padding = 0,0");

static IniSection pooling3("pooling3",
                           "Type = pooling2d |"
                           "input_layers = conv2d14 |"
                           "pooling=global_average | flatten = true");

static IniSection fclayer0("fclayer0", "Type = fully_connected |"
                                       "Unit = 256 |"
                                       "input_layers = pooling3 |"
                                       "bias_initializer = zeros |"
                                       "Activation = relu");

static IniSection fclayer1("fclayer1", "Type = fully_connected |"
                                       "Unit = 10 |"
                                       "input_layers = fclayer0 |"
                                       "bias_initializer = zeros |"
                                       "Activation = softmax");

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
  ::testing::Values(mkIniTc("basic_p",
                            {nw_sgd, input, conv2d8, conv2d9, pooling2, out0,
                             conv2d10, conv2d11, addition0, out1, conv2d12,
                             conv2d13, addition1, conv2d14, pooling3, fclayer0,
                             fclayer1},
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
