// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_nntrainer_modelfile.cpp
 * @date 16 July 2020
 * @brief NNTrainer datafile parmeterized tester.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include "nntrainer_test_util.h"
#include <gtest/gtest.h>
#include <neuralnet.h>

/**
 * @brief check given ini is failing/suceeding at load
 */
TEST_P(nntrainerIniTest, loadConfig) {
  std::cout << std::get<0>(GetParam()) << std::endl;
  int status = NN.loadFromConfig(getIniName());

  if (failAtLoad()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }
}

/**
 * @brief Negative test given ini is failing at loadingTwice
 */
TEST_P(nntrainerIniTest, loadConfigTwice_n) {
  std::cout << std::get<0>(GetParam()) << std::endl;
  NN.loadFromConfig(getIniName());
  int status = NN.loadFromConfig(getIniName());
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief check given ini is failing/succeeding at init
 */
TEST_P(nntrainerIniTest, init) {
  std::cout << std::get<0>(GetParam()) << std::endl;
  int status = NN.loadFromConfig(getIniName());

  status = NN.init();

  if (failAtInit()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }
}

/**
 * @brief check given ini is failing/succeeding when init happens twice.
 * this should fail at all time.
 */
TEST_P(nntrainerIniTest, initTwice_n) {
  std::cout << std::get<0>(GetParam()) << std::endl;
  int status = NN.loadFromConfig(getIniName());
  status = NN.init();
  status = NN.init();

  EXPECT_NE(status, ML_ERROR_NONE);
}

/**
 * @brief check given ini is failing/succeeding when init happens three times.
 * this should fail at all time.
 */
TEST_P(nntrainerIniTest, initThreetime_n) {
  std::cout << std::get<0>(GetParam()) << std::endl;
  int status = NN.loadFromConfig(getIniName());
  status = NN.init();
  status = NN.init();
  status = NN.init();

  EXPECT_NE(status, ML_ERROR_NONE);
}

/// @todo add run test could be added with iniTest flag to control skip
static IniSection nw_base("model", "Type = NeuralNetwork | "
                                   "batch_size = 32 | "
                                   "epsilon = 1e-7 | "
                                   "cost = cross");
static IniSection adam("adam", "Optimizer = adam |"
                               "Learning_rate = 0.00001 |"
                               "Decay_rate = 0.96 |"
                               "Decay_steps = 1000");

static IniSection nw_sgd = nw_base + "Optimizer = sgd |"
                                     "Learning_rate = 1";

static IniSection nw_adam = nw_base + adam;

static IniSection nw_adam_n = nw_base + "Learning_rate = -1";

static IniSection dataset("DataSet", "BufferSize = 100 |"
                                     "TrainData = trainingSet.dat | "
                                     "TestData = testSet.dat |"
                                     "ValidData = valSet.dat |"
                                     "LabelData = label.dat");

static IniSection batch_normal("bn", "Type = batch_normalization");

static IniSection flatten("flat", "Type = flatten");

static IniSection input("inputlayer", "Type = input |"
                                      "Input_Shape = 1:1:62720 |"
                                      "bias_initializer = zeros |"
                                      "Normalization = true |"
                                      "Activation = sigmoid");

static IniSection act_relu("activation_relu", "Type = activation | "
                                              "Activation = relu");

static IniSection out("fclayer", "Type = fully_connected |"
                                 "Unit = 10 |"
                                 "bias_initializer = zeros |"
                                 "Activation = softmax");

static IniSection conv2d("conv2d", "Type = conv2d |"
                                   "bias_initializer = zeros |"
                                   "Activation = sigmoid |"
                                   "filters = 6 |"
                                   "kernel_size = 5,5 |"
                                   "stride = 1,1 |"
                                   "padding = 0,0 |");

static int SUCCESS = 0;
static int LOADFAIL = initest::LOAD;
static int INITFAIL = initest::INIT;
static int ALLFAIL = LOADFAIL | INITFAIL;

using I = IniSection;

/// @note each line contains 2 (positive or negative test) + 3 negative test.
/// if, there are 6 positive tests and 9 negative tests
/// which sums up to 6 * 2 = 12 positive tests and 9 * 2 + (6 + 9) * 3 = 63
/// negative tests
// clang-format off
INSTANTIATE_TEST_CASE_P(
  nntrainerIniAutoTests, nntrainerIniTest, ::testing::Values(
  /**< positive: basic valid scenarios (2 positive and 3 negative cases) */
    mkIniTc("basic_p", {nw_adam, input, out}, SUCCESS),
    mkIniTc("basic2_p", {nw_sgd, input, out}, SUCCESS),
    mkIniTc("basic_act_p", {nw_sgd, input + "-Activation", act_relu, out }, SUCCESS),
    mkIniTc("basic_bn_p", {nw_sgd, input + "-Activation", batch_normal, act_relu, out }, SUCCESS),
    mkIniTc("basic_bn2_p", {nw_sgd, input + "-Activation", batch_normal + "Activation = relu", out }, SUCCESS),
    mkIniTc("basic_dataset_p", {nw_adam, dataset, input, out}, SUCCESS),
    mkIniTc("basic_dataset2_p", {nw_sgd, input, out, dataset}, SUCCESS),
    mkIniTc("basic_dataset3_p", {dataset, nw_sgd, input, out}, SUCCESS),
    mkIniTc("basic_conv2d_p", {nw_adam, conv2d + "input_shape = 1:1:62720"}, SUCCESS),
    mkIniTc("no_testSet_p", {nw_adam, dataset + "-TestData", input, out}, SUCCESS),
    mkIniTc("no_validSet_p", {nw_adam, dataset + "-ValidData", input, out}, SUCCESS),
    mkIniTc("no_bufferSize_p", {nw_adam, dataset + "-BufferSize", input, out}, SUCCESS),


  /**< half negative: init fail cases (1 positive and 4 negative cases) */
    mkIniTc("unknown_cost_n", {nw_adam + "cost = unknown", input, out}, INITFAIL),
    mkIniTc("activation_very_first_n", {nw_sgd, act_relu, input, out}, INITFAIL),
    mkIniTc("bnlayer_very_first_n", {nw_sgd, batch_normal, input, out}, INITFAIL),
    mkIniTc("act_layer_after_act_n", {nw_sgd, input, act_relu, out}, INITFAIL),
    mkIniTc("act_layer_after_act_bn_n", {nw_sgd, input, act_relu, batch_normal, out }, INITFAIL),
    mkIniTc("last_act_layer_relu_n", {nw_sgd, input, out, act_relu }, INITFAIL),
    mkIniTc("last_act_layer_relu2_n", {nw_sgd, input, out + "-Activation", act_relu }, INITFAIL),

  /**< negative: basic invalid scenarios (5 negative cases) */
    mkIniTc("no_model_sec_name_n", {I(nw_adam, "-", "")}, ALLFAIL),
    mkIniTc("no_model_sec_n", {input, out}, ALLFAIL),
    mkIniTc("empty_n", {}, ALLFAIL),
    mkIniTc("no_layers_n", {nw_adam}, ALLFAIL),
    mkIniTc("no_layers_2_n", {nw_adam, dataset}, ALLFAIL),
    /// #391
    // mkIniTc("ini_has_empty_value_n", {nw_adam + "epsilon = _", input, out}, ALLFAIL),

  /**< negative: property(hyperparam) validation (5 negative cases) */
    mkIniTc("wrong_opt_type_n", {nw_adam + "Optimizer = wrong_opt", input, out}, ALLFAIL),
    mkIniTc("adam_minus_lr_n", {nw_adam + "Learning_rate = -0.1", input, out}, ALLFAIL),
    mkIniTc("sgd_minus_lr_n", {nw_sgd + "Learning_rate = -0.1", input, out}, ALLFAIL),
    mkIniTc("no_cost_n", {nw_adam + "-cost", input, out}, INITFAIL),
    // #389
    // mkIniTc("buffer_size_smaller_than_batch_size_n", {nw_adam, dataset + "BufferSize=26", input, out}, ALLFAIL),
    mkIniTc("unknown_layer_type_n", {nw_adam, input + "Type = asdf", out}, ALLFAIL),
    mkIniTc("unknown_layer_type2_n", {nw_adam, input, out + "Type = asdf", I(out, "outlayer", "")}, ALLFAIL),

  /**< negative: little bit of tweeks to check determinancy (5 negative cases) */
    mkIniTc("wrong_nw_dataset_n", {nw_adam, input, out, dataset + "-LabelData"}, ALLFAIL),
    mkIniTc("wrong_nw_dataset2_n", {nw_adam, dataset + "-LabelData", input, out}, ALLFAIL),
    // #389
    // mkIniTc("buffer_size_smaller_than_batch_size2_n", {nw_adam, input, out, dataset + "BufferSize=26"}, ALLFAIL),

  /**< negative: dataset is not complete (5 negative cases) */
    mkIniTc("no_trainingSet_n", {nw_adam, dataset + "-TrainData", input, out}, ALLFAIL),
    mkIniTc("no_labelSet_n", {nw_adam, dataset + "-LabelData", input, out}, ALLFAIL)
/// #if gtest_version <= 1.7.0
));
/// #else gtest_version > 1.8.0
// ), [](const testing::TestParamInfo<nntrainerIniTest::ParamType>& info){
//  return std::get<0>(info.param);
// });
/// #end if */
// clang-format on

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

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  return result;
}
