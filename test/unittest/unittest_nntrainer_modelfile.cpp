// SPDX-License-Identifier: Apache-2.0
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

#include <gtest/gtest.h>

#include <neuralnet.h>
#include <nntrainer-api-common.h>
#include <nntrainer_internal.h>

#include <nntrainer_test_util.h>

namespace initest {
typedef enum {
  LOAD = 1 << 0, /**< should fail at load */
  COMP = 1 << 1,
  INIT = 1 << 2, /**< should fail at init */
} IniFailAt;
};

class nntrainerIniTest
  : public ::testing::TestWithParam<
      std::tuple<const char *, const IniTestWrapper::Sections, int>> {

public:
  static void save_ini(const char *filename, std::vector<IniSection> sections,
                       std::ios_base::openmode mode = std::ios_base::out) {
    IniTestWrapper::save_ini(filename, sections, mode);
  }

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

  bool failAtComp() { return failAt & initest::IniFailAt::COMP; }

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

  status = NN.compile();

  if (failAtComp()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.initialize();

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

  status = NN.compile();

  if (failAtComp()) {
    EXPECT_NE(status, ML_ERROR_NONE);
  } else {
    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  status = NN.initialize();
  status = NN.initialize();

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
                                   "loss = cross");

static IniSection nw_base_mse("model", "Type = NeuralNetwork | "
                                       "batch_size = 32 | "
                                       "epsilon = 1e-7 | "
                                       "loss = mse");

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

static IniSection batch_normal("bn", "Type = batch_normalization |"
                                     "momentum = 1.2 |"
                                     "moving_mean_initializer = zeros |"
                                     "moving_variance_initializer = ones |"
                                     "gamma_initializer = zeros |"
                                     "beta_initializer = ones");

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

static IniSection conv2d_shape("conv2d_shape", "Type = conv2d |"
                                               "input_shape = 3:300:300 |"
                                               "bias_initializer = zeros |"
                                               "Activation = sigmoid |"
                                               "filters = 6 |"
                                               "kernel_size = 5,5 |"
                                               "stride = 1,1 |"
                                               "padding = 0,0 |");

static IniSection input2d("inputlayer", "Type = input |"
                                        "Input_Shape = 3:100:100");

static IniSection backbone_random("block1", "backbone = random.ini");

static IniSection backbone_valid("block1", "backbone = base.ini");

static IniSection backbone_notrain("blockNT", "backbone = base.ini |"
                                              "trainable = false");

static IniSection backbone_train("blockT", "backbone = base.ini |"
                                           "trainable = true");

static IniSection backbone_scaled("blockT", "backbone = base.ini |"
                                            "ScaleSize = 0.5");

static IniSection backbone_scaled_zero("blockT", "backbone = base.ini |"
                                                 "ScaleSize = 0.00005");

static IniSection backbone_random_external("block1",
                                           "backbone = random.tflite");

static IniSection backbone_valid_inout("block1", "backbone = base.ini |"
                                                 "InputLayer = flat | "
                                                 "OutputLayer = fclayer");

static IniSection
  backbone_valid_external("block1",
                          "backbone = ../test/test_models/models/add.tflite | "
                          "Input_Shape = 1:1:1");

static IniSection backbone_valid_external_no_shape(
  "block1", "backbone = ../test/test_models/models/add.tflite");

static int SUCCESS = 0;
static int LOADFAIL = initest::LOAD;
static int COMPFAIL = initest::COMP;
static int INITFAIL = initest::INIT;
static int ALLFAIL = LOADFAIL | INITFAIL | COMPFAIL;

using I = IniSection;

/**
 * @brief make ini test case from given parameter
 */
std::tuple<const char *, const IniTestWrapper::Sections, int>
mkIniTc(const char *name, const IniTestWrapper::Sections vec, int flag) {
  return std::make_tuple(name, vec, flag);
}

/// @note each line contains 2 (positive or negative test) + 3 negative test.
/// if, there are 6 positive tests and 9 negative tests
/// which sums up to 6 * 2 = 12 positive tests and 9 * 2 + (6 + 9) * 3 = 63
/// negative tests
// clang-format off
INSTANTIATE_TEST_CASE_P(
  nntrainerIniAutoTests, nntrainerIniTest, ::testing::Values(
  /**< positive: basic valid scenarios (2 positive and 3 negative cases) */
    mkIniTc("basic_p", {nw_adam, input, out+"input_layers=inputlayer"}, SUCCESS),
    mkIniTc("basic2_p", {nw_sgd, input, out+"input_layers=inputlayer"}, SUCCESS),
    mkIniTc("basic_act_p", {nw_sgd, input + "-Activation", act_relu+"input_layers=inputlayer", out+"input_layers=activation_relu" }, SUCCESS),
    mkIniTc("basic_bn_p", {nw_sgd, input + "-Activation", batch_normal+"input_layers=inputlayer", act_relu+"input_layers=bn", out+"input_layers=activation_relu" }, SUCCESS),
    mkIniTc("basic_bn2_p", {nw_sgd, input + "-Activation", batch_normal + "Activation = relu"+"input_layers=inputlayer", out+"input_layers=bn" }, SUCCESS),
    mkIniTc("basic_dataset_p", {nw_adam, dataset, input, out+"input_layers=inputlayer"}, SUCCESS),
    mkIniTc("basic_dataset2_p", {nw_sgd, input, out+"input_layers=inputlayer", dataset}, SUCCESS),
    mkIniTc("basic_dataset3_p", {dataset, nw_sgd, input, out+"input_layers=inputlayer"}, SUCCESS),
    mkIniTc("basic_conv2d_p", {nw_adam, conv2d + "input_shape = 1:1:62720"}, SUCCESS),
    mkIniTc("no_testSet_p", {nw_adam, dataset + "-TestData", input, out+"input_layers=inputlayer"}, SUCCESS),
    mkIniTc("no_validSet_p", {nw_adam, dataset + "-ValidData", input, out+"input_layers=inputlayer"}, SUCCESS),
    mkIniTc("no_bufferSize_p", {nw_adam, dataset + "-BufferSize", input, out+"input_layers=inputlayer"}, SUCCESS),
    mkIniTc("buffer_size_smaller_than_batch_size_p", {nw_adam, dataset + "BufferSize=26", input, out+"input_layers=inputlayer"}, SUCCESS),
    mkIniTc("buffer_size_smaller_than_batch_size2_p", {nw_adam, input, out+"input_layers=inputlayer", dataset + "BufferSize=26"}, SUCCESS),

  /**< half negative: init fail cases (1 positive and 4 negative cases) */
    mkIniTc("unknown_loss_n", {nw_adam + "loss = unknown", input, out+"input_layers=inputlayer"}, COMPFAIL | INITFAIL),
    mkIniTc("activation_very_first_n", {nw_sgd, act_relu, input+"input_layers=activation_relu", out+"input_layers=inputlayer"}, COMPFAIL| INITFAIL),
    mkIniTc("bnlayer_very_first_n", {nw_sgd, batch_normal, input+"input_layers=bn", out+"input_layers=inputlayer"}, COMPFAIL | INITFAIL),
    mkIniTc("act_layer_after_act_n", {nw_sgd, input, act_relu+"input_layers=inputlayer", out+"input_layers=activation_relu"}, INITFAIL),
    mkIniTc("act_layer_after_act_bn_n", {nw_sgd, input, act_relu+"input_layers=inputlayer", batch_normal+"input_layers=activation_relu", out+"input_layers=bn" }, INITFAIL),
    mkIniTc("last_act_layer_relu_n", {nw_sgd, input, out+"input_layers=inputlayer", act_relu+"input_layers=fclayer" }, COMPFAIL | INITFAIL),
    mkIniTc("last_act_layer_relu2_n", {nw_sgd, input, out+"input_layers=inputlayer" + "-Activation", act_relu+"input_layers=fclayer" }, COMPFAIL | INITFAIL),

  /**< negative: basic invalid scenarios (5 negative cases) */
    mkIniTc("no_model_sec_name_n", {I(nw_adam, "-", "")}, ALLFAIL),
    mkIniTc("no_model_sec_n", {input, out+"input_layers=inputlayer"}, ALLFAIL),
    mkIniTc("empty_n", {}, ALLFAIL),
    mkIniTc("no_layers_n", {nw_adam}, ALLFAIL),
    mkIniTc("no_layers_2_n", {nw_adam, dataset}, ALLFAIL),
    /// #391
    // mkIniTc("ini_has_empty_value_n", {nw_adam + "epsilon = _", input, out}, ALLFAIL),

  /**< negative: property(hyperparam) validation (5 negative cases) */
    mkIniTc("wrong_opt_type_n", {nw_adam + "Optimizer = wrong_opt", input, out+"input_layers=inputlayer"}, ALLFAIL),
    mkIniTc("adam_minus_lr_n", {nw_adam + "Learning_rate = -0.1", input, out+"input_layers=inputlayer"}, ALLFAIL),
    mkIniTc("sgd_minus_lr_n", {nw_sgd + "Learning_rate = -0.1", input, out+"input_layers=inputlayer"}, ALLFAIL),
    mkIniTc("no_loss_n", {nw_adam + "-loss", input, out+"input_layers=inputlayer"}, COMPFAIL | INITFAIL),
    mkIniTc("unknown_layer_type_n", {nw_adam, input + "Type = asdf", out+"input_layers=inputlayer"}, ALLFAIL),
    mkIniTc("unknown_layer_type2_n", {nw_adam, input, out + "Type = asdf"+"input_layers=inputlayer", I(out, "outlayer", "")}, ALLFAIL),

  /**< negative: little bit of tweeks to check determinancy (5 negative cases) */
    mkIniTc("wrong_nw_dataset_n", {nw_adam, input, out+"input_layers=inputlayer", dataset + "-LabelData"}, ALLFAIL),
    mkIniTc("wrong_nw_dataset2_n", {nw_adam, dataset + "-LabelData", input, out+"input_layers=inputlayer"}, ALLFAIL),

  /**< negative: dataset is not complete (5 negative cases) */
    mkIniTc("no_trainingSet_n", {nw_adam, dataset + "-TrainData", input, out+"input_layers=inputlayer"}, ALLFAIL),
    mkIniTc("no_labelSet_n", {nw_adam, dataset + "-LabelData", input, out+"input_layers=inputlayer"}, ALLFAIL),

    mkIniTc("backbone_filemissing_n", {nw_adam, dataset + "-LabelData", input, out+"input_layers=inputlayer"}, ALLFAIL)
/// #if gtest_version <= 1.7.0
));
/// #else gtest_version > 1.8.0
// ), [](const testing::TestParamInfo<nntrainerIniTest::ParamType>& info){
//  return std::get<0>(info.param);
// });
/// #end if */
// clang-format on

/**
 * @brief Ini file unittest with backbone with wrong file
 */
TEST(nntrainerIniTest, backbone_n_01) {
  const char *ini_name = "backbone_n1.ini";
  nntrainerIniTest::save_ini(ini_name, {nw_base, backbone_random});
  nntrainer::NeuralNetwork NN;

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Ini file unittest with backbone with empty backbone
 */
TEST(nntrainerIniTest, backbone_n_02) {
  const char *ini_name = "backbone_n2.ini";
  nntrainerIniTest::save_ini("base.ini", {nw_base});
  nntrainerIniTest::save_ini(ini_name, {nw_base, backbone_valid});
  nntrainer::NeuralNetwork NN;

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Ini file unittest with backbone with normal backbone
 */
TEST(nntrainerIniTest, backbone_p_03) {
  const char *ini_name = "backbone_p3.ini";
  nntrainerIniTest::save_ini("base.ini", {nw_base, batch_normal});
  nntrainerIniTest::save_ini(ini_name, {nw_base, backbone_valid});
  nntrainer::NeuralNetwork NN;

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
}

/**
 * @brief Ini file unittest with backbone without model parameters
 */
TEST(nntrainerIniTest, backbone_p_04) {
  const char *ini_name = "backbone_p4.ini";
  nntrainerIniTest::save_ini("base.ini", {flatten, conv2d});
  nntrainerIniTest::save_ini(ini_name, {nw_base, backbone_valid});
  nntrainer::NeuralNetwork NN;

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
}

/**
 * @brief Ini file unittest matching model with and without backbone
 */
TEST(nntrainerIniTest, backbone_p_05) {
  const char *bb_use_ini_name = "backbone_made.ini";
  const char *direct_ini_name = "direct_made.ini";

  /** Create a backbone.ini */
  nntrainerIniTest::save_ini("base.ini", {nw_adam, conv2d});

  /** Create a model of 4 conv layers using backbone */
  std::string backbone_valid_orig_name = backbone_valid.getName();

  nntrainerIniTest::save_ini(
    bb_use_ini_name,
    {nw_sgd, input2d, backbone_valid + "input_layers=inputlayer"});
  backbone_valid.rename("block2");
  nntrainerIniTest::save_ini(bb_use_ini_name,
                             {backbone_valid + "input_layers=block1"},
                             std::ios_base::app);
  backbone_valid.rename("block3");
  nntrainerIniTest::save_ini(bb_use_ini_name,
                             {backbone_valid + "input_layers=block2"},
                             std::ios_base::app);
  backbone_valid.rename("block4");
  nntrainerIniTest::save_ini(bb_use_ini_name,
                             {backbone_valid + "input_layers=block3"},
                             std::ios_base::app);

  backbone_valid.rename(backbone_valid_orig_name);

  nntrainer::NeuralNetwork NN_backbone;
  EXPECT_EQ(NN_backbone.loadFromConfig(bb_use_ini_name), ML_ERROR_NONE);
  EXPECT_EQ(NN_backbone.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_backbone.initialize(), ML_ERROR_NONE);

  /**
   * Model defined in backbone with adam with lr 0.0001 does not affect the
   * final model to be made using the backbone.
   */
  EXPECT_EQ(NN_backbone.getLearningRate(), 1);

  /** Create the same model directly without using backbone */
  std::string conv2d_orig_name = conv2d.getName();

  nntrainerIniTest::save_ini(direct_ini_name, {nw_sgd, input2d});
  conv2d.rename("block1conv2d");
  nntrainerIniTest::save_ini(
    direct_ini_name, {conv2d + "input_layers=inputlayer"}, std::ios_base::app);
  conv2d.rename("block2conv2d");
  nntrainerIniTest::save_ini(direct_ini_name,
                             {conv2d + "input_layers=block1conv2d"},
                             std::ios_base::app);
  conv2d.rename("block3conv2d");
  nntrainerIniTest::save_ini(direct_ini_name,
                             {conv2d + "input_layers=block2conv2d"},
                             std::ios_base::app);
  conv2d.rename("block4conv2d");
  nntrainerIniTest::save_ini(direct_ini_name,
                             {conv2d + "input_layers=block3conv2d"},
                             std::ios_base::app);

  conv2d.rename(conv2d_orig_name);

  nntrainer::NeuralNetwork NN_direct;
  EXPECT_EQ(NN_direct.loadFromConfig(direct_ini_name), ML_ERROR_NONE);
  EXPECT_EQ(NN_direct.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_direct.initialize(), ML_ERROR_NONE);

  /** Summary of both the models must match precisely */
  NN_backbone.printPreset(std::cout, ML_TRAIN_SUMMARY_MODEL);
  NN_direct.printPreset(std::cout, ML_TRAIN_SUMMARY_MODEL);

  EXPECT_EQ(NN_backbone.getInputDimension(), NN_direct.getInputDimension());
  EXPECT_EQ(NN_backbone.getOutputDimension(), NN_direct.getOutputDimension());

  auto flat_backbone = NN_backbone.getFlatGraph();
  auto flat_direct = NN_direct.getFlatGraph();
  EXPECT_EQ(flat_backbone.size(), flat_direct.size());

  for (size_t idx = 0; idx < flat_backbone.size(); idx++) {
    EXPECT_EQ(flat_backbone[idx]->getType(), flat_direct[idx]->getType());
    EXPECT_EQ(flat_backbone[idx]->getInputDimension(),
              flat_direct[idx]->getInputDimension());
    EXPECT_EQ(flat_backbone[idx]->getOutputDimension(),
              flat_direct[idx]->getOutputDimension());
    EXPECT_EQ(flat_backbone[idx]->getActivationType(),
              flat_direct[idx]->getActivationType());
    EXPECT_EQ(flat_backbone[idx]->getName(), flat_direct[idx]->getName());
  }
}

/**
 * @brief Ini file unittest matching model with and without trainable
 */
TEST(nntrainerIniTest, backbone_p_06) {
  const char *ini_name = "backbone_p6.ini";
  nntrainerIniTest::save_ini("base.ini", {flatten, conv2d});
  nntrainerIniTest::save_ini(ini_name, {nw_base, backbone_valid});
  nntrainer::NeuralNetwork NN;

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);

  /** default trainable is false */
  auto graph = NN.getFlatGraph();
  for (auto &layer : graph)
    EXPECT_EQ(layer->getTrainable(), false);
}

/**
 * @brief Ini file unittest matching model with and without trainable
 */
TEST(nntrainerIniTest, backbone_p_07) {
  const char *ini_name = "backbone_p7.ini";
  nntrainerIniTest::save_ini("base.ini", {conv2d});
  nntrainerIniTest::save_ini(ini_name,
                             {nw_base, backbone_notrain, backbone_train});
  nntrainer::NeuralNetwork NN;

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);

  /** trainable is set to false */
  auto graph = NN.getFlatGraph();
  EXPECT_EQ(graph[0]->getTrainable(), false);
  EXPECT_EQ(graph[1]->getTrainable(), true);
}

/**
 * @brief Ini file unittest with backbone with normal backbone
 */
TEST(nntrainerIniTest, backbone_n_08) {
  const char *ini_name = "backbone_n8.ini";
  nntrainerIniTest::save_ini(ini_name, {nw_base, backbone_random_external});
  nntrainer::NeuralNetwork NN;

#if defined(ENABLE_NNSTREAMER_BACKBONE) || defined(ENABLE_TFLITE_BACKBONE)
  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
  EXPECT_THROW(NN.compile(), std::invalid_argument);
  EXPECT_EQ(NN.initialize(), ML_ERROR_NOT_SUPPORTED);
#else
  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NOT_SUPPORTED);
#endif
}

/**
 * @brief Ini file unittest with backbone with normal backbone
 */
TEST(nntrainerIniTest, backbone_p_09) {
  const char *ini_name = "backbone_p9.ini";
  nntrainerIniTest::save_ini(
    ini_name, {nw_base_mse + "-batch_size", backbone_valid_external});
  nntrainer::NeuralNetwork NN;

#if defined(ENABLE_NNSTREAMER_BACKBONE) || defined(ENABLE_TFLITE_BACKBONE)
  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
  EXPECT_EQ(NN.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN.initialize(), ML_ERROR_NONE);
#else
  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NOT_SUPPORTED);
#endif
}

/**
 * @brief Ini file unittest with backbone with normal backbone
 */
// Enable after sepearet memory assign and initialization of graph
TEST(nntrainerIniTest, DISABLED_backbone_p_10) {
  const char *ini_name = "backbone_p10.ini";
  nntrainerIniTest::save_ini(ini_name,
                             {nw_base_mse, backbone_valid_external_no_shape});
  nntrainer::NeuralNetwork NN;

#if defined(ENABLE_NNSTREAMER_BACKBONE) || defined(ENABLE_TFLITE_BACKBONE)
  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
  EXPECT_EQ(NN.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN.initialize(), ML_ERROR_NONE);
#else
  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NOT_SUPPORTED);
#endif
}

/**
 * @brief Ini file unittest with backbone
 * @note Input shape scaled verified for conv layer, and not for activation/bn
 * layers
 */
TEST(nntrainerIniTest, backbone_p_11) {
  const char *ini_name_scaled_half = "backbone_p11_scaled_half.ini";
  const char *ini_name_full = "backbone_p11_full.ini";
  nntrainerIniTest::save_ini(
    "base.ini", {conv2d, batch_normal + "input_layers=conv2d", conv2d});
  nntrainerIniTest::save_ini(
    ini_name_scaled_half,
    {nw_base_mse, input2d, backbone_scaled + "input_layers=inputlayer"});
  nntrainerIniTest::save_ini(
    ini_name_full,
    {nw_base_mse, input2d, backbone_valid + "input_layers=inputlayer"});
  nntrainer::NeuralNetwork NN_scaled_half, NN_full;

  EXPECT_EQ(NN_full.loadFromConfig(ini_name_full), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN_scaled_half.loadFromConfig(ini_name_scaled_half), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled_half.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled_half.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN_full.getInputDimension()[0].channel(),
            NN_scaled_half.getInputDimension()[0].channel());
  EXPECT_EQ(
    (unsigned int)((float)NN_full.getOutputDimension()[0].channel() * 0.5),
    NN_scaled_half.getOutputDimension()[0].channel());
}

/**
 * @brief Ini file unittest with backbone
 * @note Input shape scaled verified for fc layer, and not for activation/bn
 * layers
 */
TEST(nntrainerIniTest, backbone_p_12) {
  const char *ini_name_scaled_half = "backbone_p12_scaled_half.ini";
  const char *ini_name_full = "backbone_p12_full.ini";
  nntrainerIniTest::save_ini("base.ini",
                             {out, batch_normal + "input_layers=fclayer"});
  nntrainerIniTest::save_ini(
    ini_name_scaled_half,
    {nw_base_mse, input, backbone_scaled + "input_layers=inputlayer"});
  nntrainerIniTest::save_ini(
    ini_name_full,
    {nw_base_mse, input, backbone_valid + "input_layers=inputlayer"});
  nntrainer::NeuralNetwork NN_scaled_half, NN_full;

  EXPECT_EQ(NN_full.loadFromConfig(ini_name_full), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN_scaled_half.loadFromConfig(ini_name_scaled_half), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled_half.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled_half.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN_full.getInputDimension()[0].channel(),
            NN_scaled_half.getInputDimension()[0].channel());
  EXPECT_EQ(
    (unsigned int)((float)NN_full.getOutputDimension()[0].width() * 0.5),
    NN_scaled_half.getOutputDimension()[0].width());
}

/**
 * @brief Ini file unittest with backbone
 * @note Input shape from layers of backbone are striped off
 */
TEST(nntrainerIniTest, backbone_p_13) {
  const char *ini_name_scaled_half = "backbone_p13_scaled_half.ini";
  const char *ini_name_full = "backbone_p13_full.ini";
  nntrainerIniTest::save_ini(
    "base.ini", {conv2d_shape, batch_normal + "input_layers=conv2d_shape",
                 conv2d + "input_layers=bn"});
  nntrainerIniTest::save_ini(
    ini_name_scaled_half,
    {nw_base_mse, input2d, backbone_scaled + "input_layers=inputlayer"});
  nntrainerIniTest::save_ini(
    ini_name_full,
    {nw_base_mse, input2d, backbone_valid + "input_layers=inputlayer"});
  nntrainer::NeuralNetwork NN_scaled_half, NN_full;

  EXPECT_EQ(NN_full.loadFromConfig(ini_name_full), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN_scaled_half.loadFromConfig(ini_name_scaled_half), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled_half.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled_half.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN_full.getInputDimension()[0].channel(),
            NN_scaled_half.getInputDimension()[0].channel());
  EXPECT_EQ(
    (unsigned int)((float)NN_full.getOutputDimension()[0].channel() * 0.5),
    NN_scaled_half.getOutputDimension()[0].channel());
}

/**
 * @brief Ini file unittest with backbone
 * @note Scaled size is at least 1
 */
TEST(nntrainerIniTest, backbone_p_14) {
  const char *ini_name_scaled_zero = "backbone_p14_scaled_zero.ini";
  const char *ini_name_full = "backbone_p14_full.ini";
  nntrainerIniTest::save_ini(
    "base.ini", {conv2d_shape, conv2d + "input_layers=conv2d_shape"});
  nntrainerIniTest::save_ini(
    ini_name_scaled_zero,
    {nw_base_mse, input2d, backbone_scaled_zero + "input_layers=inputlayer"});
  nntrainerIniTest::save_ini(
    ini_name_full,
    {nw_base_mse, input2d, backbone_valid + "input_layers=inputlayer"});
  nntrainer::NeuralNetwork NN_scaled_zero, NN_full;

  EXPECT_EQ(NN_full.loadFromConfig(ini_name_full), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN_scaled_zero.loadFromConfig(ini_name_scaled_zero), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled_zero.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled_zero.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN_full.getInputDimension()[0].channel(),
            NN_scaled_zero.getInputDimension()[0].channel());
  EXPECT_EQ(1, NN_scaled_zero.getOutputDimension()[0].channel());
}

/**
 * @brief Ini file unittest with backbone
 * @note Input shape is provided in model file
 */
TEST(nntrainerIniTest, backbone_n_15) {
  const char *ini_name_scaled = "backbone_n15_scaled.ini";
  const char *ini_name_full = "backbone_n15_full.ini";
  nntrainer::NeuralNetwork NN_scaled, NN_full;

  nntrainerIniTest::save_ini("base.ini", {conv2d, conv2d});

  nntrainerIniTest::save_ini(ini_name_full, {nw_base_mse, backbone_valid});

  EXPECT_EQ(NN_full.loadFromConfig(ini_name_full), ML_ERROR_NONE);
  EXPECT_THROW(NN_full.compile(), std::invalid_argument);
  EXPECT_EQ(NN_full.initialize(), ML_ERROR_NOT_SUPPORTED);

  nntrainerIniTest::save_ini(ini_name_scaled, {nw_base_mse, backbone_scaled});

  EXPECT_EQ(NN_scaled.loadFromConfig(ini_name_scaled), ML_ERROR_NONE);
  EXPECT_THROW(NN_scaled.compile(), std::invalid_argument);
  EXPECT_EQ(NN_scaled.initialize(), ML_ERROR_NOT_SUPPORTED);
}

/**
 * @brief Ini file unittest with backbone
 * @note Input shape is striped from backbone and not provided in model file
 */
TEST(nntrainerIniTest, backbone_n_16) {
  const char *ini_name_scaled = "backbone_n16_scaled.ini";
  const char *ini_name_full = "backbone_n16_full.ini";
  nntrainer::NeuralNetwork NN_scaled, NN_full;

  nntrainerIniTest::save_ini(
    "base.ini", {conv2d_shape, conv2d + "input_layers=conv2d_shape"});

  nntrainerIniTest::save_ini(ini_name_full, {nw_base_mse, backbone_valid});

  EXPECT_EQ(NN_full.loadFromConfig(ini_name_full), ML_ERROR_NONE);
  EXPECT_THROW(NN_full.compile(), std::invalid_argument);
  EXPECT_EQ(NN_full.initialize(), ML_ERROR_NOT_SUPPORTED);

  nntrainerIniTest::save_ini(ini_name_scaled, {nw_base_mse, backbone_scaled});

  EXPECT_EQ(NN_scaled.loadFromConfig(ini_name_scaled), ML_ERROR_NONE);
  EXPECT_THROW(NN_scaled.compile(), std::invalid_argument);
  EXPECT_EQ(NN_scaled.initialize(), ML_ERROR_NOT_SUPPORTED);
}
/**
 * @brief Ini file unittest with backbone
 * @note Input shape is striped from backbone and not provided in model file
 */
TEST(nntrainerIniTest, backbone_p_17) {
  const char *ini_name_scaled = "backbone_p17_scaled.ini";
  const char *ini_name_full = "backbone_p17_full.ini";
  nntrainer::NeuralNetwork NN_scaled, NN_full;

  nntrainerIniTest::save_ini(
    "base.ini", {conv2d_shape, conv2d + "input_layers=conv2d_shape"});

  nntrainerIniTest::save_ini(
    ini_name_full,
    {nw_base_mse, input2d, backbone_valid + "input_layers=inputlayer"});

  EXPECT_EQ(NN_full.loadFromConfig(ini_name_full), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_full.initialize(), ML_ERROR_NONE);

  nntrainerIniTest::save_ini(
    ini_name_scaled,
    {nw_base_mse, input2d, backbone_scaled + "input_layers=inputlayer"});

  EXPECT_EQ(NN_scaled.loadFromConfig(ini_name_scaled), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN_scaled.initialize(), ML_ERROR_NONE);
}

/**
 * @brief Ini file unittest with backbone
 * @note Output layer name not found, epmty backbone
 */
TEST(nntrainerIniTest, backbone_n_18) {
  const char *ini_name = "backbone_n18.ini";
  nntrainer::NeuralNetwork NN;

  nntrainerIniTest::save_ini("base.ini",
                             {input2d, conv2d + "input_layers=inputlayer",
                              flatten + "input_layers=conv2d"});
  nntrainerIniTest::save_ini(
    ini_name,
    {nw_base_mse, input, backbone_valid_inout + "input_layers=inputlayer"});

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
  EXPECT_EQ(NN.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN.getNetworkGraph().getSorted().size(), 3);
}

/**
 * @brief Ini file unittest with backbone
 * @note Input layer name not found, epmty backbone
 */
TEST(nntrainerIniTest, backbone_n_19) {
  const char *ini_name = "backbone_n19.ini";
  nntrainer::NeuralNetwork NN;

  nntrainerIniTest::save_ini("base.ini",
                             {input2d, conv2d + "input_layers=inputlayer",
                              batch_normal + "input_layers=conv2d"});
  nntrainerIniTest::save_ini(
    ini_name,
    {nw_base_mse, input, backbone_valid_inout + "input_layers=inputlayer"});

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
  EXPECT_EQ(NN.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN.initialize(), ML_ERROR_NONE);

  EXPECT_EQ(NN.getNetworkGraph().getSorted().size(), 3);
}

/**
 * @brief Ini file unittest with backbone
 * @note input and output layer specified are found
 */
TEST(nntrainerIniTest, backbone_p_20) {
  const char *ini_name = "backbone_p20.ini";
  nntrainer::NeuralNetwork NN;

  nntrainerIniTest::save_ini(
    "base.ini", {input2d, conv2d + "input_layers=inputlayer",
                 flatten + "input_layers=conv2d", out + "input_layers=flat"});
  nntrainerIniTest::save_ini(
    ini_name,
    {nw_base_mse, input, backbone_valid_inout + "input_layers=inputlayer"});

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
  EXPECT_EQ(NN.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN.initialize(), ML_ERROR_NONE);
  EXPECT_EQ(NN.getNetworkGraph().getSorted().size(), 6);
}

/**
 * @brief Ini file unittest with backbone
 * @note Input layer name not found, epmty backbone
 */
TEST(nntrainerIniTest, backbone_n_21) {
  const char *ini_name = "backbone_n21.ini";
  nntrainer::NeuralNetwork NN;

  nntrainerIniTest::save_ini("base.ini",
                             {input2d, conv2d + "input_layers=inputlayer",
                              batch_normal + "input_layers=conv2d",
                              out + "input_layers=bn"});
  nntrainerIniTest::save_ini(
    ini_name,
    {nw_base_mse, input, backbone_valid_inout + "input_layers=inputlayer"});

  EXPECT_EQ(NN.loadFromConfig(ini_name), ML_ERROR_NONE);
  EXPECT_EQ(NN.compile(), ML_ERROR_NONE);
  EXPECT_EQ(NN.initialize(), ML_ERROR_NONE);
  EXPECT_EQ(NN.getNetworkGraph().getSorted().size(), 3);
  // EXPECT_EQ(NN.getGraph().size(), 3);
}

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

#if defined(__TIZEN__)
  /** ignore tizen feature check while running the testcases */
  set_feature_state(SUPPORTED);
#endif

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

#if defined(__TIZEN__)
  /** reset tizen feature check state */
  set_feature_state(NOT_CHECKED_YET);
#endif

  return result;
}
