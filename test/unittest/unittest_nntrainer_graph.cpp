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
#include <ini_wrapper.h>
#include <neuralnet.h>

#include "nntrainer_test_util.h"

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

namespace initest {
typedef enum {
  LOAD = 1 << 0,   /**< should fail at load */
  INIT = 1 << 1,   /**< should fail at init */
  REINIT = 1 << 2, /**< should fail at reinit */
} IniFailAt;
};

/**
 * @brief Graph Tester
 *
 */
class nntrainerGraphTest
  : public ::testing::TestWithParam<
      std::tuple<const char *, const nntrainer::IniWrapper::Sections, int>> {

protected:
  nntrainerGraphTest() : failAt(0), name("") {}
  virtual void SetUp() {
    name = std::string(std::get<0>(GetParam()));
    std::cout << "starting test case : " << name << std::endl << std::endl;

    const auto &sections = std::get<1>(GetParam());

    ini = nntrainer::IniWrapper(name, sections);

    failAt = std::get<2>(GetParam());
    ini.save_ini();
  }

  virtual void TearDown() { ini.erase_ini(); }

  std::string getIniName() { return ini.getIniName(); }

  bool failAtLoad() { return failAt & initest::IniFailAt::LOAD; }

  bool failAtInit() { return failAt & initest::IniFailAt::INIT; }

  bool failAtReinit() { return failAt & initest::IniFailAt::REINIT; }

  nntrainer::NeuralNetwork NN;

private:
  int failAt;
  std::string name;
  nntrainer::IniWrapper ini;
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

  status = NN.reinitialize();
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
  NN.backwarding(1);
}

static nntrainer::IniSection nw_base("model", "Type = NeuralNetwork | "
                                              "batch_size = 16 | "
                                              "loss = cross");

static nntrainer::IniSection sgd("Optimizer", "Type = sgd |"
                                              "Learning_rate = 1");

static nntrainer::IniSection input("inputlayer", "Type = input |"
                                                 "Input_Shape = 3:32:32");

static nntrainer::IniSection conv2d8("conv2d8", "Type = conv2d |"
                                                "input_layers=inputlayer |"
                                                "bias_initializer = zeros |"
                                                "Activation = relu |"
                                                "filters = 32 |"
                                                "kernel_size = 3,3 |"
                                                "stride = 1,1 |"
                                                "padding = 0,0");

static nntrainer::IniSection conv2d9("conv2d9", "Type = conv2d |"
                                                "input_layers=conv2d8 |"
                                                "bias_initializer = zeros |"
                                                "Activation = relu |"
                                                "filters = 64 |"
                                                "kernel_size = 3,3 |"
                                                "stride = 1,1 |"
                                                "padding = 0,0");

static nntrainer::IniSection pooling2("pooling2", "Type = pooling2d |"
                                                  "input_layers = conv2d9 |"
                                                  "pool_size = 3, 3 |"
                                                  "stride = 3, 3 |"
                                                  "padding = 0, 0 |"
                                                  "pooling=max");

static nntrainer::IniSection out0("out0", "Type = multiout |"
                                          "input_layers = pooling2");

static nntrainer::IniSection conv2d10("conv2d10", "Type = conv2d |"
                                                  "input_layers=out0 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 1,1");

static nntrainer::IniSection conv2d11("conv2d11", "Type = conv2d |"
                                                  "input_layers=conv2d10 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 1,1");

static nntrainer::IniSection addition0("addition0",
                                       "Type=addition |"
                                       "input_layers = conv2d11, out0 ");

static nntrainer::IniSection out1("out1", "Type = multiout |"
                                          "input_layers = addition0");

static nntrainer::IniSection conv2d12("conv2d12", "Type = conv2d |"
                                                  "input_layers=out1 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 1,1");

static nntrainer::IniSection conv2d13("conv2d13", "Type = conv2d |"
                                                  "input_layers=conv2d12 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 1,1");

static nntrainer::IniSection addition1("addition1",
                                       "Type=addition |"
                                       "input_layers = conv2d13, out1 ");

static nntrainer::IniSection conv2d14("conv2d14", "Type = conv2d |"
                                                  "input_layers=addition1 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu |"
                                                  "filters = 64 |"
                                                  "kernel_size = 3,3 |"
                                                  "stride = 1,1 |"
                                                  "padding = 0,0");

static nntrainer::IniSection
  pooling3("pooling3", "Type = pooling2d |"
                       "input_layers = conv2d14 |"
                       "pooling=global_average | flatten = true");

static nntrainer::IniSection fclayer0("fclayer0", "Type = fully_connected |"
                                                  "Unit = 256 |"
                                                  "input_layers = pooling3 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = relu");

static nntrainer::IniSection fclayer1("fclayer1", "Type = fully_connected |"
                                                  "Unit = 10 |"
                                                  "input_layers = fclayer0 |"
                                                  "bias_initializer = zeros |"
                                                  "Activation = softmax");

static int SUCCESS = 0;

/**
 * @brief make ini test case from given parameter
 */
std::tuple<const char *, const nntrainer::IniWrapper::Sections, int>
mkIniTc(const char *name, const nntrainer::IniWrapper::Sections vec, int flag) {
  return std::make_tuple(name, vec, flag);
}

GTEST_PARAMETER_TEST(nntrainerIniAutoTests, nntrainerGraphTest,
                     ::testing::Values(mkIniTc(
                       "basic_p",
                       {nw_base, sgd, input, conv2d8, conv2d9, pooling2, out0,
                        conv2d10, conv2d11, addition0, out1, conv2d12, conv2d13,
                        addition1, conv2d14, pooling3, fclayer0, fclayer1},
                       SUCCESS)));

TEST(nntrainerGraphUnitTest, cross_with_relu) {
  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});
  auto relu0 = LayerRepresentation(
    "activation", {"name=relu0", "activation=relu", "input_layers=in0"});

  auto g = makeGraph({input0, relu0});

  nntrainer::NetworkGraph ng;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "cross")});

  for (auto &node : g) {
    EXPECT_NO_THROW(nn_model->addLayer(node));
  }

  EXPECT_NE(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_NE(nn_model->initialize(), ML_ERROR_NONE);
}

TEST(nntrainerGraphUnitTest, compile_twice) {
  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});
  auto relu0 = LayerRepresentation(
    "activation", {"name=relu0", "activation=softmax", "input_layers=in0"});

  auto g = makeGraph({input0, relu0});

  nntrainer::NetworkGraph ng;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "cross")});

  for (auto &node : g) {
    EXPECT_NO_THROW(nn_model->addLayer(node));
  }

  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);
  try {
    nn_model->compile();
  } catch (const std::exception &e) {
    EXPECT_STREQ(e.what(), "cannot remap identifiers after finalized");
  }
}

TEST(nntrainerGraphUnitTest, call_functions) {
  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});
  auto relu0 = LayerRepresentation(
    "activation", {"name=relu0", "activation=softmax", "input_layers=in0"});

  auto g = makeGraph({input0, relu0});

  nntrainer::NetworkGraph ng;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "cross")});

  for (auto &node : g) {
    EXPECT_NO_THROW(nn_model->addLayer(node));
  }

  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  try {
    for (auto &node : g) {
      nn_model->addLayer(node);
    }
  } catch (const std::exception &e) {
    EXPECT_STREQ(e.what(), "Cannot modify graph after compile");
  }
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
