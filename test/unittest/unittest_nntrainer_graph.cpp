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
#include <util_func.h>

#include "nntrainer_test_util.h"

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

namespace initest {
typedef enum {
  LOAD = 1 << 0,   /**< should fail at load */
  INIT = 1 << 1,   /**< should fail at init */
  REINIT = 1 << 2, /**< should fail at reinit */
} IniFailAt;
};

template <typename T>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  std::mt19937 gen(42);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

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
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

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
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

  for (auto &node : g) {
    EXPECT_NO_THROW(nn_model->addLayer(node));
  }

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  EXPECT_EQ(nn_model->setOptimizer(std::move(optimizer)), ML_ERROR_NONE);
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
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

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

TEST(nntrainerGraphUnitTest, NoLossLayerWhenInferenceMode) {
  std::unique_ptr<ml::train::Model> model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  model->addLayer(ml::train::createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape", "1:1:256")}));

  for (int i = 0; i < 3; ++i) {
    model->addLayer(ml::train::createLayer(
      "fully_connected",
      {nntrainer::withKey("unit", 1024),
       nntrainer::withKey("weight_initializer", "xavier_uniform"),
       nntrainer::withKey("bias_initializer", "zeros")}));
  }
  model->addLayer(ml::train::createLayer(
    "fully_connected",
    {nntrainer::withKey("unit", 100),
     nntrainer::withKey("weight_initializer", "xavier_uniform"),
     nntrainer::withKey("bias_initializer", "zeros")}));

  model->setProperty({nntrainer::withKey("batch_size", 1),
                      nntrainer::withKey("epochs", 1),
                      nntrainer::withKey("fsu", "false"),
                      nntrainer::withKey("model_tensor_type", "FP32-FP32")});

  int status = model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  float input[256];

  for (unsigned int i = 0; i < 256; ++i) {
    input[i] = i;
  }

  std::vector<float *> in;
  std::vector<float *> ans;

  in.push_back(input);

  ans = model->inference(1, in);

  in.clear();
  ans.clear();
}

#ifdef ENABLE_GGML
TEST(nntrainerGraphUnitTest, Q4_K_FP32_Model) {
  std::unique_ptr<ml::train::Model> model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  model->addLayer(ml::train::createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape", "1:1:1024")}));
  
  int NUM_LAYER = 1;

  for (int i = 0; i < NUM_LAYER; ++i) {
    model->addLayer(ml::train::createLayer(
      "fully_connected", {nntrainer::withKey("unit", 1024),
                          nntrainer::withKey("disable_bias", "true")}));
  }

  model->setProperty({nntrainer::withKey("batch_size", 1),
                      nntrainer::withKey("epochs", 1),
                      nntrainer::withKey("fsu", "false"),
                      nntrainer::withKey("model_tensor_type", "Q4_K-FP32")});

  int status = model->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = model->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  model->load("fc_q4kx8.bin");

  std::unique_ptr<ml::train::Model> model_fp =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  model_fp->addLayer(ml::train::createLayer(
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape", "1:1:1024")}));

  for (int i = 0; i < NUM_LAYER; ++i) {
    model_fp->addLayer(ml::train::createLayer(
      "fully_connected", {nntrainer::withKey("unit", 1024),
                          nntrainer::withKey("disable_bias", "true")}));
  }

  model_fp->setProperty({nntrainer::withKey("batch_size", 1),
                         nntrainer::withKey("epochs", 1),
                         nntrainer::withKey("fsu", "false"),
                         nntrainer::withKey("model_tensor_type", "FP32-FP32")});

  status = model_fp->compile(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = model_fp->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_EQ(status, ML_ERROR_NONE);

  model_fp->load("fc_float.bin");

  float input[1024];
  std::vector<float> input_data = generate_random_vector<float>(1024, -0.05, 0.05);

  for (unsigned int i = 0; i < 1024; ++i) {
    input[i] = input_data[i];
  }

  std::vector<float *> in;
  std::vector<float *> ans;
  std::vector<float *> ans2;

  in.push_back(input);

  ans = model->inference(1, in);
  ans2 = model_fp->inference(1, in);

  const float eps = 1e-5;
  auto mean_squared_error = mse<float, float>(ans[0], ans2[0], 1024);
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << ans[0][i] << " " ;
  }
  std::cout << " ... ";
  for (unsigned int i = 5; i > 0; --i) {
    std::cout << ans[0][1024-i] << " " ;
  }
  std::cout << std::endl;
  for (unsigned int i = 0; i < 5; ++i) {
    std::cout << ans2[0][i] << " " ;
  }
  std::cout << " ... ";
  for (unsigned int i = 5; i > 0; --i) {
    std::cout << ans2[0][1024-i] << " " ;
  }
  std::cout << std::endl;

  in.clear();
  ans.clear();
  ans2.clear();
}
#endif

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
