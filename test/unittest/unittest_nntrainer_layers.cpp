// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_nntrainer_layers.cpp
 * @date        03 June 2020
 * @brief       Unit test utility for layers.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <fstream>
#include <regex>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <conv2d_layer.h>
#include <embedding.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <layer_internal.h>
#include <loss_layer.h>
#include <manager.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <optimizer_factory.h>
#include <parse_util.h>
#include <pooling2d_layer.h>
#include <preprocess_flip_layer.h>
#include <preprocess_translate_layer.h>
#include <rnn.h>
#include <tensor_dim.h>
#include <util_func.h>

using nntrainer::sharedConstTensor;
using nntrainer::sharedTensor;

static std::string getDimensionString(const nntrainer::TensorDim &dim) {
  std::string dim_str;
  for (unsigned int i = 0; i < nntrainer::MAXDIM; i++) {
    dim_str += std::to_string(dim.getTensorDim(i));
    dim_str += ":";
  }
  dim_str.pop_back();

  return dim_str;
}

static std::string getLayerResPath(const std::string &file_name) {
  return getResPath(file_name, {"test", "unittest_layers"});
}

template <typename LayerType>
class nntrainer_abstractLayer : public ::testing::Test {
protected:
  virtual void SetUp() {
    manager = nntrainer::Manager(true, false);
    status = ML_ERROR_NONE;
    manager.setInPlaceActivationOptimization(false);
    manager.setInferenceInOutMemoryOptimization(false);
    prepareLayer();
    initialize();
  }

  virtual int initialize() {
    int status = layer.initialize(manager);
    EXPECT_EQ(status, ML_ERROR_NONE);

    manager.initializeWeights();
    manager.allocateWeights();

    in = nntrainer::Tensor(layer.getInputDimension()[0]);
    out = nntrainer::Tensor(layer.getOutputDimension()[0]);

    layer.setInputBuffers(manager.trackLayerInputs(
      layer.getType(), layer.getName(), layer.getInputDimension()));
    layer.setOutputBuffers(manager.trackLayerOutputs(
      layer.getType(), layer.getName(), layer.getOutputDimension()));

    return status;
  }

  virtual int reinitialize() {
    resetLayer();
    prepareLayer();
    status = initialize();
    EXPECT_EQ(status, ML_ERROR_NONE);
    return status;
  }

  virtual int reinitialize(const std::string str, int batch_size = 1) {
    resetLayer();
    prepareLayer();
    int status = setProperty(str);
    EXPECT_EQ(status, ML_ERROR_NONE);
    setBatch(batch_size);
    status = initialize();
    EXPECT_EQ(status, ML_ERROR_NONE);
    return status;
  }

  // anchor point to prepare layer
  virtual void prepareLayer(){};

  virtual void resetLayer() {
    manager.reset();
    layer = LayerType();
  }

  virtual void setInputDim(const std::string &dimension) {
    ASSERT_EQ(layer.setProperty({"input_shape=" + dimension}), ML_ERROR_NONE);
  }

  void setBatch(unsigned int batch) { layer.setBatch(batch); }

  void matchOutput(const float *result, const float *golden, size_t length) {
    for (size_t i = 0; i < length; ++i) {
      EXPECT_NEAR(result[i], golden[i], local_tolerance);
    }
  }

  void matchOutput(const nntrainer::Tensor &result,
                   const nntrainer::Tensor &golden) {
    matchOutput(result.getData(), golden.getData(), result.length());
  }

  void matchOutput(const float *result, const char *expected) {
    nntrainer::Tensor golden;
    loadFile(expected, golden);
    /** FIXME: golden.length() is possibly 0 many times, verify and fix this */
    matchOutput(result, golden.getData(), golden.length());
  }

  void matchOutput(const nntrainer::Tensor &result, const char *expected) {
    matchOutput(result.getData(), expected);
  }

  void matchOutput(const std::vector<float> result, const char *expected) {
    matchOutput(result.data(), expected);
  }

  // setting property separated by "|"
  int setProperty(const std::string &str) {
    std::vector<std::string> input_str;
    std::regex words_regex("[^|]+");
    auto words_begin =
      std::sregex_iterator(str.begin(), str.end(), words_regex);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
      input_str.push_back((*i).str());
    }
    int status = layer.setProperty(input_str);
    EXPECT_EQ(status, ML_ERROR_NONE);

    return status;
  }

  // setting optimizer property separated by "|"
  int setOptimizer(nntrainer::OptType type, const std::string &str = "") {
    std::vector<std::string> input_str;
    std::regex words_regex("[^|]+");
    auto words_begin =
      std::sregex_iterator(str.begin(), str.end(), words_regex);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
      input_str.push_back((*i).str());
    }

    EXPECT_NO_THROW(opt = nntrainer::createOptimizer(type));

    status = opt->setProperty(input_str);
    EXPECT_EQ(status, ML_ERROR_NONE);

    status = opt->initialize();
    EXPECT_EQ(status, ML_ERROR_NONE);

    EXPECT_NO_THROW(opt->addOptimizerVariable(layer.getWeightsRef()));

    return status;
  }

  void allocateMemory() {
    manager.initializeTensors(true);
    manager.allocateTensors();
  }

  template <typename T> void saveFile(const char *filename, T &t) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.good()) {
      throw std::runtime_error("could not save file");
    }
    t.save(file);
    file.close();
  }

  template <typename T> void loadFile(const char *filename, T &t) {
    std::ifstream file(getLayerResPath(filename));
    if (!file.good()) {
      throw std::runtime_error("could not read, check filename");
    }
    t.read(file);
    file.close();
  }

  template <typename T>
  void loadFile(const char *filename, std::vector<T> &ts) {
    std::ifstream file(getLayerResPath(filename));
    if (!file.good()) {
      throw std::runtime_error("could not read, check filename");
    }
    /// @note if you want to load weight only, you either have to load weight
    /// one by one or do that before setting optimizer to the layer
    for (auto &t : ts)
      t.read(file);
    file.close();
  }

  LayerType layer;
  int status;
  nntrainer::Tensor in;
  nntrainer::Tensor out;
  float local_tolerance = tolerance;
  nntrainer::Manager manager;
  std::shared_ptr<nntrainer::Optimizer> opt;
};

class nntrainer_InputLayer
  : public nntrainer_abstractLayer<nntrainer::InputLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("3:28:28");
    setBatch(1);
  }
};

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, initialize_01_p) {
  int status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_InputLayer, set_property_01_n) {
  EXPECT_THROW(
    layer.setProperty(nntrainer::Layer::PropertyType::input_shape, "0:3:2:1"),
    std::invalid_argument);
}

TEST_F(nntrainer_InputLayer, set_property_02_p) {
  nntrainer::TensorDim dim;
  int status = setProperty("input_shape=3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension()[0];
  EXPECT_EQ(dim.getTensorDim(0), 1u);
  EXPECT_EQ(dim.getTensorDim(1), 3u);
  EXPECT_EQ(dim.getTensorDim(2), 2u);
  EXPECT_EQ(dim.getTensorDim(3), 1u);
}

TEST_F(nntrainer_InputLayer, set_property_03_p) {
  nntrainer::TensorDim dim;
  int status = setProperty("input_shape=1:3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension()[0];
  EXPECT_EQ(dim.getTensorDim(0), 1u);
  EXPECT_EQ(dim.getTensorDim(1), 3u);
  EXPECT_EQ(dim.getTensorDim(2), 2u);
  EXPECT_EQ(dim.getTensorDim(3), 1u);
}

TEST_F(nntrainer_InputLayer, set_property_04_p) {
  nntrainer::TensorDim dim;
  int status = setProperty("input_shape=4:3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Set input shape ignores batch size */
  dim = layer.getInputDimension()[0];
  EXPECT_EQ(dim.getTensorDim(0), 1u);
  EXPECT_EQ(dim.getTensorDim(1), 3u);
  EXPECT_EQ(dim.getTensorDim(2), 2u);
  EXPECT_EQ(dim.getTensorDim(3), 1u);
}

TEST_F(nntrainer_InputLayer, set_property_05_p) {
  nntrainer::TensorDim dim;
  int status = ML_ERROR_NONE;

  setBatch(5);
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension()[0];
  EXPECT_EQ(dim.getTensorDim(0), 5u);
  EXPECT_EQ(dim.getTensorDim(1), 3u);
  EXPECT_EQ(dim.getTensorDim(2), 28u);
  EXPECT_EQ(dim.getTensorDim(3), 28u);

  /** Original batch size is retained */
  status = setProperty("input_shape=1:3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension()[0];
  EXPECT_EQ(dim.getTensorDim(0), 5u);
  EXPECT_EQ(dim.getTensorDim(1), 3u);
  EXPECT_EQ(dim.getTensorDim(2), 2u);
  EXPECT_EQ(dim.getTensorDim(3), 1u);

  /** Original batch size is retained */
  status = setProperty("input_shape=4:3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension()[0];
  EXPECT_EQ(dim.getTensorDim(0), 5u);
  EXPECT_EQ(dim.getTensorDim(1), 3u);
  EXPECT_EQ(dim.getTensorDim(2), 2u);
  EXPECT_EQ(dim.getTensorDim(3), 1u);
}

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, setOptimizer_01_p) {
  status = setOptimizer(nntrainer::OptType::ADAM, "learning_rate=0.001 |"
                                                  "beta1=0.9 |"
                                                  "beta2=0.9999 |"
                                                  "epsilon=1e-7");

  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, setActivation_01_p) {
  int status = layer.setProperty({"activation=tanh"});
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, setActivation_02_n) {
  int status = layer.setProperty({"activation=unknown"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = layer.setProperty({"activation=random"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, checkValidation_01_p) {
  int status = layer.setProperty({"activation=tanh"});
  ASSERT_EQ(status, ML_ERROR_NONE);

  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

class nntrainer_PreprocessFlipLayer
  : public nntrainer_abstractLayer<nntrainer::PreprocessFlipLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("3:5:5");
    setBatch(1);
  }
};

/**
 * @brief Preprocess Flip Layer
 */
TEST_F(nntrainer_PreprocessFlipLayer, initialize_01_p) {
  int status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Preprocess Flip Layer
 */
TEST_F(nntrainer_PreprocessFlipLayer, set_property_01_n) {
  int status = layer.setProperty({"flip_direction=vertical_and_horizontal"});
  EXPECT_NE(status, ML_ERROR_NONE);
}

/**
 * @brief Preprocess Flip Layer
 */
TEST_F(nntrainer_PreprocessFlipLayer, set_property_02_n) {
  int status = layer.setProperty({"flip_direction=flip"});
  EXPECT_NE(status, ML_ERROR_NONE);
}

/**
 * @brief Preprocess Flip Layer
 */
TEST_F(nntrainer_PreprocessFlipLayer, forwarding_01_p) {
  layer.setProperty({"input_shape=1:3:5:5"});
  layer.setProperty({"flip_direction=horizontal"});
  nntrainer::Tensor in(nntrainer::TensorDim({1, 3, 5, 5}));
  nntrainer::Tensor out_flip, out_orig;

  in.setRandNormal(0.0f, 10.0f);

  while (out_flip.uninitialized() || (out_flip == in)) {
    EXPECT_NO_THROW(out_flip =
                      *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  }
  EXPECT_NE(out_flip, in);

  while (out_orig.uninitialized() || (out_orig != in)) {
    EXPECT_NO_THROW(
      out_orig = *layer.forwarding_with_val({MAKE_SHARED_TENSOR(out_flip)})[0]);
  }
  EXPECT_EQ(out_orig, in);
}

/**
 * @brief Preprocess Flip Layer
 */
TEST_F(nntrainer_PreprocessFlipLayer, forwarding_02_p) {
  layer.setProperty({"input_shape=1:2:4:4"});
  layer.setProperty({"flip_direction=horizontal"});
  nntrainer::Tensor in(nntrainer::TensorDim({1, 2, 4, 4}));
  nntrainer::Tensor out_flip, out_orig;

  in.setRandNormal(0.0f, 10.0f);

  while (out_flip.uninitialized() || (out_flip == in)) {
    EXPECT_NO_THROW(out_flip =
                      *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  }
  EXPECT_NE(out_flip, in);

  while (out_orig.uninitialized() || (out_orig != in)) {
    EXPECT_NO_THROW(
      out_orig = *layer.forwarding_with_val({MAKE_SHARED_TENSOR(out_flip)})[0]);
  }
  EXPECT_EQ(out_orig, in);
}

/**
 * @brief Preprocess Flip Layer
 */
TEST_F(nntrainer_PreprocessFlipLayer, forwarding_03_p) {
  layer.setProperty({"input_shape=1:3:5:5"});
  layer.setProperty({"flip_direction=vertical"});
  nntrainer::Tensor in(nntrainer::TensorDim({1, 3, 5, 5}));
  nntrainer::Tensor out_flip, out_orig;

  in.setRandNormal(0.0f, 10.0f);

  while (out_flip.uninitialized() || (out_flip == in)) {
    EXPECT_NO_THROW(out_flip =
                      *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  }
  EXPECT_NE(out_flip, in);

  while (out_orig.uninitialized() || (out_orig != in)) {
    EXPECT_NO_THROW(
      out_orig = *layer.forwarding_with_val({MAKE_SHARED_TENSOR(out_flip)})[0]);
  }
  EXPECT_EQ(out_orig, in);
}

/**
 * @brief Preprocess Flip Layer
 */
TEST_F(nntrainer_PreprocessFlipLayer, forwarding_04_p) {
  layer.setProperty({"input_shape=1:2:4:4"});
  layer.setProperty({"flip_direction=vertical"});
  nntrainer::Tensor in(nntrainer::TensorDim({1, 2, 4, 4}));
  nntrainer::Tensor out_flip, out_orig;

  in.setRandNormal(0.0f, 10.0f);

  while (out_flip.uninitialized() || (out_flip == in)) {
    EXPECT_NO_THROW(out_flip =
                      *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  }
  EXPECT_NE(out_flip, in);

  while (out_orig.uninitialized() || (out_orig != in)) {
    EXPECT_NO_THROW(
      out_orig = *layer.forwarding_with_val({MAKE_SHARED_TENSOR(out_flip)})[0]);
  }
  EXPECT_EQ(out_orig, in);
}

/**
 * @brief Preprocess Flip Layer
 */
TEST_F(nntrainer_PreprocessFlipLayer, forwarding_05_p) {
  layer.setProperty({"input_shape=1:2:4:4"});
  layer.setBatch(5);
  layer.setProperty({"flip_direction=horizontal_and_vertical"});
  nntrainer::Tensor in(nntrainer::TensorDim({5, 2, 4, 4}));
  nntrainer::Tensor out_flip;

  in.setRandNormal(0.0f, 10.0f);

  EXPECT_NO_THROW(out_flip =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
}

class nntrainer_PreprocessTranslateLayer
  : public nntrainer_abstractLayer<nntrainer::PreprocessTranslateLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("3:32:32");
    setBatch(1);
  }
};

/**
 * @brief Preprocess Translate Layer
 */
TEST_F(nntrainer_PreprocessTranslateLayer, initialize_01_p) {
  int status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Preprocess Translate Layer
 */
TEST_F(nntrainer_PreprocessTranslateLayer, set_property_01_p) {
  int status = layer.setProperty({"random_translate=0.5"});
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Preprocess Translate Layer
 */
TEST_F(nntrainer_PreprocessTranslateLayer, forwarding_01_p) {
  layer.setBatch(2);
  layer.setProperty({"random_translate=0.0"});
  layer.initialize(manager);

  manager.initializeTensors(true);

  nntrainer::Tensor in(nntrainer::TensorDim({2, 3, 32, 32}));
  nntrainer::Tensor out_trans;

  in.setRandNormal(0.0f, 10.0f);

  EXPECT_NO_THROW(out_trans =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  EXPECT_EQ(out_trans, in);
}

/**
 * @brief Preprocess Translate Layer
 */
TEST_F(nntrainer_PreprocessTranslateLayer, forwarding_02_p) {
#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
  layer.setBatch(1);
  layer.setProperty({"random_translate=0.1"});
  layer.initialize(manager);

  manager.initializeTensors(true);

  nntrainer::Tensor in(nntrainer::TensorDim({1, 3, 32, 32}));

  in.setRandNormal(0.0f, 10.0f);

  nntrainer::Tensor out_trans;
  EXPECT_NO_THROW(out_trans =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  EXPECT_NE(out_trans, in);
#else
  layer.setBatch(1);
  layer.setProperty({"random_translate=0.1"});
  EXPECT_THROW(layer.initialize(manager), nntrainer::exception::not_supported);
#endif
}

class nntrainer_FullyConnectedLayer
  : public nntrainer_abstractLayer<nntrainer::FullyConnectedLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("1:28:28");
    setBatch(5);
    setProperty("unit=1");
  }
};

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, initialize_01_p) {
  int status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer without setting any parameter
 */
TEST(nntrainer_FullyConnectedLayer_n, initialize_02_n) {
  nntrainer::Manager manager{true, false};
  manager.setInferenceInOutMemoryOptimization(false);
  nntrainer::FullyConnectedLayer layer;
  EXPECT_THROW(layer.initialize(manager), std::invalid_argument);
}

/**
 * @brief Fully Connected Layer without setting unit
 */
TEST(nntrainer_FullyConnectedLayer_n, initialize_03_n) {
  nntrainer::Manager manager{true, false};
  manager.setInferenceInOutMemoryOptimization(false);
  nntrainer::FullyConnectedLayer layer;
  layer.setProperty({"input_shape=32:1:28:28"});

  EXPECT_THROW(layer.initialize(manager), std::invalid_argument);
}

/**
 * @brief FullyConnected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, initialize_04_p) {
  std::string layer_name;

  /** Layer name can be set */
  layer_name = "FCLayer0";
  status = layer.setProperty({"name=" + layer_name});
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(layer.getName(), layer_name);

  /** Layer name cannot be updated once set */
  layer_name = "FCLayer1";
  status = layer.setProperty({"name=" + layer_name});
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(layer.getName(), layer_name);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer_init_name, initialize_05_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer0, layer1;
  nntrainer::TensorDim d;
  std::string layer_name;

  /** no name is set */
  layer_name = layer0.getName();
  EXPECT_EQ(layer_name.length(), 0u);

  /** Set empty name */
  status = layer0.setProperty({"name="});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, setOptimizer_01_p) {
  status = setOptimizer(nntrainer::OptType::ADAM, "learning_rate=0.001 |"
                                                  "beta1=0.9 |"
                                                  "beta2=0.9999 |"
                                                  "epsilon=1e-7");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief FullyConnected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, setOptimizer_02_p) {
  status = setOptimizer(nntrainer::OptType::SGD, "learning_rate=0.1");
  EXPECT_EQ(status, ML_ERROR_NONE);
}
/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, setActivation_01_p) {
  status = layer.setProperty({"activation=tanh"});
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, setActivation_02_n) {
  status = layer.setProperty({"activation=unknown"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief FullyConnected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, checkValidation_01_p) {
  status = layer.setProperty({"activation=ReLU"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

class nntrainer_FullyConnectedLayer_TFmatch
  : public nntrainer_abstractLayer<nntrainer::FullyConnectedLayer> {

protected:
  typedef nntrainer_abstractLayer<nntrainer::FullyConnectedLayer> super;

  virtual int initialize() {
    int status = super::initialize();
    label =
      MAKE_SHARED_TENSOR(nntrainer::Tensor(layer.getOutputDimension()[0]));

    layers.clear();

    return status;
  }

  void addActivation(nntrainer::ActivationType type) {
    std::shared_ptr<nntrainer::ActivationLayer> act_layer =
      std::make_shared<nntrainer::ActivationLayer>(type);

    status = act_layer->setProperty(
      {"input_shape=" + getDimensionString(layer.getOutputDimension()[0])});
    EXPECT_EQ(status, ML_ERROR_NONE);

    act_layer->setBatch(layer.getOutputDimension()[0].batch());

    status = act_layer->initialize(manager);
    EXPECT_EQ(status, ML_ERROR_NONE);

    act_layer->setInputBuffers(
      manager.trackLayerInputs(act_layer->getType(), act_layer->getName(),
                               act_layer->getInputDimension()));
    act_layer->setOutputBuffers(
      manager.trackLayerOutputs(act_layer->getType(), act_layer->getName(),
                                act_layer->getOutputDimension()));

    layers.push_back(act_layer);
  }

  void addLoss(nntrainer::LossType type) {
    std::shared_ptr<nntrainer::LossLayer> loss_layer =
      std::make_shared<nntrainer::LossLayer>();

    status = loss_layer->setProperty(
      {"input_shape=" + getDimensionString(layer.getOutputDimension()[0])});
    EXPECT_EQ(status, ML_ERROR_NONE);

    loss_layer->setBatch(layer.getOutputDimension()[0].batch());

    status = loss_layer->initialize(manager);
    EXPECT_EQ(status, ML_ERROR_NONE);
    status = loss_layer->setLoss(type);
    EXPECT_EQ(status, ML_ERROR_NONE);
    loss_type = type;

    loss_layer->setInputBuffers(
      manager.trackLayerInputs(loss_layer->getType(), loss_layer->getName(),
                               loss_layer->getInputDimension()));
    loss_layer->setOutputBuffers(
      manager.trackLayerOutputs(loss_layer->getType(), loss_layer->getName(),
                                loss_layer->getOutputDimension()));

    layers.push_back(loss_layer);

    if (type == nntrainer::LossType::LOSS_ENTROPY_SOFTMAX) {
      nntrainer::Tensor weight = layer.weightAt(0).getVariable();
      loadFile("tc_fc_1_FCLayer_sensible.in", in);
      loadFile("tc_fc_1_FCKernel_sensible.in", weight);
      loadFile("tc_fc_1_FCLabel_sensible.in", *label);
    }
  }

  void matchForwarding(const char *file) {
    std::vector<nntrainer::Tensor> v;
    for (unsigned int i = 0; i < layer.getNumWeights(); ++i) {
      v.push_back(layer.weightAt(i).getVariable());
    }

    loadFile("tc_fc_1_FCLayer.in", in);
    loadFile("tc_fc_1_FCKernel.in", v);
    loadFile("tc_fc_1_FCLabel.in", *label);

    sharedConstTensor out;
    EXPECT_NO_THROW(out =
                      layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

    if (layers.size() > 0) {
      for (unsigned int idx = 0; idx < layers.size() - 1; idx++) {
        EXPECT_NO_THROW(out = layers[idx]->forwarding_with_val({out})[0]);
      }

      if (nntrainer::istrequal(layers.back()->getType(),
                               nntrainer::LossLayer::type)) {
        std::shared_ptr<nntrainer::LossLayer> loss_layer =
          std::static_pointer_cast<nntrainer::LossLayer>(layers.back());
        EXPECT_NO_THROW(out =
                          loss_layer->forwarding_with_val({out}, {label})[0]);
      } else {
        EXPECT_NO_THROW(out = layers.back()->forwarding_with_val({out})[0]);
      }
      EXPECT_EQ(status, ML_ERROR_NONE);
    }
    matchOutput(*out, file);
  }

  void matchLoss(const char *file) {
    nntrainer::Tensor loss(1, 1, 1, 1);
    loadFile(file, loss);
    EXPECT_NEAR(layers.back()->getLoss(), *(loss.getData()), local_tolerance);
  }

  void matchBackwarding(const char *file_dx, const char *file_uw,
                        const char *file_g, const bool with_loss = false) {

    int idx = layers.size() - 1;
    sharedTensor def_derivative =
      MAKE_SHARED_TENSOR(constant(1.0, 3, 1, 1, 15));
    sharedConstTensor back_out;

    if (layers.size() && nntrainer::istrequal(layers.back()->getType(),
                                              nntrainer::LossLayer::type)) {
      if (with_loss) {
        EXPECT_NO_THROW(layers.back()->backwarding_with_val({label}));
        back_out = MAKE_SHARED_TENSOR(layers.back()->getDerivatives()[0]);
      } else {
        back_out = def_derivative;
      }
      idx -= 1;
    } else {
      back_out = def_derivative;
    }

    for (; idx >= 0; --idx)
      EXPECT_NO_THROW(
        back_out = layers[idx]->backwarding_with_val(1, {back_out}, opt)[0]);

    EXPECT_NO_THROW(back_out =
                      layer.backwarding_with_val(1, {back_out}, opt)[0]);
    matchOutput(*back_out.get(), file_dx);

    loadUpdatedWeightsGradients(file_uw, file_g);
    matchUpdatedWeightsGradients();
  }

  void loadUpdatedWeightsGradients(const char *file_uw, const char *file_g) {
    for (int idx = 0; idx < 2; ++idx) {
      new_w.push_back(
        nntrainer::Tensor(layer.weightAt(idx).getVariable().getDim()));
      grad.push_back(
        nntrainer::Tensor(layer.weightAt(idx).getGradient().getDim()));
    }

    loadFile(file_uw, new_w);
    loadFile(file_g, grad);
  }

  virtual void prepareLayer() {
    setInputDim("1:1:12");
    setBatch(3);
    setProperty("unit=15");
    setProperty("bias_initializer=zeros");
  }

  void matchUpdatedWeightsGradients() {
    std::vector<nntrainer::Weight> params = layer.getWeights();

    bool match_grads = true;
    if (loss_type != nntrainer::LossType::LOSS_UNKNOWN)
      match_grads = false;

    /** Match gradients and updated weights */
    for (int idx = 0; idx < 2; ++idx) {
      if (match_grads)
        matchOutput(params[idx].getGradient(), grad[idx]);
      matchOutput(params[idx].getVariable(), new_w[idx]);
    }
  }

  sharedTensor label;
  std::vector<nntrainer::Tensor> new_w;
  std::vector<nntrainer::Tensor> grad;
  std::vector<std::shared_ptr<nntrainer::Layer>> layers;
  nntrainer::LossType loss_type = nntrainer::LossType::LOSS_UNKNOWN;
};

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_00_p) {
  std::vector<float> weight_data;
  std::vector<float> bias_data;

  setOptimizer(nntrainer::OptType::ADAM, "learning_rate=1.0");
  allocateMemory();

  sharedConstTensor out;

  EXPECT_NO_THROW(out = layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  nntrainer::Tensor derivatives(3, 1, 1, 15);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }

  nntrainer::Tensor result;
  EXPECT_NO_THROW(result = *layer.backwarding_with_val(
                    1, {MAKE_SHARED_TENSOR(derivatives)}, opt)[0]);

  matchOutput(result, "tc_fc_1_goldenFCGradientAdam.out");

  auto param_data = layer.getWeights();

  nntrainer::Weight &param = param_data[0];
  nntrainer::Tensor weight = param.getVariable();
  matchOutput(weight, "tc_fc_1_goldenFCUpdatedWeightAdam.out");

  nntrainer::Weight &bias_param = param_data[1];
  nntrainer::Tensor bias = bias_param.getVariable();
  matchOutput(bias, "tc_fc_1_goldenFCUpdatedBiasAdam.out");
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch,
       forwarding_backwarding_loss_00_p) {
  addLoss(nntrainer::LossType::LOSS_ENTROPY_SOFTMAX);
  setOptimizer(nntrainer::OptType::ADAM, "learning_rate=0.0001");
  allocateMemory();

  matchForwarding("tc_fc_1_goldenFCResultSoftmaxCrossAdam.out");

  matchBackwarding("tc_fc_1_goldenFCGradientDxSoftmaxCrossAdam.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSoftmaxCrossAdam.out",
                   "tc_fc_1_goldenFCGradientsSoftmaxCrossAdam.out", true);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_01_p) {

  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  /** Verify forwarding and backwarding without loss */
  matchForwarding("tc_fc_1_goldenFCResultActNone.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxActNone.out",
                   "tc_fc_1_goldenFCUpdatedWeightsActNone.out",
                   "tc_fc_1_goldenFCGradientsActNone.out");
}

/**
 * @brief Fully Connected Layer forward with MSE loss
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_02_p) {

  addActivation(nntrainer::ActivationType::ACT_SIGMOID);
  addLoss(nntrainer::LossType::LOSS_MSE);
  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSigmoidMse.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSigmoidMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSigmoid.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSigmoid.out",
                   "tc_fc_1_goldenFCGradientsSigmoid.out");
}

/**
 * @brief Fully Connected Layer forward with MSE loss
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_03_p) {

  addActivation(nntrainer::ActivationType::ACT_SOFTMAX);
  addLoss(nntrainer::LossType::LOSS_MSE);
  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSoftmaxMse.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSoftmaxMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSoftmax.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSoftmax.out",
                   "tc_fc_1_goldenFCGradientsSoftmax.out");
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_04_p) {

  addLoss(nntrainer::LossType::LOSS_MSE);
  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultActNone.out");
  matchOutput(*label, "tc_fc_1_FCLabel.in");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossActNoneMse.out");

  /**
   * This lowers the tolerance for below check. As the data values are in the
   * range [1, 10) (integer represented as floats), the values get very large
   * which leads to higher values floating point error.
   * This error exists in gradient. However, when added to weight with learning
   * rate of 1.0, this error disappears. So, for now, local tolerance just for
   * this test has been reduced to match the output.
   * Note: this issue occurs only for a single value out of matrix of 180
   * elements
   */
  local_tolerance = 1.3e-4;
  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxActNoneMse.out",
                   "tc_fc_1_goldenFCUpdatedWeightsActNoneMse.out",
                   "tc_fc_1_goldenFCGradientsActNoneMse.out", true);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_05_p) {

  addActivation(nntrainer::ActivationType::ACT_SIGMOID);
  addLoss(nntrainer::LossType::LOSS_MSE);
  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSigmoidMse.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSigmoidMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSigmoidMse.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSigmoidMse.out",
                   "tc_fc_1_goldenFCGradientsSigmoidMse.out", true);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_06_p) {

  addActivation(nntrainer::ActivationType::ACT_SOFTMAX);
  addLoss(nntrainer::LossType::LOSS_MSE);
  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSoftmaxMse.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSoftmaxMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSoftmaxMse.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSoftmaxMse.out",
                   "tc_fc_1_goldenFCGradientsSoftmaxMse.out", true);
}

/**
 * @brief Fully Connected Layer forward with Cross Entropy loss
 * @todo Upgrade this to adam to verify adam
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_07_p) {

  addLoss(nntrainer::LossType::LOSS_ENTROPY_SIGMOID);
  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSigmoidCross.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSigmoidCross.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSigmoidCross.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSigmoidCross.out",
                   "tc_fc_1_goldenFCGradientsSigmoidCross.out", true);
}

/**
 * @brief Fully Connected Layer forward with Cross Entropy loss
 * @todo Upgrade this to adam to verify adam
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_08_p) {

  addLoss(nntrainer::LossType::LOSS_ENTROPY_SOFTMAX);
  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSoftmaxCross.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSoftmaxCross.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSoftmaxCross.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSoftmaxCross.out",
                   "tc_fc_1_goldenFCGradientsSoftmaxCross.out", true);
}

class nntrainer_BatchNormalizationLayer
  : public nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> {
protected:
  typedef nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> super;

  virtual int reinitialize() {
    int status = super::reinitialize();

    std::vector<nntrainer::Tensor> v;

    for (unsigned int i = 0; i < layer.getNumWeights(); ++i) {
      v.push_back(layer.weightAt(i).getVariable());
    }

    loadFile("tc_bn_fc_1_BNLayerInput.in", in);
    loadFile("tc_bn_fc_1_BNLayerWeights.in", v);
    return status;
  }

  virtual void prepareLayer() {
    setProperty("input_shape=1:1:12 | epsilon=0.001 | momentum=0.90");
    setBatch(3);
    setOptimizer(nntrainer::OptType::SGD, "learning_rate=1");
  }
};

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, initialize_01_p) {
  status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, setOptimizer_01_p) {
  status = setOptimizer(
    nntrainer::OptType::ADAM,
    "learning_rate=0.001 | beta1=0.9 | beta2=0.9999 | epsilon=1e-7");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, setActivation_01_p) {
  status = layer.setProperty({"activation=sigmoid"});
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, setActivation_02_n) {
  status = layer.setProperty({"activation=unknown"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, checkValidation_01_p) {
  status = layer.setProperty({"activation=relu"});
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_BatchNormalizationLayer, forward_backward_training_01_p) {
  layer.setTrainable(true);
  sharedConstTensor forward_result;

  allocateMemory();
  EXPECT_NO_THROW(forward_result =
                    layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  matchOutput(*forward_result, "tc_bn_fc_1_goldenBNResultForward.out");

  nntrainer::Tensor backward_in(layer.getOutputDimension()[0]);
  loadFile("tc_bn_fc_1_goldenBNLayerBackwardDxIn.out", backward_in);

  nntrainer::Tensor backward_result =
    *layer.backwarding_with_val(1, {MAKE_SHARED_TENSOR(backward_in)}, opt)[0];

  matchOutput(backward_result, "tc_bn_fc_1_goldenBNLayerBackwardDx.out");
}

class nntrainer_BatchNormalizationLayer_Conv
  : public nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> {
protected:
  typedef nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> super;

  virtual int reinitialize() {
    int status = super::reinitialize();
    std::vector<nntrainer::Tensor> v;
    for (unsigned int i = 0; i < layer.getNumWeights(); ++i) {
      v.push_back(layer.weightAt(i).getVariable());
    }
    loadFile("tc_bn_conv_1_BNLayerInput.in", in);
    loadFile("tc_bn_conv_1_BNLayerWeights.in", v);
    return status;
  }

  virtual void prepareLayer() {
    setProperty("input_shape=2:4:5 | epsilon=0.001 | momentum=0.90");
    setBatch(3);
    setOptimizer(nntrainer::OptType::SGD, "learning_rate=1");
  }
};

TEST_F(nntrainer_BatchNormalizationLayer_Conv, forward_backward_training_01_p) {
  layer.setTrainable(true);
  sharedConstTensor forward_result;
  allocateMemory();

  forward_result = layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0];
  matchOutput(*forward_result, "tc_bn_conv_1_goldenBNResultForward.out");

  nntrainer::Tensor backward_in(layer.getOutputDimension()[0]);
  loadFile("tc_bn_conv_1_goldenBNLayerBackwardDxIn.out", backward_in);

  nntrainer::Tensor backward_result =
    *layer.backwarding_with_val(1, {MAKE_SHARED_TENSOR(backward_in)}, opt)[0];

  matchOutput(backward_result, "tc_bn_conv_1_goldenBNLayerBackwardDx.out");
}

class nntrainer_BatchNormalizationLayer_Conv2
  : public nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> {
protected:
  typedef nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> super;

  virtual int reinitialize() {
    int status = super::reinitialize();

    std::vector<nntrainer::Tensor> v;
    for (unsigned int i = 0; i < layer.getNumWeights(); ++i) {
      v.push_back(layer.weightAt(i).getVariable());
    }

    loadFile("tc_bn_conv_2_BNLayerInput.in", in);
    loadFile("tc_bn_conv_2_BNLayerWeights.in", v);
    return status;
  }

  virtual void prepareLayer() {
    setProperty("input_shape=2:4:5 | epsilon=0.001 | momentum=0.90");
    setBatch(1);
    setOptimizer(nntrainer::OptType::SGD, "learning_rate=1");
  }
};

TEST_F(nntrainer_BatchNormalizationLayer_Conv2,
       forward_backward_training_01_p) {
  layer.setTrainable(true);
  sharedConstTensor forward_result;
  allocateMemory();

  forward_result = layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0];
  matchOutput(*forward_result, "tc_bn_conv_2_goldenBNResultForward.out");

  nntrainer::Tensor backward_in(layer.getOutputDimension()[0]);
  loadFile("tc_bn_conv_2_goldenBNLayerBackwardDxIn.out", backward_in);

  nntrainer::Tensor backward_result =
    *layer.backwarding_with_val(1, {MAKE_SHARED_TENSOR(backward_in)}, opt)[0];

  matchOutput(backward_result, "tc_bn_conv_2_goldenBNLayerBackwardDx.out");
}

class nntrainer_Conv2DLayer
  : public nntrainer_abstractLayer<nntrainer::Conv2DLayer> {

protected:
  typedef nntrainer_abstractLayer<nntrainer::Conv2DLayer> super;

  virtual void prepareLayer() {
    int status =
      setProperty("input_shape=3:28:28 |"
                  "bias_initializer=zeros |"
                  "activation=sigmoid |"
                  "weight_regularizer=l2norm |"
                  "weight_regularizer_constant= 0.005 |"
                  "weight_initializer=xavier_uniform |"
                  "filters=12 | kernel_size= 5,5 | stride=3,3 | padding=1,1");

    EXPECT_EQ(status, ML_ERROR_NONE);
    setBatch(32);
  }

  nntrainer::Tensor result;
};

TEST_F(nntrainer_Conv2DLayer, print_01_p) {
  std::stringstream ss, ss2;
  layer.printPreset(ss, nntrainer::Layer::PrintPreset::PRINT_ALL);
  ss2 << layer;
  EXPECT_GT(ss.str().size(), 100u);
  EXPECT_GT(ss2.str().size(), 100u);
}

/**
 * @brief Convolution 2D Layer
 */
TEST_F(nntrainer_Conv2DLayer, initialize_01_p) {
  status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Convolution 2D Layer save and read and save
 */
TEST_F(nntrainer_Conv2DLayer, save_read_01_p) {
  allocateMemory();
  saveFile("save.bin", layer);
  saveFile("save1.bin", layer);

  std::ifstream read_file("save.bin");
  ASSERT_TRUE(read_file.good());

  std::ifstream read_file2("save1.bin");
  ASSERT_TRUE(read_file2.good());

  float d1, d2;

  for (int i = 0; i < (5 * 5 * 3 * 12) + 12; ++i) {
    read_file.read((char *)&d1, sizeof(float));
    read_file2.read((char *)&d2, sizeof(float));
    EXPECT_FLOAT_EQ(d1, d2);
  }

  read_file.close();
  read_file2.close();
}

/**
 * @brief Convolution 2D Layer
 */
TEST_F(nntrainer_Conv2DLayer, forwarding_01_p) {
  reinitialize("input_shape=3:7:7 |"
               "bias_initializer = zeros |"
               "weight_initializer=xavier_uniform |"
               "filters=2 | kernel_size=3,3 | stride=1, 1 | padding=0,0");
  allocateMemory();

  ASSERT_EQ(in.getDim(), nntrainer::TensorDim(1, 3, 7, 7));
  ASSERT_EQ(out.getDim(), nntrainer::TensorDim(1, 2, 5, 5));

  loadFile("tc_conv2d_1_conv2DLayer.in", in);
  loadFile("tc_conv2d_1_conv2DKernel.in", layer);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  matchOutput(out, "tc_conv2d_1_goldenConv2DResult.out");
}

/**
 * @brief Convolution 2D Layer
 */

TEST_F(nntrainer_Conv2DLayer, forwarding_02_p) {
  status =
    reinitialize("input_shape=3:7:7 |"
                 "bias_initializer = zeros |"
                 "weight_initializer=xavier_uniform |"
                 "filters=3 | kernel_size=3,3 | stride=1, 1 | padding=0,0",
                 2);
  allocateMemory();

  ASSERT_EQ(in.getDim(), nntrainer::TensorDim(2, 3, 7, 7));
  ASSERT_EQ(out.getDim(), nntrainer::TensorDim(2, 3, 5, 5));

  loadFile("tc_conv2d_2_conv2DLayer.in", in);
  loadFile("tc_conv2d_2_conv2DKernel.in", layer);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  matchOutput(out, "tc_conv2d_2_goldenConv2DResult.out");
}

TEST_F(nntrainer_Conv2DLayer, backwarding_01_p) {
  status = reinitialize("input_shape=3:7:7 |"
                        "bias_initializer=zeros |"
                        "weight_initializer=xavier_uniform |"
                        "filters=2 |"
                        "kernel_size= 3,3 |"
                        "stride=1, 1 |"
                        "padding=0,0");

  nntrainer::Tensor derivatives(1, 2, 5, 5);

  loadFile("tc_conv2d_1_conv2DLayer.in", in);
  loadFile("tc_conv2d_1_conv2DKernel.in", layer);
  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(result = *layer.backwarding_with_val(
                    1, {MAKE_SHARED_TENSOR(derivatives)}, opt)[0]);

  auto param_data = layer.getWeights();
  const float *weight_grad = param_data[0].getGradient().getData();
  const float *bias_grad = param_data[1].getGradient().getData();

  matchOutput(weight_grad, "tc_conv2d_1_goldenKernelGrad.out");

  matchOutput(result, "tc_conv2d_1_goldenInputGrad.out");

  matchOutput(bias_grad, "tc_conv2d_1_goldenBiasGrad.out");
}

TEST_F(nntrainer_Conv2DLayer, backwarding_02_p) {
  status = reinitialize("input_shape=3:7:7 |"
                        "bias_initializer=zeros |"
                        "weight_initializer=xavier_uniform |"
                        "filters=3 |"
                        "kernel_size= 3,3 |"
                        "stride=1, 1 |"
                        "padding=0,0",
                        2);

  nntrainer::Tensor derivatives(2, 3, 5, 5);

  loadFile("tc_conv2d_2_conv2DLayer.in", in);
  loadFile("tc_conv2d_2_conv2DKernel.in", layer);

  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }
  EXPECT_NO_THROW(result = *layer.backwarding_with_val(
                    1, {MAKE_SHARED_TENSOR(derivatives)}, opt)[0]);

  auto param_data = layer.getWeights();
  const float *weight_grad = param_data[0].getGradient().getData();
  const float *bias_grad = param_data[1].getGradient().getData();

  matchOutput(out, "tc_conv2d_2_goldenConv2DResult.out");
  matchOutput(weight_grad, "tc_conv2d_2_goldenKernelGrad.out");
  matchOutput(result, "tc_conv2d_2_goldenInputGrad.out");
  matchOutput(bias_grad, "tc_conv2d_2_goldenBiasGrad.out");

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }

  for (int i = 0; i < 4; i++) {
    EXPECT_NO_THROW(out =
                      *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
    EXPECT_NO_THROW(result = *layer.backwarding_with_val(
                      0, {MAKE_SHARED_TENSOR(derivatives)}, opt)[0]);
  }

  /// @fixme: the output value of this test is around +/- 1.0e+07 which can't
  // be compared with smaller tolerance
  // for example, first value is -14422792, out value is -1.44228e+07
  // matchOutput(out, "tc_conv2d_2_goldenConv2DResult2.out");
  matchOutput(weight_grad, "tc_conv2d_2_goldenKernelGrad2.out");
  matchOutput(result, "tc_conv2d_2_goldenInputGrad2.out");
  matchOutput(bias_grad, "tc_conv2d_2_goldenBiasGrad2.out");
}

#ifdef USE_BLAS
// CONV2D TEST Cases is disabled. Instread of this, we could replace
// unitest_nntrainer_models
TEST_F(nntrainer_Conv2DLayer, DISABLED_backwarding_03_p) {
  status = reinitialize("input_shape=3:28:28 |"
                        "bias_initializer=zeros |"
                        "weight_initializer=zeros |"
                        "filters=6 |"
                        "kernel_size= 5,5 |"
                        "stride=1, 1 |"
                        "padding=0, 0");

  loadFile("tc_conv2d_int_conv2DLayer.in", in);
  auto manager = nntrainer::Manager();
  manager.setInferenceInOutMemoryOptimization(false);

  manager.setGradientMemoryOptimization(false);

  nntrainer::Conv2DLayer layer1;
  status =
    layer1.setProperty({"input_shape=3:28:28", "bias_initializer=zeros",
                        "weight_initializer=zeros", "filters=6",
                        "kernel_size= 5,5", "stride=1, 1", "padding=0, 0"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  layer1.setBatch(1);
  status = layer1.initialize(manager);
  EXPECT_EQ(status, ML_ERROR_NONE);

  loadFile("tc_conv2d_int_conv2DKernel.in", layer1);

  std::shared_ptr<nntrainer::Optimizer> op;
  EXPECT_NO_THROW(op = nntrainer::createOptimizer(nntrainer::OptType::SGD));
  status = op->setProperty({"learning_rate=1.0"});
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Conv2DLayer layer2;
  status = layer2.setProperty(
    {"bias_initializer=zeros", "weight_initializer=zeros", "filters=12",
     "kernel_size= 1,1", "stride=1, 1", "padding=0, 0"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  layer2.setBatch(1);
  status = layer2.setProperty(
    {"input_shape=" + getDimensionString(layer1.getOutputDimension()[0])});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer2.initialize(manager);
  EXPECT_EQ(status, ML_ERROR_NONE);

  loadFile("tc_conv2d_int_conv2DKernel2.in", layer2);
  std::shared_ptr<nntrainer::Optimizer> op2;
  EXPECT_NO_THROW(op2 = nntrainer::createOptimizer(nntrainer::OptType::SGD));
  status = op2->setProperty({"learning_rate=1.0"});
  EXPECT_EQ(status, ML_ERROR_NONE);

  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  nntrainer::Tensor derivatives(1, 12, 24, 24);

  nntrainer::Tensor out1;
  EXPECT_NO_THROW(out1 =
                    *layer1.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  nntrainer::Tensor out2;

  EXPECT_NO_THROW(out2 =
                    *layer2.forwarding_with_val({MAKE_SHARED_TENSOR(out1)})[0]);

  matchOutput(out1, "tc_conv2d_int_goldenConv2DResult.out");
  matchOutput(out2, "tc_conv2d_int_goldenConv2DResult2.out");

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }

  nntrainer::Tensor result2;
  EXPECT_NO_THROW(result2 = *layer2.backwarding_with_val(
                    1, {MAKE_SHARED_TENSOR(derivatives)}, opt)[0]);

  EXPECT_NO_THROW(result = *layer1.backwarding_with_val(
                    1, {MAKE_SHARED_TENSOR(result2)}, opt)[0]);

  /** Compare second conv */
  auto param_data = layer2.getWeights();
  const float *weight_grad = param_data[0].getGradient().getData();
  const float *bias_grad = param_data[1].getGradient().getData();

  matchOutput(weight_grad, "tc_conv2d_int_goldenKernel2Grad.out");
  matchOutput(bias_grad, "tc_conv2d_int_goldenBias2Grad.out");

  /** Compare first conv */
  param_data = layer1.getWeights();
  weight_grad = param_data[0].getGradient().getData();
  bias_grad = param_data[1].getGradient().getData();

  matchOutput(weight_grad, "tc_conv2d_int_goldenKernelGrad.out");
  matchOutput(bias_grad, "tc_conv2d_int_goldenBiasGrad.out");

  matchOutput(result, "tc_conv2d_int_goldenInputGrad.out");
}
#endif

TEST_F(nntrainer_Conv2DLayer, backwarding_04_p) {
  status = reinitialize("input_shape=6:24:24 |"
                        "bias_initializer=zeros |"
                        "weight_initializer=xavier_uniform |"
                        "filters=12 |"
                        "kernel_size=5,5 |"
                        "stride=1,1 |"
                        "padding=0,0");

  nntrainer::Tensor derivatives(1, 12, 20, 20);

  loadFile("tc_conv2d_3_conv2DLayer.in", in);
  loadFile("tc_conv2d_3_conv2DKernel.in", layer);

  setOptimizer(nntrainer::OptType::SGD, "learning_rate=1.0");
  allocateMemory();

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }
  EXPECT_NO_THROW(result = *layer.backwarding_with_val(
                    1, {MAKE_SHARED_TENSOR(derivatives)}, opt)[0]);

  auto param_data = layer.getWeights();
  const float *weight_grad = param_data[0].getGradient().getData();
  const float *bias_grad = param_data[1].getGradient().getData();

  matchOutput(weight_grad, "tc_conv2d_3_goldenKernelGrad.out");

  matchOutput(result, "tc_conv2d_3_goldenInputGrad.out");

  matchOutput(bias_grad, "tc_conv2d_3_goldenBiasGrad.out");
}

class nntrainer_Pooling2DLayer
  : public nntrainer_abstractLayer<nntrainer::Pooling2DLayer> {
protected:
  void matchData(float *golden) {
    float *out_ptr;

    out_ptr = out.getData();

    for (size_t i = 0; i < out.getDim().getDataLen(); ++i) {
      EXPECT_NEAR(out_ptr[i], golden[i], local_tolerance);
    }
  }

  virtual void prepareLayer() { setInputDim("2:3:5:5"); }
};

TEST_F(nntrainer_Pooling2DLayer, setProperty_01_p) {
  setInputDim("3:5:5");
  setBatch(2);
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");
}

TEST_F(nntrainer_Pooling2DLayer, setProperty_02_n) {
  setInputDim("3:5:5");
  setBatch(2);
  int status = layer.setProperty({"pool_size="});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST_F(nntrainer_Pooling2DLayer, initialize_01_p) { reinitialize(); }

TEST_F(nntrainer_Pooling2DLayer, forwarding_01_p) {
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=max");

  initialize();
  allocateMemory();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dmax.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_02_p) {
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");

  initialize();
  allocateMemory();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Daverage.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_03_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("pooling=global_max");
  initialize();
  allocateMemory();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dglobal_max.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_04_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("pooling=global_average");
  initialize();
  allocateMemory();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dglobal_average.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_05_p) {
  resetLayer();
  setInputDim("2:5:5");
  setBatch(2);
  setProperty("pooling=global_max");
  initialize();
  allocateMemory();

  loadFile("tc_pooling2d_2.in", in);
  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dglobal_max.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_06_p) {
  resetLayer();
  setInputDim("2:5:5");
  setBatch(2);
  setProperty("pooling=global_average");
  initialize();
  allocateMemory();

  loadFile("tc_pooling2d_2.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);
  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dglobal_average.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_01_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=max");

  initialize();
  allocateMemory();
  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  nntrainer::Tensor grad(out.getDim());

  for (unsigned int i = 0; i < grad.length(); ++i) {
    grad.getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(
    in = *layer.backwarding_with_val(1, {MAKE_SHARED_TENSOR(grad)}, opt)[0]);

  matchOutput(in, "tc_pooling2d_1_goldenPooling2DmaxGrad.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_02_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");
  initialize();
  allocateMemory();
  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  sharedTensor grad = MAKE_SHARED_TENSOR(out.getDim());

  for (unsigned int i = 0; i < grad->length(); ++i) {
    grad->getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(in = *layer.backwarding_with_val(1, {grad}, opt)[0]);

  matchOutput(in, "tc_pooling2d_1_goldenPooling2DaverageGrad.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_03_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("pooling=global_max");
  initialize();
  allocateMemory();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  nntrainer::Tensor grad(out.getDim());

  for (unsigned int i = 0; i < grad.getDim().getDataLen(); ++i) {
    grad.getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(
    in = *layer.backwarding_with_val(1, {MAKE_SHARED_TENSOR(grad)}, opt)[0]);

  matchOutput(in, "tc_pooling2d_1_goldenPooling2Dglobal_maxGrad.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_04_p) {
  setInputDim("2:5:5");
  setProperty("pooling=global_average");
  initialize();
  allocateMemory();
  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  nntrainer::Tensor grad(out.getDim());

  for (unsigned int i = 0; i < grad.getDim().getDataLen(); ++i) {
    grad.getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(
    in = *layer.backwarding_with_val(1, {MAKE_SHARED_TENSOR(grad)}, opt)[0]);

  matchOutput(in, "tc_pooling2d_1_goldenPooling2Dglobal_averageGrad.out");
}

class nntrainer_FlattenLayer
  : public nntrainer_abstractLayer<nntrainer::FlattenLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("2:4:4");
    setBatch(1);
  }
};

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, forwarding_01_p) {
  reinitialize();

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(1, 1, 1, 32));

  loadFile("tc_pooling2d_1_goldenPooling2Dmax.out", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dmax.out");
}

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, forwarding_02_p) {
  setInputDim("2:4:4");
  setBatch(2);
  initialize();

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(2, 1, 1, 32));

  loadFile("tc_pooling2d_2_goldenPooling2Dmax.out", in);

  EXPECT_NO_THROW(out =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(in)})[0]);

  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dmax.out");
}

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, backwarding_01_p) {
  reinitialize();

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(1, 1, 1, 32));

  loadFile("tc_pooling2d_1_goldenPooling2Dmax.out", out);

  EXPECT_NO_THROW(
    in = *layer.backwarding_with_val(1, {MAKE_SHARED_TENSOR(out)}, opt)[0]);
  EXPECT_EQ(in.getDim(), nntrainer::TensorDim(1, 2, 4, 4));

  matchOutput(in, "tc_pooling2d_1_goldenPooling2Dmax.out");
}

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, backwarding_02_p) {
  setInputDim("2:4:4");
  setBatch(2);
  initialize();

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(2, 1, 1, 32));

  loadFile("tc_pooling2d_2_goldenPooling2Dmax.out", out);

  EXPECT_NO_THROW(
    in = *layer.backwarding_with_val(1, {MAKE_SHARED_TENSOR(out)}, opt)[0]);
  EXPECT_EQ(in.getDim(), nntrainer::TensorDim(2, 2, 4, 4));

  matchOutput(in, "tc_pooling2d_2_goldenPooling2Dmax.out");
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setLoss_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setLoss(nntrainer::LossType::LOSS_ENTROPY);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setLoss_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setLoss(nntrainer::LossType::LOSS_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_LossLayer, forward_loss_unknown_n) {
  nntrainer::LossLayer layer;
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);
  nntrainer::Tensor b = constant(1.0, 1, 1, 1, 1);

  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();
  EXPECT_THROW(
    layer.forwarding_with_val({MAKE_SHARED_TENSOR(a)}, {MAKE_SHARED_TENSOR(b)}),
    std::runtime_error);
}

TEST(nntrainer_LossLayer, backward_loss_unknown_n) {
  nntrainer::LossLayer layer;
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);

  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();
  EXPECT_THROW(layer.backwarding_with_val({MAKE_SHARED_TENSOR(a)}),
               std::runtime_error);
}

TEST(nntrainer_LossLayer, forward_loss_forward_entropy_n) {
  nntrainer::LossLayer layer;
  layer.setLoss(nntrainer::LossType::LOSS_ENTROPY);
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);
  nntrainer::Tensor b = constant(1.0, 1, 1, 1, 1);

  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();
  EXPECT_THROW(
    layer.forwarding_with_val({MAKE_SHARED_TENSOR(a)}, {MAKE_SHARED_TENSOR(b)}),
    std::runtime_error);
}

TEST(nntrainer_LossLayer, backward_loss_backward_entropy_n) {
  nntrainer::LossLayer layer;
  layer.setLoss(nntrainer::LossType::LOSS_ENTROPY);
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);

  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();
  EXPECT_THROW(layer.backwarding_with_val({MAKE_SHARED_TENSOR(a)}),
               std::runtime_error);
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setProperty_through_vector_n) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setProperty({"loss=cross"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_LossLayer, setProperty_individual_01_n) {
  nntrainer::LossLayer layer;
  EXPECT_THROW(
    layer.setProperty(nntrainer::Layer::PropertyType::filters, "1:2"),
    nntrainer::exception::not_supported);
}

TEST(nntrainer_LossLayer, setProperty_individual_02_n) {
  nntrainer::LossLayer layer;
  EXPECT_THROW(layer.setProperty(nntrainer::Layer::PropertyType::filters,
                                 "invalid_string"),
               nntrainer::exception::not_supported);
}

TEST(nntrainer_ActivationLayer, init_01_n) {
  nntrainer::Manager manager{true, false};
  manager.setInferenceInOutMemoryOptimization(false);
  nntrainer::ActivationLayer layer;
  EXPECT_THROW(layer.initialize(manager), std::invalid_argument);
}

TEST(nntrainer_ActivationLayer, init_02_p) {
  nntrainer::Manager manager{true, false};
  manager.setInferenceInOutMemoryOptimization(false);
  int status = ML_ERROR_NONE;
  nntrainer::ActivationLayer layer;

  status = layer.setProperty({"input_shape=1:1:1:1"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.initialize(manager);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_ActivationLayer, setType_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::ActivationLayer layer;

  status = layer.setProperty({"activation=relu"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.setProperty({"activation=softmax"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.setProperty({"activation=sigmoid"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer.setProperty({"activation=tanh"});
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_ActivationLayer, setType_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::ActivationLayer layer;

  status = layer.setProperty({"activation=random"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = layer.setProperty({"activation=unknown"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_ActivationLayer, forward_backward_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::ActivationLayer layer(nntrainer::ActivationType::ACT_RELU);

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));
  nntrainer::Tensor expected(batch, channel, height, width);
  GEN_TEST_INPUT(expected, nntrainer::ActiFunc::relu((l - 4) * 0.1 * (i + 1)));

  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);
  manager.setInPlaceActivationOptimization(true);

  layer.setProperty({"input_shape=3:1:1:10"});
  layer.setBatch(3);
  layer.initialize(manager);

  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));
  manager.initializeTensors(true);
  manager.allocateTensors();

  nntrainer::Tensor result;
  EXPECT_NO_THROW(result =
                    *layer.forwarding_with_val({MAKE_SHARED_TENSOR(input)})[0]);
  EXPECT_TRUE(result == expected);

  expected.copy(input);
  EXPECT_NO_THROW(result = *layer.backwarding_with_val(
                    1, {MAKE_SHARED_TENSOR(constant(1.0, 3, 1, 1, 10))})[0]);
  GEN_TEST_INPUT(expected,
                 nntrainer::ActiFunc::reluPrime(
                   nntrainer::ActiFunc::relu((l - 4) * 0.1 * (i + 1))));
  EXPECT_TRUE(result == expected);
}

/**
 * @brief Addition Layer
 */
class nntrainer_AdditionLayer
  : public nntrainer_abstractLayer<nntrainer::AdditionLayer> {
protected:
  virtual void prepareLayer() {
    setProperty("num_inputs=1");
    setInputDim("3:28:28");
    setBatch(32);
  }
};

TEST_F(nntrainer_AdditionLayer, initialize_01_p) {
  status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_AdditionLayer, initialize_02_p) {
  nntrainer::AdditionLayer layer;
  layer.setProperty({"input_shape=1:1:1:1"});
  status = layer.initialize(manager);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_AdditionLayer, checkValidation_01_p) {
  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_AdditionLayer, setProperty_01_p) {
  setProperty("num_inputs=10");
  status = layer.initialize(manager);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_AdditionLayer, setProperty_02_n) {
  status = layer.setProperty({"num_inputs=0"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/*
 *Disabled until input_layer keyward is enabled.
 */

TEST_F(nntrainer_AdditionLayer, DISABLED_forwarding_02_n) {
  setProperty("num_inputs=2");

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[1], std::default_delete<nntrainer::Tensor[]>());
  nntrainer::Tensor &in = *input;

  in = nntrainer::Tensor(layer.getInputDimension()[0]);

  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();

  EXPECT_THROW(layer.forwarding_with_val({input}), std::runtime_error);
}

TEST_F(nntrainer_AdditionLayer, DISABLED_forwarding_03_p) {
  setProperty("num_inputs=2");

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[2], std::default_delete<nntrainer::Tensor[]>());
  nntrainer::Tensor &in = *input;
  in = nntrainer::Tensor(layer.getInputDimension()[0]);

  input.get()[1] = *input;

  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  EXPECT_NO_THROW(layer.forwarding_with_val({input}));
}

class nntrainer_EmbeddingLayer
  : public nntrainer_abstractLayer<nntrainer::EmbeddingLayer> {

protected:
  typedef nntrainer_abstractLayer<nntrainer::EmbeddingLayer> super;

  virtual void prepareLayer() {
    int status = setProperty("in_dim=50 |"
                             "out_dim=8 |"
                             "in_length=12");
    EXPECT_EQ(status, ML_ERROR_NONE);
    setBatch(3);
  }

  nntrainer::Tensor result;
};

TEST_F(nntrainer_EmbeddingLayer, initialize_01_p) {
  status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_EmbeddingLayer, forwarding_01_p) {
  float sentence[36] = {45, 16, 32, 27, 34, 33, 0,  0,  0, 0, 0,  0,
                        24, 2,  27, 34, 33, 37, 32, 27, 3, 0, 0,  0,
                        22, 27, 16, 28, 35, 33, 7,  2,  2, 3, 33, 35};

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[1], std::default_delete<nntrainer::Tensor[]>());

  nntrainer::Tensor &in = *input;

  in = nntrainer::Tensor(nntrainer::TensorDim(3, 1, 1, 12), sentence);

  nntrainer::Manager manager;

  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeTensors(false);
  manager.allocateTensors();

  EXPECT_NO_THROW(layer.forwarding_with_val({input}));
}

TEST_F(nntrainer_EmbeddingLayer, forwarding_02_p) {

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[1], std::default_delete<nntrainer::Tensor[]>());

  nntrainer::Tensor &in = *input;

  in = nntrainer::Tensor(nntrainer::TensorDim(3, 1, 1, 12));

  loadFile("tc_embedding_01_Input.in", in);

  nntrainer::Manager manager;

  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeTensors(false);
  manager.allocateTensors();

  EXPECT_NO_THROW(out = *layer.forwarding_with_val({input})[0]);

  matchOutput(out, "tc_embedding_01_golden.out");
}

TEST_F(nntrainer_EmbeddingLayer, forwarding_backwarding_01_p) {
  float sentence[36] = {45, 16, 32, 27, 34, 33, 0,  0,  0, 0, 0,  0,
                        24, 2,  27, 34, 33, 37, 32, 27, 3, 0, 0,  0,
                        22, 27, 16, 28, 35, 33, 7,  2,  2, 3, 33, 35};

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[1], std::default_delete<nntrainer::Tensor[]>());

  nntrainer::Tensor &in = *input;

  in = nntrainer::Tensor(nntrainer::TensorDim(3, 1, 1, 12), sentence);

  nntrainer::Manager manager;

  manager.setInferenceInOutMemoryOptimization(false);
  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeTensors(true);
  manager.allocateTensors();

  EXPECT_NO_THROW(layer.forwarding_with_val({input}));

  nntrainer::Tensor derivatives(3, 1, 12, 8);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }

  setOptimizer(nntrainer::OptType::ADAM, "learning_rate=1.0");

  allocateMemory();

  nntrainer::Tensor result;
  EXPECT_NO_THROW(result = *layer.backwarding_with_val(
                    1, {MAKE_SHARED_TENSOR(derivatives)}, opt)[0]);
}

class nntrainer_RNNLayer : public nntrainer_abstractLayer<nntrainer::RNNLayer> {

protected:
  typedef nntrainer_abstractLayer<nntrainer::RNNLayer> super;

  virtual void prepareLayer() {
    int status = setProperty("unit=3 | weight_initializer=ones");
    EXPECT_EQ(status, ML_ERROR_NONE);
    setInputDim("2:1:3:3");
    setBatch(2);
  }

  nntrainer::Tensor result;
};

TEST_F(nntrainer_RNNLayer, initialize_01_p) {
  status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_RNNLayer, forwarding_01_p) {
  float data[18] = {1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8};

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[1], std::default_delete<nntrainer::Tensor[]>());
  nntrainer::Tensor &in = *input;
  in = nntrainer::Tensor(nntrainer::TensorDim(2, 1, 3, 3), data);
  nntrainer::Manager manager;
  manager.setInferenceInOutMemoryOptimization(false);

  layer.setInputBuffers(manager.trackLayerInputs(
    layer.getType(), layer.getName(), layer.getInputDimension()));
  layer.setOutputBuffers(manager.trackLayerOutputs(
    layer.getType(), layer.getName(), layer.getOutputDimension()));

  manager.initializeWeights();
  manager.initializeTensors(false);
  manager.allocateTensors();
  EXPECT_NO_THROW(layer.forwarding_with_val({input}, {}, false));
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

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  return result;
}
