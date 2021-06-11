// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   unittest_nntrainer_models.cpp
 * @date   19 Oct 2020
 * @brief  Model multi iteration, itegrated test
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <input_layer.h>
#include <layer.h>
#include <layer_factory.h>
#include <loss_layer.h>
#include <neuralnet.h>
#include <output_layer.h>
#include <weight.h>

#include "nntrainer_test_util.h"

/**
 * @brief Get the Models Path object
 *
 * @param file_name file name
 * @return const std::string model path
 */
static const std::string getModelsPath(const std::string &file_name) {
  return getResPath(file_name, {"test", "unittest_models"});
}
/********************************************************
 * Watcher Classes                                      *
 ********************************************************/

using NodeType = std::shared_ptr<const nntrainer::LayerNode>;
using FlatGraphType = nntrainer::NeuralNetwork::FlatGraphType;
using NetworkGraphType = nntrainer::NetworkGraph;

/**
 * @brief verify tensor to the reference and throw if not match to stop
 *
 * @param actual actual tensor
 * @param expected reference tensor
 * @param error_msg error msg to print if not match
 */
void verify(const nntrainer::Tensor &actual, const nntrainer::Tensor &expected,
            const std::string &error_msg) {

  if (actual != expected) {
    std::cout
      << "============================================================\n";
    std::cout << "\033[1;33m" << error_msg << "\033[0m\n";
    std::cout << "\033[1;33mcurrent\033[0m  " << actual
              << "\033[1;33mexpected\033[0m  " << expected;

    if (actual.getDim() == expected.getDim()) {
      nntrainer::Tensor diff = actual.subtract(expected);
      const float *diff_data = diff.getData();
      std::cout << "\033[1;33mdifference\033[0m " << diff;
      std::cout << "number of data: " << diff.length() << std::endl;
      std::cout << "\033[4;33mMAX DIFF: "
                << *std::max_element(diff_data, diff_data + diff.length())
                << "\033[0m\n";
    }
    std::stringstream ss;
    ss << "\033[4;33m" << error_msg << "\033[0m";
    throw std::invalid_argument(ss.str().c_str());
  }
}

/**
 * @brief verify tensor to the reference and throw if not match to stop
 *
 * @param actual actual tensor vector
 * @param expected reference tensor vector
 * @param error_msg error msg to print if not match
 */
void verify(const std::vector<nntrainer::Tensor> &actual,
            const std::vector<nntrainer::Tensor> &expected,
            const std::string &error_msg) {
  NNTR_THROW_IF(actual.size() != expected.size(), std::invalid_argument)
    << error_msg;

  for (size_t i = 0; i < actual.size(); ++i) {
    verify(actual[i], expected[i], error_msg);
  }
}

/**
 * @brief NodeWatcher has an operation Node. This class monitors data in and out
 * happening in the node.
 *
 */
class NodeWatcher {
public:
  /**
   * @brief Construct a new Node Watcher object
   *
   */
  NodeWatcher() : node(nullptr) {}

  /**
   * @brief Construct a new Node Watcher object
   *
   * @param node node to watch.
   */
  NodeWatcher(const NodeType &node) : node(node) {
    unsigned int num_weights = node->getObject()->getNumWeights();
    try {
      node->getObject()->setTrainable(true);
    } catch (...) {
      std::cout << "Cannot set layer " << node->getType() << " trainable";
    }

    for (unsigned int i = 0; i < num_weights; ++i) {
      const nntrainer::Weight &w = node->getObject()->weightAt(i);
      expected_weights.push_back(w.clone());
    }

    for (auto &out_dim : node->getObject()->getOutputDimension()) {
      expected_output.emplace_back(out_dim);
    }

    for (auto &in_dim : node->getObject()->getInputDimension()) {
      expected_dx.emplace_back(in_dim);
    }
  }

  /**
   * @brief clones from expected weights to node->weights
   *
   */
  void readLayerWeight(std::ifstream &f) {
    for (unsigned int i = 0; i < node->getObject()->getNumWeights(); ++i) {
      /// @note below is harrasing the fact the tensor shares same base memory
      node->getObject()->weightAt(i).getVariable().read(f);
    }
  }

  /**
   * @brief forward node with verifying inputs/weights/outputs
   *
   * @param in input tensor
   * @param iteration iteration
   * @return nntrainer::sharedConstTensor
   */
  void forward(int iteration, NodeWatcher &next_node);

  /**
   * @brief forward loss node with verifying inputs/weights/outputs
   *
   * @param pred tensor predicted from the graph
   * @param answer label tensor
   * @param iteration iteration
   * @return nntrainer::sharedConstTensor
   */
  nntrainer::sharedConstTensors
  lossForward(nntrainer::sharedConstTensors pred,
              nntrainer::sharedConstTensors answer, int iteration);

  /**
   * @brief backward pass of the node with verifying inputs/gradients/outputs
   *
   * @param deriv dervatives
   * @param iteration iteration
   * @param verify_deriv should verify the derivatives
   * @param verify_grad should verify the derivatives
   * @return nntrainer::sharedConstTensor
   */
  void backward(int iteration, bool verify_derv = true,
                bool verify_grad = true);

  /**
   * @brief verify weights of the current node
   *
   * @param error_msg error_msg to put when fail
   */
  void verifyWeight(const std::string &error_msg);

  /**
   * @brief verify gradient of the current node
   *
   * @param error_msg error_msg to put when fail
   */
  void verifyGrad(const std::string &error_msg);

  /**
   * @brief Get the Loss
   *
   * @return float loss
   */
  float getLoss() { return node->getObject()->getLoss(); }

  /**
   * @brief read Node
   *
   * @param in input file stream
   */
  void read(std::ifstream &in);

  /**
   * @brief get Node type
   *
   * @return LayerType
   */
  std::string getNodeType() { return node->getType(); }

private:
  NodeType node;
  std::vector<nntrainer::Tensor> expected_output;
  std::vector<nntrainer::Tensor> expected_dx;
  std::vector<nntrainer::Weight> expected_weights;
};

/**
 * @brief GraphWatcher monitors and checks the graph operation like
 * forwarding & backwarding
 */
class GraphWatcher {
public:
  using WatchedFlatGraph = std::vector<NodeWatcher>;
  /**
   * @brief   GraphWatcher constructor
   */
  GraphWatcher(const std::string &config, const bool opt);

  /**
   * @brief check forwarding & backwarding & inference throws or not
   * @param reference model file name
   * @param label_shape shape of label tensor
   * @param iterations tensor dimension of label
   */
  void compareFor(const std::string &reference,
                  const nntrainer::TensorDim &label_shape,
                  unsigned int iterations);

  /**
   * @brief   Validate the running of the graph without any errors
   * @param label_shape shape of label tensor
   */
  void validateFor(const nntrainer::TensorDim &label_shape);

private:
  /**
   * @brief read and prepare the image & label data
   * @param f input file stream
   * @param label_dim tensor dimension of label
   * @return std::array<nntrainer::Tensor, 2> {input, label} tensors
   */
  std::array<nntrainer::Tensor, 2>
  prepareData(std::ifstream &f, const nntrainer::TensorDim &label_dim);

  /**
   * @brief read Graph
   * @param f input file stream
   */
  void readIteration(std::ifstream &f);

  nntrainer::NeuralNetwork nn;
  WatchedFlatGraph nodes;
  NodeWatcher loss_node;
  float expected_loss;
  bool optimize;
};

void NodeWatcher::read(std::ifstream &in) {
  // log prints are commented on purpose
  // std::cout << "[=======" << node->getName() << "==========]\n";

  auto read_ = [&in](auto &target) {
    // std::cout << target.getDim();
    target.read(in);
  };

  // std::cout << "expected_output " << expected_output.size() << "\n";
  std::for_each(expected_output.begin(), expected_output.end(), read_);
  // std::cout << "expected_dx " << expected_dx.size() << "\n";
  std::for_each(expected_dx.begin(), expected_dx.end(), read_);

  for (auto &i : expected_weights) {
    if (i.getTrainable()) {
      // std::cout << "weight-" << i.getName() << ": " << i.getDim();
      i.getGradientRef().read(in);
    }
  }

  for (auto &i : expected_weights) {
    // std::cout << "grad-" << i.getName() << ": " << i.getDim();
    i.getVariableRef().read(in);
  }
}

void NodeWatcher::verifyWeight(const std::string &error_msg) {
  for (unsigned int i = 0; i < expected_weights.size(); ++i) {
    verify(node->getObject()->weightAt(i).getVariable(),
           expected_weights[i].getVariable(),
           error_msg + " " + node->getObject()->weightAt(i).getName() +
             " weight");
  }
}

void NodeWatcher::verifyGrad(const std::string &error_msg) {
  for (unsigned int i = 0; i < expected_weights.size(); ++i) {
    auto weight = node->getObject()->weightAt(i);
    if (weight.getTrainable()) {
      verify(weight.getGradient(), expected_weights[i].getGradient(),
             error_msg + " " + weight.getName() + " grad");
    }
  }
}

void NodeWatcher::forward(int iteration, NodeWatcher &next_node) {
  std::stringstream ss;
  ss << "forward failed at " << node->getObject()->getName() << " at iteration "
     << iteration;
  std::string err_msg = ss.str();

  std::vector<nntrainer::Tensor> out = node->getObject()->getOutputs();

  if (!next_node.node->getObject()->supportInPlace() &&
      getNodeType() != nntrainer::OutputLayer::type)
    verify(out, expected_output, err_msg + " at output");
}

nntrainer::sharedConstTensors
NodeWatcher::lossForward(nntrainer::sharedConstTensors pred,
                         nntrainer::sharedConstTensors answer, int iteration) {
  std::stringstream ss;
  ss << "loss failed at " << node->getObject()->getName() << " at iteration "
     << iteration;
  std::string err_msg = ss.str();

  nntrainer::sharedConstTensors out =
    std::static_pointer_cast<nntrainer::LossLayer>(node->getObject())
      ->forwarding_with_val(pred, answer);

  return out;
}

void NodeWatcher::backward(int iteration, bool verify_deriv, bool verify_grad) {

  if (getNodeType() == nntrainer::OutputLayer::type) {
    return;
  }

  std::stringstream ss;
  ss << "backward failed at " << node->getObject()->getName()
     << " at iteration " << iteration;
  std::string err_msg = ss.str();

  std::vector<nntrainer::Tensor> out = node->getObject()->getDerivatives();

  if (verify_grad) {
    verifyGrad(err_msg + " grad");
  }

  if (verify_deriv) {
    verify(out, expected_dx, err_msg + " derivative");
  }

  verifyWeight(err_msg);
}

GraphWatcher::GraphWatcher(const std::string &config, const bool opt) :
  expected_loss(0.0),
  optimize(opt) {
  nn = nntrainer::NeuralNetwork();

  /** Disable gradient optimization as gradient is being matched for each layer
   */
  nn.setGradientMemoryOptimization(optimize);
  nn.setDerivativeMemoryOptimization(optimize);
  nn.setInPlaceLayerOptimization(optimize);
  nn.setInferenceInOutMemoryOptimization(optimize);

  if (nn.loadFromConfig(config)) {
    throw std::invalid_argument("load from config failed!");
  };

  if (nn.compile()) {
    throw std::invalid_argument("initiation failed");
  };

  if (nn.initialize()) {
    throw std::invalid_argument("initiation failed");
  };

  if (nn.allocate()) {
    throw std::invalid_argument("assign Memory failed");
  };

  NetworkGraphType model_graph = nn.getNetworkGraph();

  for (auto it = model_graph.cbegin(); it != model_graph.cend() - 1; ++it) {
    auto const &lnode = *it;
    nodes.push_back(NodeWatcher(lnode));
  }

  loss_node =
    NodeWatcher(model_graph.getSortedLayerNode(model_graph.size() - 1));
}

void GraphWatcher::compareFor(const std::string &reference,
                              const nntrainer::TensorDim &label_shape,
                              const unsigned int iterations) {
  std::ifstream ref(getModelsPath(reference),
                    std::ios_base::in | std::ios_base::binary);

  if (!ref.good()) {
    std::stringstream ss;
    ss << "ref is bad! ref path: " << reference;
    throw std::runtime_error(ss.str().c_str());
  }

  auto data = prepareData(ref, label_shape);
  nntrainer::sharedConstTensors input;

  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
    input = {MAKE_SHARED_TENSOR(std::get<0>(data).clone())};
    nntrainer::sharedConstTensors label = {
      MAKE_SHARED_TENSOR(std::get<1>(data).clone())};

    readIteration(ref);

    nn.forwarding(input, label);
    EXPECT_NEAR(expected_loss, loss_node.getLoss(), nntrainer::Tensor::epsilon);

    for (auto it = nodes.begin(); it != nodes.end() - 1; ++it) {
      it->forward(iteration, *(it + 1));
    }

    if (loss_node.getNodeType() == nntrainer::LossLayer::type) {
      nn.backwarding(label, iteration);

      for (auto it = nodes.rbegin(); it != nodes.rend() - 1; it++) {
        if (it == nodes.rend())
          it->backward(iteration, true, !optimize);
        else
          it->backward(iteration, !optimize, !optimize);
      }
    } else {
      EXPECT_THROW(nn.backwarding(label, iteration), std::runtime_error);
    }
  }

  /**
   * This inference is to ensure that inference runs with/without optimizations
   * for various kinds of models
   */
  EXPECT_NO_THROW(nn.inference(input, false));
}

void GraphWatcher::validateFor(const nntrainer::TensorDim &label_shape) {
  auto in_tensor = MAKE_SHARED_TENSOR(nn.getInputDimension()[0]);
  in_tensor->setRandNormal();
  nntrainer::sharedConstTensors input = {in_tensor};

  auto label_tensor = MAKE_SHARED_TENSOR(label_shape);
  label_tensor->setRandNormal();
  nntrainer::sharedConstTensors label = {label_tensor};

  EXPECT_NO_THROW(nn.forwarding(input, label));

  if (loss_node.getNodeType() == nntrainer::LossLayer::type) {
    EXPECT_NO_THROW(nn.backwarding(label, 0));
  }

  /**
   * This inference is to ensure that inference runs with/without optimizations
   * for various kinds of models
   */
  EXPECT_NO_THROW(nn.inference(input, false));
  /** run inference again which frees the memory */
  EXPECT_NO_THROW(nn.inference(input, true));
  /** run inference again which will force to allocate memory again */
  EXPECT_NO_THROW(nn.inference(input, true));
}

std::array<nntrainer::Tensor, 2>
GraphWatcher::prepareData(std::ifstream &f,
                          const nntrainer::TensorDim &label_dim) {
  nntrainer::Tensor in(nn.getInputDimension()[0]);
  nntrainer::Tensor lb(label_dim);

  in.read(f);
  lb.read(f);

  for (auto &i : nodes) {
    i.readLayerWeight(f);
  }

  return {in, lb};
}

void GraphWatcher::readIteration(std::ifstream &f) {
  for (auto &i : nodes) {
    if (i.getNodeType() == nntrainer::OutputLayer::type) {
      continue;
    }

    i.read(f);
  }

  f.read((char *)&expected_loss, sizeof(float));
}

/********************************************************
 * Tester Classes                                       *
 ********************************************************/

/**
 * @brief nntrainerModelTest fixture for parametrized test
 *
 * @param nntrainer::IniWrapper ini data
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 */
class nntrainerModelTest
  : public ::testing::TestWithParam<
      std::tuple<const nntrainer::IniWrapper, const nntrainer::TensorDim,
                 const unsigned int>> {

protected:
  nntrainerModelTest() : iteration(0), name("") {}
  virtual void SetUp() {
    auto param = GetParam();

    ini = std::get<0>(param);
    name = ini.getName();
    std::cout << "starting test case : " << name << "\n\n";

    label_dim = std::get<1>(param);
    iteration = std::get<2>(param);
    ini.save_ini();
  }

  virtual void TearDown() { ini.erase_ini(); }

  std::string getIniName() { return ini.getIniName(); }
  std::string getGoldenName() { return name + ".info"; }
  int getIteration() { return iteration; };
  nntrainer::TensorDim getLabelDim() { return label_dim; }

private:
  nntrainer::TensorDim label_dim;
  int iteration;
  std::string name;
  nntrainer::IniWrapper ini;
};

/**
 * @brief check given ini is failing/suceeding at unoptimized running
 */
TEST_P(nntrainerModelTest, model_test) {
  /** Check model with all optimizations off */
  if (getIniName().find("validate") == std::string::npos) {
    GraphWatcher g_unopt(getIniName(), false);
    g_unopt.compareFor(getGoldenName(), getLabelDim(), getIteration());

    /// add stub test for tcm
    EXPECT_EQ(std::get<0>(GetParam()), std::get<0>(GetParam()));
  }
}

/**
 * @brief check given ini is failing/suceeding at optimized running
 */
TEST_P(nntrainerModelTest, model_test_optimized) {
  /** Check model with all optimizations on */
  if (getIniName().find("validate") == std::string::npos) {
    GraphWatcher g_opt(getIniName(), true);
    g_opt.compareFor(getGoldenName(), getLabelDim(), getIteration());

    /// add stub test for tcm
    EXPECT_EQ(std::get<0>(GetParam()), std::get<0>(GetParam()));
  }
}

/**
 * @brief check given ini is failing/suceeding at validation
 */
TEST_P(nntrainerModelTest, model_test_validate) {
  /** Check model with all optimizations on */
  GraphWatcher g_opt(getIniName(), true);
  g_opt.validateFor(getLabelDim());

  /// add stub test for tcm
  EXPECT_EQ(std::get<0>(GetParam()), std::get<0>(GetParam()));
}

/**
 * @brief helper function to make model testcase
 *
 * @param const char * name of the ini and test. the tester generates name.ini
 * and try to read name.info
 * @param nntrainer::IniWrapper::Sections ini data
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 */
auto mkModelTc(const nntrainer::IniWrapper &ini, const std::string &label_dim,
               const unsigned int iteration) {
  return std::make_tuple(ini, nntrainer::TensorDim(label_dim), iteration);
}

/********************************************************
 * Actual Test                                          *
 ********************************************************/

static nntrainer::IniSection nn_base("model", "type = NeuralNetwork");
static std::string input_base = "type = input";
static std::string fc_base = "type = Fully_connected";
static std::string conv_base = "type = conv2d | stride = 1,1 | padding = 0,0";
static std::string rnn_base = "type = rnn";
static std::string lstm_base = "type = lstm";
static std::string pooling_base = "type = pooling2d | padding = 0,0";
static std::string preprocess_flip_base = "type = preprocess_flip";
static std::string preprocess_translate_base = "type = preprocess_translate";

static std::string adam_base = "optimizer=adam | beta1 = 0.9 | beta2 = 0.999 | "
                               "epsilon = 1e-7";

static nntrainer::IniSection act_base("activation", "Type = Activation");
static nntrainer::IniSection softmax_base = act_base + "Activation = softmax";
static nntrainer::IniSection sigmoid_base = act_base + "Activation = sigmoid";
static nntrainer::IniSection relu_base = act_base + "Activation = relu";
static nntrainer::IniSection bn_base("bn", "Type=batch_normalization");
static nntrainer::IniSection sgd_base("optimizer", "Type = sgd");

using I = nntrainer::IniSection;
using INI = nntrainer::IniWrapper;

/**
 * This is just a wrapper for an ini file with save / erase attached.
 * for example, fc_softmax_mse contains following ini file representation as a
 * series of IniSection
 *
 * [model]
 * Type = NeuralNetwork
 * Learning_rate = 1
 * Optimizer = sgd
 * Loss = mse
 * batch_Size = 3
 *
 * [input_1]
 * Type = input
 * Input_Shape = 1:1:3
 *
 * [dense]
 * Type = fully_connected
 * Unit = 5
 *
 * [activation]
 * Type = Activation
 * Activation = softmax
 *
 * [dense]
 * Type = fully_connected
 * Unit = 10
 *
 * [activation]
 * Type = Activation
 * Activation = softmax
 */
// clang-format off
INI fc_sigmoid_mse(
  "fc_sigmoid_mse",
  {nn_base + "loss=mse | batch_size = 3",
   sgd_base + "learning_rate = 1",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 5",
   I("act") + sigmoid_base,
   I("dense_1") + fc_base + "unit = 10",
   I("act_1") + softmax_base});

INI fc_sigmoid_cross =
  INI("fc_sigmoid_cross") + fc_sigmoid_mse + "model/loss=cross";

INI fc_relu_mse(
  "fc_relu_mse",
  {nn_base + "Loss=mse | batch_size = 3",
   sgd_base + "learning_rate = 0.1",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 10",
   I("act") + relu_base,
   I("dense_1") + fc_base + "unit = 2",
   I("act_1") + sigmoid_base + "input_layers=dense" + "input_layers=dense_1"});

INI fc_bn_sigmoid_cross(
  "fc_bn_sigmoid_cross",
  {nn_base + "loss=cross | batch_size = 3",
   sgd_base + "learning_rate = 1",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 10" + "input_layers=input",
   I("bn") + bn_base + "input_layers=dense",
   I("act") + sigmoid_base + "input_layers=bn",
   I("dense_2") + fc_base + "unit = 10" + "input_layers=act",
   I("act_3") + softmax_base + "input_layers=dense_2"});

INI fc_bn_sigmoid_mse =
  INI("fc_bn_sigmoid_mse") + fc_bn_sigmoid_cross + "model/loss=mse";

std::string mnist_pooling =
  pooling_base + "| pool_size=2,2 | stride=2,2 | pooling=average | padding=0,0";

INI mnist_conv_cross(
  "mnist_conv_cross",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=3,4 | filters=2" +"input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("pool_1") + mnist_pooling+"input_layers=act_1",
    I("flatten", "type=flatten")+"input_layers=pool_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_3") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_1x1(
  "conv_1x1",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=1,1 | filters=4",
    I("act_1") + sigmoid_base,
    I("flatten", "type=flatten") ,
    I("outputlayer") + fc_base + "unit = 10",
    I("act_2") + softmax_base
  }
);

INI conv_input_matches_kernel(
  "conv_input_matches_kernel",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=4,5 | filters=4" +"input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_basic(
  "conv_basic",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_same_padding(
  "conv_same_padding",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4 | padding =1,1" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_multi_stride(
  "conv_multi_stride",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4 | stride=2,2" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_uneven_strides(
  "conv_uneven_strides",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4 | stride=3,3" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_uneven_strides2(
  "conv_uneven_strides2",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
    I("input") + input_base + "input_shape=2:4:4",
    I("conv2d_c1") + conv_base + "kernel_size = 2,2 | filters=2 | stride=1,2",
    I("act_1") + sigmoid_base,
    I("flatten", "type=flatten"),
    I("outputlayer") + fc_base + "unit = 10",
    I("act_2") + softmax_base
  }
);

INI conv_uneven_strides3(
  "conv_uneven_strides3",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
    I("input") + input_base + "input_shape=2:4:4",
    I("conv2d_c1") + conv_base + "kernel_size = 2,2 | filters=2 | stride=2,1",
    I("act_1") + sigmoid_base,
    I("flatten", "type=flatten"),
    I("outputlayer") + fc_base + "unit = 10",
    I("act_2") + softmax_base
  }
);


INI conv_same_padding_multi_stride(
  "conv_same_padding_multi_stride",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:5:3",
    I("conv2d_c1") + conv_base +
            "kernel_size = 3,3 | filters=4 | stride=2,2 | padding=1,1" + "input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1",
    I("flatten", "type=flatten")+"input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_no_loss_validate(
  "conv_no_loss_validate",
  {
    nn_base + "batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=4,5 | filters=4" +"input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("flatten", "type=flatten")+"input_layers=act_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI conv_none_loss_validate(
  "conv_none_loss_validate",
  {
    nn_base + "loss=none | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=4,5 | filters=4" +"input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("flatten", "type=flatten")+"input_layers=act_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_2") + softmax_base +"input_layers=outputlayer"
  }
);

INI pooling_max_same_padding(
  "pooling_max_same_padding",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=max | pool_size = 3,3 | padding =1,1" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_max_same_padding_multi_stride(
  "pooling_max_same_padding_multi_stride",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:3:5",
    I("pooling_1") + pooling_base +
            "pooling=max | pool_size = 3,3 | padding =1,1 | stride=2,2" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_max_valid_padding(
  "pooling_max_valid_padding",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=max | pool_size = 3,3 | padding =0,0" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_avg_same_padding(
  "pooling_avg_same_padding",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=average | pool_size = 3,3 | padding =1,1" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_avg_valid_padding(
  "pooling_avg_valid_padding",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=average | pool_size = 3,3 | padding =0,0" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_avg_same_padding_multi_stride(
  "pooling_avg_same_padding_multi_stride",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:3:5",
    I("pooling_1") + pooling_base +
            "pooling=average | pool_size = 3,3 | padding =1,1 | stride=2,2" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_global_avg(
  "pooling_global_avg",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=global_average" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI pooling_global_max(
  "pooling_global_max",
  {
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
        I("input") + input_base + "input_shape=2:5:3",
    I("pooling_1") + pooling_base +
            "pooling=global_max" + "input_layers=input",
    I("act_1") + sigmoid_base + "input_layers=pooling_1",
    I("flatten", "type=flatten")+ "input_layers=act_1",
    I("outputlayer") + fc_base + "unit = 10" + "input_layers=flatten",
    I("act_2") + softmax_base + "input_layers=outputlayer"
  }
);

INI preprocess_flip_validate(
  "preprocess_flip_validate",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("preprocess_flip") + preprocess_flip_base +
            "flip_direction=vertical" + "input_layers=input",
    I("conv2d_c1_layer") + conv_base + "kernel_size=3,4 | filters=2" +"input_layers=preprocess_flip",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("pool_1") + mnist_pooling+"input_layers=act_1",
    I("flatten", "type=flatten")+"input_layers=pool_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_3") + softmax_base +"input_layers=outputlayer"
  }
);

INI preprocess_translate_validate(
  "preprocess_translate_validate",
  {
    nn_base + "loss=cross | batch_size=3",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=2:4:5",
    I("preprocess_translate") + preprocess_translate_base +
            "random_translate=0.5" + "input_layers=input",
    I("conv2d_c1_layer") + conv_base + "kernel_size=3,4 | filters=2" +"input_layers=preprocess_translate",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("pool_1") + mnist_pooling+"input_layers=act_1",
    I("flatten", "type=flatten")+"input_layers=pool_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_3") + softmax_base +"input_layers=outputlayer"
  }
);

INI mnist_conv_cross_one_input = INI("mnist_conv_cross_one_input") + mnist_conv_cross + "model/batch_size=1";

INI fc_softmax_mse_distribute_validate(
  "fc_softmax_mse_distribute_validate",
  {
    nn_base + "loss=mse | batch_size = 3",
    sgd_base + "learning_rate = 1",
    I("input") + input_base + "input_shape = 1:5:5",
    I("dense") + fc_base + "unit = 3"+"activation=softmax"+"distribute=true"
  }
);

INI fc_softmax_cross_distribute_validate(
  "fc_softmax_cross_distribute_validate",
  {
    nn_base + "loss=cross | batch_size = 3",
    sgd_base + "learning_rate = 1",
    I("input") + input_base + "input_shape = 1:5:5",
    I("dense") + fc_base + "unit = 3"+"activation=softmax"+"distribute=true"
  }
);

INI fc_sigmoid_cross_distribute_validate(
  "fc_sigmoid_mse_distribute_validate",
  {
    nn_base + "loss=cross | batch_size = 3",
    sgd_base + "learning_rate = 1",
    I("input") + input_base + "input_shape = 1:5:5",
    I("dense") + fc_base + "unit = 3"+"activation=sigmoid"+"distribute=true"
  }
);

INI addition_resnet_like(
  "addition_resnet_like",
  {
    nn_base + "loss=mse | batch_size = 3",
    sgd_base + "learning_rate = 0.1",
    I("x") + input_base + "input_shape = 2:3:5",
    I("addition_a1") + conv_base
      + "filters=4 | kernel_size=3,3 | stride=2,2 | padding=1,1",
    I("addition_a2") + relu_base,
    I("addition_a3") + conv_base + "filters=4 | kernel_size=3,3 | padding=1,1",
    I("addition_b1") + conv_base
      + "filters=4 | kernel_size=1,1 | stride=2,2"
      + "input_layers=x",
    I("addition_c1", "type=addition | input_layers=addition_a3, addition_b1"),
    I("addition_c2", "type=flatten"),
    I("addition_c3") + fc_base + "unit=10",
    I("addition_c4") + softmax_base,
  }
);

INI lstm_basic(
  "lstm_basic",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:1:1",
    I("lstm") + lstm_base +
      "unit = 1" + "input_layers=input",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm"
  }
);

INI lstm_return_sequence(
  "lstm_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("lstm") + lstm_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm"
  }
);

INI lstm_return_sequence_with_batch(
  "lstm_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("lstm") + lstm_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm"
  }
);
INI rnn_basic(
  "rnn_basic",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:1:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn"
  }
);

INI rnn_return_sequences(
  "rnn_return_sequences",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input" + "return_sequences=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn"
  }
);

INI multi_lstm_return_sequence(
  "multi_lstm_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("lstm") + lstm_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true",
    I("lstm2") + lstm_base +
      "unit = 2" + "input_layers=lstm",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=lstm2"
  }
);

INI rnn_return_sequence_with_batch(
  "rnn_return_sequence_with_batch",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn"
  }
);

INI rnn_return_sequence_with_batch_n(
  "rnn_return_sequence_with_batch_n",
  {
    nn_base + "loss=mse | batch_size=2",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn"
  }
);

INI multi_rnn_return_sequence(
  "multi_rnn_return_sequence",
  {
    nn_base + "loss=mse | batch_size=1",
    sgd_base + "learning_rate = 0.1",
    I("input") + input_base + "input_shape=1:2:1",
    I("rnn") + rnn_base +
      "unit = 2" + "input_layers=input"+ "return_sequences=true",
    I("rnn2") + rnn_base +
      "unit = 2" + "input_layers=rnn",
    I("outputlayer") + fc_base + "unit = 1" + "input_layers=rnn2"
  }
);

INSTANTIATE_TEST_CASE_P(
  nntrainerModelAutoTests, nntrainerModelTest, ::testing::Values(
    mkModelTc(fc_sigmoid_mse, "3:1:1:10", 10),
    mkModelTc(fc_sigmoid_cross, "3:1:1:10", 10),
    mkModelTc(fc_relu_mse, "3:1:1:2", 10),
    mkModelTc(fc_bn_sigmoid_cross, "3:1:1:10", 10),
    mkModelTc(fc_bn_sigmoid_mse, "3:1:1:10", 10),
    mkModelTc(mnist_conv_cross, "3:1:1:10", 10),
    mkModelTc(mnist_conv_cross_one_input, "1:1:1:10", 10),
    /**< single conv2d layer test */
    mkModelTc(conv_1x1, "3:1:1:10", 10),
    mkModelTc(conv_input_matches_kernel, "3:1:1:10", 10),
    mkModelTc(conv_basic, "3:1:1:10", 10),
    mkModelTc(conv_same_padding, "3:1:1:10", 10),
    mkModelTc(conv_multi_stride, "3:1:1:10", 10),
    mkModelTc(conv_uneven_strides, "3:1:1:10", 10),
    mkModelTc(conv_uneven_strides2, "3:1:1:10", 10),
    mkModelTc(conv_uneven_strides3, "3:1:1:10", 10),
    mkModelTc(conv_same_padding_multi_stride, "3:1:1:10", 10),
    mkModelTc(conv_no_loss_validate, "3:1:1:10", 1),
    mkModelTc(conv_none_loss_validate, "3:1:1:10", 1),
    /**< single pooling layer test */
    mkModelTc(pooling_max_same_padding, "3:1:1:10", 10),
    mkModelTc(pooling_max_same_padding_multi_stride, "3:1:1:10", 10),
    mkModelTc(pooling_max_valid_padding, "3:1:1:10", 10),
    mkModelTc(pooling_avg_same_padding, "3:1:1:10", 10),
    mkModelTc(pooling_avg_same_padding_multi_stride, "3:1:1:10", 10),
    mkModelTc(pooling_avg_valid_padding, "3:1:1:10", 10),
    mkModelTc(pooling_global_avg, "3:1:1:10", 10),
    mkModelTc(pooling_global_max, "3:1:1:10", 10),
    /**< augmentation layer */
#if defined(ENABLE_DATA_AUGMENTATION_OPENCV)
    mkModelTc(preprocess_translate_validate, "3:1:1:10", 10),
#endif
    mkModelTc(preprocess_flip_validate, "3:1:1:10", 10),

    /**< Addition test */
    mkModelTc(addition_resnet_like, "3:1:1:10", 10),

    /// #1192 time distribution inference bug
    // mkModelTc(fc_softmax_mse_distribute_validate, "3:1:5:3", 1),
    // mkModelTc(fc_softmax_cross_distribute_validate, "3:1:5:3", 1),
    // mkModelTc(fc_sigmoid_cross_distribute_validate, "3:1:5:3", 1)
    mkModelTc(lstm_basic, "1:1:1:1", 10),
    mkModelTc(lstm_return_sequence, "1:1:2:1", 10),
    mkModelTc(lstm_return_sequence_with_batch, "2:1:2:1", 10),
    mkModelTc(multi_lstm_return_sequence, "1:1:1:1", 10),
    mkModelTc(rnn_basic, "1:1:1:1", 10),
    mkModelTc(rnn_return_sequences, "1:1:2:1", 10),
    mkModelTc(rnn_return_sequence_with_batch, "2:1:2:1", 10),
    mkModelTc(multi_rnn_return_sequence, "1:1:1:1", 10)
// / #if gtest_version <= 1.7.0
));
/// #else gtest_version > 1.8.0
// ), [](const testing::TestParamInfo<nntrainerModelTest::ParamType>& info){
//  return std::get<0>(info.param).getName();
// });
/// #end if */
// clang-format on

/**
 * @brief Read or save the model before initialize
 */
TEST(nntrainerModels, read_save_01_n) {
  nntrainer::NeuralNetwork NN;
  std::shared_ptr<nntrainer::Layer> layer =
    nntrainer::createLayer(nntrainer::InputLayer::type);
  layer->setProperty(
    {"input_shape=1:1:62720", "normalization=true", "bias_initializer=zeros"});
  std::shared_ptr<nntrainer::LayerNode> layer_node =
    std::make_unique<nntrainer::LayerNode>(layer);

  EXPECT_NO_THROW(NN.addLayer(layer_node));
  EXPECT_NO_THROW(NN.setProperty({"loss=mse"}));

  EXPECT_THROW(NN.readModel(), std::runtime_error);
  EXPECT_THROW(NN.saveModel(), std::runtime_error);

  EXPECT_EQ(NN.compile(), ML_ERROR_NONE);

  EXPECT_THROW(NN.readModel(), std::runtime_error);
  EXPECT_THROW(NN.saveModel(), std::runtime_error);
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
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
