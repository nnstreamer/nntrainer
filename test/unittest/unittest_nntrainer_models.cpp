// SPDX-License-Identifier: Apache-2.0
/* Copyright (C) 2020 Jihoon Lee <jihoon.it.lee@samsung.com>
 *
 * @file	unittest_nntrainer_models.cpp
 * @date	19 Oct 2020
 * @brief	Model multi iteration, itegrated test
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jihoon Lee <jihoon.it.lee@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <layer.h>
#include <neuralnet.h>
#include <weight.h>

#include "nntrainer_test_util.h"

/********************************************************
 * Watcher Classes                                      *
 ********************************************************/

using NodeType = nntrainer::LayerNode;
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
  NodeWatcher() {}

  /**
   * @brief Construct a new Node Watcher object
   *
   * @param node node to watch.
   */
  NodeWatcher(const NodeType &node) : node(node) {
    unsigned int num_weights = node.layer->getNumWeights();
    node.layer->setTrainable(true);

    for (unsigned int i = 0; i < num_weights; ++i) {
      const nntrainer::Weight &w = node.layer->weightAt(i);
      expected_weights.push_back(w);
    }

    expected_output = nntrainer::Tensor(node.layer->getOutputDimension()[0]);
    expected_dx = nntrainer::Tensor(node.layer->getInputDimension()[0]);
  }

  /**
   * @brief clones from expected weights to node->weights
   *
   */
  void readLayerWeight(std::ifstream &f) {
    for (unsigned int i = 0; i < node.layer->getNumWeights(); ++i) {
      /// @note below is harrasing the fact the tensor shares same base memory
      node.layer->weightAt(i).getVariable().read(f);
    }
  }

  /**
   * @brief forward node with verifying inputs/weights/outputs
   *
   * @param in input tensor
   * @param iteration iteration
   * @return nntrainer::sharedConstTensor
   */
  void forward(int iteration);

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
   * @param should_verify should verify the inputs/gradients/outputs
   * @return nntrainer::sharedConstTensor
   */
  void backward(int iteration, bool should_verify = true);

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
  float getLoss() { return node.layer->getLoss(); }

  /**
   * @brief read Node
   *
   * @param in input file stream
   */
  void read(std::ifstream &in);

private:
  NodeType node;
  nntrainer::Tensor expected_output;
  nntrainer::Tensor expected_dx;
  std::vector<nntrainer::Weight> expected_weights;
};

class GraphWatcher {
public:
  using WatchedFlatGraph = std::vector<NodeWatcher>;
  GraphWatcher(const std::string &config);

  void compareFor(const std::string &reference,
                  const nntrainer::TensorDim &label_shape,
                  unsigned int iterations);

private:
  std::array<nntrainer::Tensor, 2>
  prepareData(std::ifstream &f, const nntrainer::TensorDim &label_dim);

  void readIteration(std::ifstream &f);

  nntrainer::NeuralNetwork nn;
  WatchedFlatGraph nodes;
  NodeWatcher loss_node;
  float expected_loss;
};

void NodeWatcher::read(std::ifstream &in) {
  expected_output.read(in);
  expected_dx.read(in);

  /// @note below is harrasing the fact the tensor shares same base memory
  /// it should better be getGraidentRef() or somewhat equivalent
  for (auto &i : expected_weights) {
    if (i.getTrainable())
      i.getGradient().read(in);
  }

  for (auto &i : expected_weights) {
    i.getVariable().read(in);
  }
}

void NodeWatcher::verifyWeight(const std::string &error_msg) {
  for (unsigned int i = 0; i < expected_weights.size(); ++i) {
    verify(node.layer->weightAt(i).getVariable(),
           expected_weights[i].getVariable(),
           error_msg + " " + node.layer->weightAt(i).getName() + " weight");
  }
}

void NodeWatcher::verifyGrad(const std::string &error_msg) {
  for (unsigned int i = 0; i < expected_weights.size(); ++i) {
    auto weight = node.layer->weightAt(i);
    if (weight.getTrainable()) {
      verify(weight.getGradient(), expected_weights[i].getGradient(),
             error_msg + " " + weight.getName() + " grad");
    }
  }
}

void NodeWatcher::forward(int iteration) {
  std::stringstream ss;
  ss << "forward failed at " << node.layer->getName() << " at iteration "
     << iteration;
  std::string err_msg = ss.str();

  std::vector<nntrainer::Tensor> out = node.layer->getHidden();

  verify(out[0], expected_output, err_msg + " at output");
}

nntrainer::sharedConstTensors
NodeWatcher::lossForward(nntrainer::sharedConstTensors pred,
                         nntrainer::sharedConstTensors answer, int iteration) {
  std::stringstream ss;
  ss << "loss failed at " << node.layer->getName() << " at iteration "
     << iteration;
  std::string err_msg = ss.str();

  nntrainer::sharedConstTensors out =
    std::static_pointer_cast<nntrainer::LossLayer>(node.layer)
      ->forwarding(pred, answer);

  return out;
}

void NodeWatcher::backward(int iteration, bool should_verify) {
  std::stringstream ss;
  ss << "backward failed at " << node.layer->getName() << " at iteration "
     << iteration;
  std::string err_msg = ss.str();

  std::vector<nntrainer::Tensor> out = node.layer->getGradient();

  if (should_verify) {
    verify(out[0], expected_dx, err_msg);
    verifyGrad(err_msg);
    verifyWeight(err_msg);
  }
}

GraphWatcher::GraphWatcher(const std::string &config) {
  nn = nntrainer::NeuralNetwork();
  if (nn.loadFromConfig(config)) {
    throw std::invalid_argument("load from config failed!");
  };

  if (nn.compile()) {
    throw std::invalid_argument("initiation failed");
  };

  if (nn.initialize()) {
    throw std::invalid_argument("initiation failed");
  };

  if (nn.assignMem()) {
    throw std::invalid_argument("assign Memory failed");
  };

  NetworkGraphType model_graph = nn.getNetworkGraph();

  std::vector<NodeType> graph = model_graph.getSorted();

  for (auto it = graph.begin(); it != graph.end() - 1; ++it) {
    nodes.push_back(NodeWatcher(*it));
  }

  loss_node = NodeWatcher(graph.back());
}

void GraphWatcher::compareFor(const std::string &reference,
                              const nntrainer::TensorDim &label_shape,
                              const unsigned int iterations) {
  std::ifstream ref(reference, std::ios_base::in | std::ios_base::binary);

  if (!ref.good()) {
    std::stringstream ss;
    ss << "ref is bad! ref path: " << reference;
    throw std::runtime_error(ss.str().c_str());
  }

  auto data = prepareData(ref, label_shape);

  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
    nntrainer::sharedConstTensors input = {
      MAKE_SHARED_TENSOR(std::get<0>(data).clone())};
    nntrainer::sharedConstTensors label = {
      MAKE_SHARED_TENSOR(std::get<1>(data).clone())};

    readIteration(ref);

    nn.forwarding(input, label);
    EXPECT_NEAR(expected_loss, loss_node.getLoss(), nntrainer::Tensor::epsilon);

    for (auto it = nodes.begin(); it != nodes.end() - 1; ++it) {
      it->forward(iteration);
    }

    nn.getNetworkGraph().backwarding(label, iteration);

    for (auto it = nodes.rbegin(); it != nodes.rend() - 1; it++)
      it->backward(iteration);
  }
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
 * @param IniTestWrapper ini data
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 */
class nntrainerModelTest
  : public ::testing::TestWithParam<std::tuple<
      const IniTestWrapper, const nntrainer::TensorDim, const unsigned int>> {

protected:
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
  IniTestWrapper ini;
};

/**
 * @brief check given ini is failing/suceeding at load
 */
TEST_P(nntrainerModelTest, model_test) {
  GraphWatcher g(getIniName());

  g.compareFor(getGoldenName(), getLabelDim(), getIteration());

  /// add stub test for tcm
  EXPECT_EQ(std::get<0>(GetParam()), std::get<0>(GetParam()));
}

/**
 * @brief helper function to make model testcase
 *
 * @param const char * name of the ini and test. the tester generates name.ini
 * and try to read name.info
 * @param IniTestWrapper::Sections ini data
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 */
auto mkModelTc(const IniTestWrapper &ini, const std::string &label_dim,
               const unsigned int iteration) {
  return std::make_tuple(ini, nntrainer::TensorDim(label_dim), iteration);
}

/********************************************************
 * Actual Test                                          *
 ********************************************************/

static IniSection nn_base("model", "type = NeuralNetwork");
static std::string input_base = "type = input";
static std::string fc_base = "type = Fully_connected";
static std::string conv_base = "type = conv2d | stride = 1,1 | padding = 0,0";
static std::string pooling_base = "type = pooling2d | padding = 0,0";

static std::string adam_base = "optimizer=adam | beta1 = 0.9 | beta2 = 0.999 | "
                               "epsilon = 1e-7";

static IniSection act_base("activation", "Type = Activation");
static IniSection softmax_base = act_base + "Activation = softmax";
static IniSection sigmoid_base = act_base + "Activation = sigmoid";
static IniSection relu_base = act_base + "Activation = relu";
static IniSection bn_base("bn", "Type=batch_normalization");

using I = IniSection;
using INI = IniTestWrapper;

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
  {nn_base + "learning_rate=1 | optimizer=sgd | loss=mse | batch_size = 3",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 5" + "input_layers=input",
   I("act") + sigmoid_base + "input_layers=dense",
   I("dense_1") + fc_base + "unit = 10" + "input_layers=act",
   I("act_1") + softmax_base + "input_layers=dense_1"});

INI fc_sigmoid_cross =
  INI("fc_sigmoid_cross") + fc_sigmoid_mse + "model/loss=cross";

INI fc_relu_mse(
  "fc_relu_mse",
  {nn_base + "Learning_rate=0.1 | Optimizer=sgd | Loss=mse | batch_size = 3",
   I("input") + input_base + "input_shape = 1:1:3",
   I("dense") + fc_base + "unit = 10" + "input_layers=input",
   I("act") + relu_base + "input_layers=dense",
   I("dense_1") + fc_base + "unit = 2" + "input_layers=act",
   I("act_1") + sigmoid_base + "input_layers=dense" + "input_layers=dense_1"});

INI fc_bn_sigmoid_cross(
  "fc_bn_sigmoid_cross",
  {nn_base + "learning_rate=1 | optimizer=sgd | loss=cross | batch_size = 3",
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
    nn_base + "learning_rate=0.1 | optimizer=sgd | loss=cross | batch_size=3",
    I("input") + input_base + "input_shape=2:4:5",
    I("conv2d_c1_layer") + conv_base + "kernel_size=3,4 | filters=2" +"input_layers=input",
    I("act_1") + sigmoid_base +"input_layers=conv2d_c1_layer",
    I("pool_1") + mnist_pooling+"input_layers=act_1",
    I("flatten", "type=flatten")+"input_layers=pool_1" ,
    I("outputlayer") + fc_base + "unit = 10" +"input_layers=flatten",
    I("act_3") + softmax_base +"input_layers=outputlayer"
  }
);

INI mnist_conv_cross_one_input = INI("mnist_conv_cross_one_input") + mnist_conv_cross + "model/batch_size=1";

INSTANTIATE_TEST_CASE_P(
  nntrainerModelAutoTests, nntrainerModelTest, ::testing::Values(
    mkModelTc(fc_sigmoid_mse, "3:1:1:10", 10),
    mkModelTc(fc_sigmoid_cross, "3:1:1:10", 10),
    mkModelTc(fc_relu_mse, "3:1:1:2", 10),
    mkModelTc(fc_bn_sigmoid_cross, "3:1:1:10", 10),
    mkModelTc(fc_bn_sigmoid_mse, "3:1:1:10", 10),
    mkModelTc(mnist_conv_cross, "3:1:1:10", 10),
    mkModelTc(mnist_conv_cross_one_input, "1:1:1:10", 10)
// / #if gtest_version <= 1.7.0
));
/// #else gtest_version > 1.8.0
// ), [](const testing::TestParamInfo<nntrainerModelTest::ParamType>& info){
//  return std::get<0>(info.param).getName();
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
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
