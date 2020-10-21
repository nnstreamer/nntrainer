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

using NodeType = nntrainer::NeuralNetwork::NodeType;
using FlatGraphType = nntrainer::NeuralNetwork::FlatGraphType;

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
    std::cout << "current " << actual << "expected " << expected;
    throw std::invalid_argument(error_msg.c_str());
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
    unsigned int num_weights = node->getNumWeights();
    node->setTrainable(false);

    expected_input = nntrainer::Tensor(node->getInputDimension());

    for (unsigned int i = 0; i < num_weights; ++i) {
      const nntrainer::Weight &w = node->weightAt(i);
      expected_weights.push_back(w);
    }

    expected_output = nntrainer::Tensor(node->getOutputDimension());
    expected_dx = nntrainer::Tensor(node->getInputDimension());
  }

  /**
   * @brief clones from expected weights to node->weights
   *
   */
  void cloneWeightsFromExpected() {
    for (unsigned int i = 0; i < expected_weights.size(); ++i) {
      node->weightAt(i) = expected_weights[i];
    }
  }

  /**
   * @brief forward node with verifying inputs/weights/outputs
   *
   * @param in input tensor
   * @param iteration iteration
   * @return nntrainer::sharedConstTensor
   */
  nntrainer::sharedConstTensor forward(nntrainer::sharedConstTensor in,
                                       int iteration);

  /**
   * @brief forward loss node with verifying inputs/weights/outputs
   *
   * @param pred tensor predicted from the graph
   * @param answer label tensor
   * @param iteration iteration
   * @return nntrainer::sharedConstTensor
   */
  nntrainer::sharedConstTensor lossForward(nntrainer::sharedConstTensor pred,
                                           nntrainer::sharedConstTensor answer,
                                           int iteration);

  /**
   * @brief backward pass of the node with verifying inputs/gradients/outputs
   *
   * @param deriv dervatives
   * @param iteration iteration
   * @param should_verify should verify the inputs/gradients/outputs
   * @return nntrainer::sharedConstTensor
   */
  nntrainer::sharedConstTensor backward(nntrainer::sharedConstTensor deriv,
                                        int iteration,
                                        bool should_verify = true);

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
  float getLoss() { return node->getLoss(); }

  /**
   * @brief read Node
   *
   * @param in input file stream
   */
  void read(std::ifstream &in);

private:
  NodeType node;
  nntrainer::Tensor expected_input;
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
  void readIteration(std::ifstream &f);

  nntrainer::NeuralNetwork nn;
  WatchedFlatGraph nodes;
  NodeWatcher loss_node;
  float expected_loss;
};

void NodeWatcher::read(std::ifstream &in) {
  expected_input.read(in);

  /// @note below is harrasing the fact the tensor shares same base memory
  /// it should better be getVariableRef() or somewhat equivalent
  for (auto &i : expected_weights) {
    i.getVariable().read(in);
  }

  for (auto &i : expected_weights) {
    i.getGradient().read(in);
  }

  expected_output.read(in);
  expected_dx.read(in);
}

void NodeWatcher::verifyWeight(const std::string &error_msg) {
  for (unsigned int i = 0; i < expected_weights.size(); ++i) {
    verify(node->weightAt(i).getVariable(), expected_weights[i].getVariable(),
           error_msg + " " + node->weightAt(i).getName() + "weight");
  }
}

void NodeWatcher::verifyGrad(const std::string &error_msg) {
  for (unsigned int i = 0; i < expected_weights.size(); ++i) {
    verify(node->weightAt(i).getGradient(), expected_weights[i].getGradient(),
           error_msg + " " + node->weightAt(i).getName() + "grad");
  }
}

nntrainer::sharedConstTensor
NodeWatcher::forward(nntrainer::sharedConstTensor in, int iteration) {
  std::stringstream ss;
  ss << "forward failed at " << node->getName() << " at iteration "
     << iteration;
  std::string err_msg = ss.str();

  verify(*in, expected_input, err_msg + " at input ");
  nntrainer::sharedConstTensor out = node->forwarding(in);
  verify(*out, expected_output, err_msg + " at output ");
  return out;
}

nntrainer::sharedConstTensor
NodeWatcher::lossForward(nntrainer::sharedConstTensor pred,
                         nntrainer::sharedConstTensor answer, int iteration) {
  std::stringstream ss;
  ss << "loss failed at " << node->getName() << " at iteration " << iteration;
  std::string err_msg = ss.str();

  nntrainer::sharedConstTensor out =
    std::static_pointer_cast<nntrainer::LossLayer>(node)->forwarding(pred,
                                                                     answer);

  return out;
}

nntrainer::sharedConstTensor
NodeWatcher::backward(nntrainer::sharedConstTensor deriv, int iteration,
                      bool should_verify) {
  std::stringstream ss;
  ss << "backward failed at " << node->getName() << " at iteration "
     << iteration;
  std::string err_msg = ss.str();

  nntrainer::sharedConstTensor out = node->backwarding(deriv, iteration);

  if (should_verify) {
    verify(*out, expected_dx, err_msg);
    verifyGrad(err_msg);
  }

  auto opt = node->getOptimizer();
  if (opt) {
    opt->apply_gradients(node->getWeights(), node->getNumWeights(), iteration);
  }

  return out;
}

GraphWatcher::GraphWatcher(const std::string &config) {
  nn = nntrainer::NeuralNetwork();
  nn.loadFromConfig(config);
  nn.init();

  FlatGraphType graph = nn.getFlatGraph();

  for (auto it = graph.begin(); it != graph.end() - 1; ++it) {
    nodes.push_back(NodeWatcher(*it));
  }

  loss_node = NodeWatcher(graph.back());
}

void GraphWatcher::compareFor(const std::string &reference,
                              const nntrainer::TensorDim &label_shape,
                              const unsigned int iterations) {
  std::ifstream ref(reference, std::ios_base::in | std::ios_base::binary);

  if (ref.bad()) {
    throw std::runtime_error("ref is bad!");
  }

  nntrainer::Tensor in(nn.getInputDimension());
  nntrainer::Tensor lb(label_shape);

  in.read(ref);
  lb.read(ref);

  auto prepareInitialWeight = [this]() {
    std::for_each(nodes.begin(), nodes.end(),
                  [](NodeWatcher &n) { n.cloneWeightsFromExpected(); });
  };

  auto matchWeightAfterUpdation = [this]() {
    std::for_each(nodes.begin(), nodes.end(), [](NodeWatcher &n) {
      n.verifyWeight("weight is diffrent after updation, check optimizer");
    });
  };

  for (unsigned int iteration = 1; iteration <= iterations; ++iteration) {
    nntrainer::sharedConstTensor input = MAKE_SHARED_TENSOR(in.clone());
    nntrainer::sharedConstTensor label = MAKE_SHARED_TENSOR(lb.clone());

    readIteration(ref);
    iteration == 1 ? prepareInitialWeight() : matchWeightAfterUpdation();

    /// forward pass
    for (auto &i : nodes)
      input = i.forward(input, iteration);

    loss_node.lossForward(input, label, iteration);
    EXPECT_NEAR(expected_loss, loss_node.getLoss(), nntrainer::Tensor::epsilon);

    /// backward pass and update weights
    nntrainer::sharedConstTensor output =
      loss_node.backward(label, iteration, false);
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++)
      output = it->backward(output, iteration);
  }

  /// note that last weight update is not checked up. this need to be fixed
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
 * @param const char * name of the ini and test. the tester generates name.ini
 * and try to read name.info
 * @param IniTestWrapper::Sections ini data
 * @param nntrainer::TensorDim label dimension
 * @param int Iteration
 */
class nntrainerModelTest
  : public ::testing::TestWithParam<
      std::tuple<const char *, const IniTestWrapper::Sections,
                 const nntrainer::TensorDim, const unsigned int>> {

protected:
  virtual void SetUp() {
    auto param = GetParam();
    name = std::string(std::get<0>(param));
    std::cout << "starting test case : " << name << "\n\n";

    auto sections = std::get<1>(param);
    ini = IniTestWrapper(name, sections);

    label_dim = std::get<2>(param);
    iteration = std::get<3>(param);
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
auto mkModelTc(const char *name, const IniTestWrapper::Sections &vec,
               const std::string &label_dim, const unsigned int iteration) {
  return std::make_tuple(name, vec, nntrainer::TensorDim(label_dim), iteration);
}

/********************************************************
 * Actual Test                                          *
 ********************************************************/

static IniSection nn_base("model", "Type = NeuralNetwork");
static IniSection input_base("input", "Type = input");
static IniSection fc_base("fc", "Type = Fully_connected");

static IniSection act_base("activation", "Type = Activation");
static IniSection softmax = act_base + "Activation = softmax";
static IniSection sigmoid = act_base + "Activation = sigmoid";
static IniSection relu = act_base + "Activation = relu";

using I = IniSection;

/**
 * This is just a wrapper for an ini file with save / erase attached.
 * for example, fc_softmax_mse contains following ini file representation as a
 * series of IniSection
 *
 * [Model]
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
 * Unit = 10
 *
 * [dense_1]
 * Type = fully_connected
 * Unit = 10
 *
 * [dense_2]
 * Type = fully_connected
 * Unit = 10
 *
 * [activation]
 * Type = Activation
 * Activation = softmax
 */
IniTestWrapper::Sections fc_softmax_mse{
  nn_base + "Learning_rate=1 | Optimizer=sgd | Loss=mse | batch_size = 3",
  I("input") + input_base + "input_shape = 1:1:3",
  I("dense") + fc_base + "unit = 5",
  I("dense_1") + fc_base + "unit = 5",
  I("dense_2") + fc_base + "unit = 10",
  softmax};

IniTestWrapper::Sections fc_sigmoid_mse{
  nn_base + "Learning_rate=1 | Optimizer=sgd | Loss=mse | batch_size = 3",
  I("input") + input_base + "input_shape = 1:1:3",
  I("dense") + fc_base + "unit = 10",
  I("dense_1") + fc_base + "unit = 10",
  I("dense_2") + fc_base + "unit = 2",
  sigmoid};

IniTestWrapper::Sections fc_relu_mse{
  nn_base + "Learning_rate=0.001 | Optimizer=sgd | Loss=mse | batch_size = 3",
  I("input") + input_base + "input_shape = 1:1:3",
  I("dense") + fc_base + "unit = 10",
  I("act") + relu,
  I("dense_1") + fc_base + "unit = 10",
  I("act_1") + relu,
  I("dense_2") + fc_base + "unit = 2",
  I("act_2") + relu};

// clang-format off
INSTANTIATE_TEST_CASE_P(
  nntrainerModelAutoTests, nntrainerModelTest, ::testing::Values(
mkModelTc("fc_softmax_mse", fc_softmax_mse, "3:1:1:10", 10),
mkModelTc("fc_relu_mse", fc_relu_mse, "3:1:1:2", 10)
/// #if gtest_version <= 1.7.0
));
/// #else gtest_version > 1.8.0
// ), [](const testing::TestParamInfo<nntrainerModelTest::ParamType>& info){
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
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
