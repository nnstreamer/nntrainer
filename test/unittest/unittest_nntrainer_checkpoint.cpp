/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        unittest_nntrainer_checkpoint.cpp
 * @date        29 March 2023
 * @brief       Unit test for checkpoint
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jiho Chu <jiho.chu@samsung.com>
 * @bug         No known bugs
 */

#include <gmock/gmock-cardinalities.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "activation_layer.h"
#include "fc_layer.h"
#include "input_layer.h"
#include "mse_loss_layer.h"
#include "neuralnet.h"
#include "util_func.h"

#include <nntrainer_test_util.h>

/**
 * @brief Mock class for InputLayer
 */
class MockInputLayer : public nntrainer::InputLayer {
public:
  MockInputLayer() {
    ON_CALL(*this, forwarding)
      .WillByDefault([&](nntrainer::RunLayerContext &context, bool training) {
        return nntrainer::InputLayer::forwarding(context, training);
      });
  }

  /**
   * @brief Mock method for forwarding
   */
  MOCK_METHOD(void, forwarding,
              (nntrainer::RunLayerContext & context, bool training),
              (override));
};

/**
 * @brief Mock class for FullyConnectedLayer
 */
class MockFullyConnectedLayer : public nntrainer::FullyConnectedLayer {
public:
  MockFullyConnectedLayer() {
    ON_CALL(*this, forwarding)
      .WillByDefault([&](nntrainer::RunLayerContext &context, bool training) {
        return nntrainer::FullyConnectedLayer::forwarding(context, training);
      });
  }

  /**
   * @brief Mock method for forwarding
   */
  MOCK_METHOD(void, forwarding,
              (nntrainer::RunLayerContext & context, bool training),
              (override));
};

/**
 * @brief Mock class for MSE Loss Layer
 */
class MockMSELossLayer : public nntrainer::MSELossLayer {
public:
  MockMSELossLayer() {
    ON_CALL(*this, forwarding)
      .WillByDefault([&](nntrainer::RunLayerContext &context, bool training) {
        return nntrainer::MSELossLayer::forwarding(context, training);
      });
  }

  /**
   * @brief Mock method for forwarding
   */
  MOCK_METHOD(void, forwarding,
              (nntrainer::RunLayerContext & context, bool training),
              (override));
};

/**
 * @brief Checkpoint test class
 */
class CheckPointTest : public ::testing::Test {
public:
  /**
   * @brief Construct a new CheckPointTest object
   */
  static void SetUpTestCase() {
    auto &ac = nntrainer::AppContext::Global();
    ac.registerFactory(nntrainer::createLayer<MockInputLayer>, "mock_input");
    ac.registerFactory(nntrainer::createLayer<MockFullyConnectedLayer>,
                       "mock_fully_connected");
    ac.registerFactory(nntrainer::createLayer<MockMSELossLayer>, "mock_mse");
  }

  /**
   * @brief Destroy the CheckPointTest object
   */
  static void TearDownTestCase() {}

  /**
   * @brief Construct a new CheckPointTest object
   */
  void SetUp(void) {
    nn = std::make_shared<nntrainer::NeuralNetwork>();

    graph = makeGraph({
      {"mock_input", {"name=in", "input_shape=1:1:3"}},
      {"mock_fully_connected",
       {"name=fc1", "input_layers=in", "unit=10", "activation=relu"}},
      {"mock_fully_connected",
       {"name=fc2", "input_layers=fc1", "unit=10", "activation=relu"}},
      {"mock_fully_connected",
       {"name=fc3", "input_layers=fc2", "unit=2", "activation=sigmoid"}},
      {"mock_fully_connected",
       {"name=fc4", "input_layers=fc3", "unit=2", "activation=sigmoid"}},
      {"mock_fully_connected",
       {"name=fc5", "input_layers=fc4", "unit=2", "activation=sigmoid"}},
      {"mock_fully_connected",
       {"name=fc6", "input_layers=fc5", "unit=2", "activation=sigmoid"}},
      {"mock_fully_connected",
       {"name=fc7", "input_layers=fc6", "unit=2", "activation=sigmoid"}},
      {"mock_fully_connected",
       {"name=fc8", "input_layers=fc7", "unit=2", "activation=sigmoid"}},
      {"mock_fully_connected",
       {"name=fc9", "input_layers=fc8", "unit=2", "activation=sigmoid"}},
      {"mock_fully_connected",
       {"name=fc10", "input_layers=fc9", "unit=2", "activation=sigmoid"}},
      {"mock_mse", {"name=loss", "input_layers=fc10"}},
    });

    for (auto &node : graph) {
      nn->addLayer(node);
    }

    auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
    nn->setOptimizer(std::move(optimizer));
  }

  /**
   * @brief Destroy the CheckPointTest object
   */
  void TearDown(void) {}

  /**
   * @brief Generate model
   */
  void generateModel(unsigned int checkpoint_len, bool summary = false) {
    nn->setProperty({"checkpoint_len=" + std::to_string(checkpoint_len)});

    nn->compile();
    nn->initialize();
    nn->allocate(nntrainer::ExecutionMode::TRAIN);

    if (summary)
      nn->summarize(std::cout, ml_train_summary_type_e::ML_TRAIN_SUMMARY_LAYER);
  }

  /**
   * @brief Generate Tensor
   */
  nntrainer::sharedConstTensors
  generateTensors(std::vector<nntrainer::TensorDim> dims) {
    auto toSharedTensors = [](const std::vector<nntrainer::Tensor> &ts) {
      nntrainer::sharedConstTensors sts;
      sts.reserve(ts.size());
      std::transform(ts.begin(), ts.end(), std::back_inserter(sts),
                     [](const auto &ts) { return MAKE_SHARED_TENSOR(ts); });
      return sts;
    };

    auto tensors = std::vector<nntrainer::Tensor>(dims.begin(), dims.end());

    return toSharedTensors(tensors);
  }

  /**
   * @brief test checkpoint in layer property
   */
  void testCheckPoint(std::vector<nntrainer::CheckPointType> checkpoints) {
    int idx = 0;
    auto graph = nn->getNetworkGraph();
    std::for_each(graph.cbegin(), graph.cend(), [&](auto &node) {
      EXPECT_EQ(node->getCheckPoint().get(), checkpoints[idx++]);
    });
  }

  std::shared_ptr<nntrainer::NeuralNetwork> nn; //< Neural Network
  nntrainer::GraphRepresentation graph;         //< Graph representation
};

/**
 * @brief Set checkpoint with checkpoint_len = 0
 */
TEST_F(CheckPointTest, default_checkpoint) {
  generateModel(0);
  testCheckPoint({
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
  });
}

/**
 * @brief Set checkpoint with checkpoint_len = 1
 */
TEST_F(CheckPointTest, set_length_1) {
  generateModel(1);
  testCheckPoint({
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
  });
}

/**
 * @brief Set checkpoint with checkpoint_len = 2
 */
TEST_F(CheckPointTest, set_length_2) {
  generateModel(2);
  testCheckPoint({
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::NONCHECK_UNLOAD,
    nntrainer::CheckPointType::NONCHECK_UNLOAD,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::NONCHECK_UNLOAD,
    nntrainer::CheckPointType::NONCHECK_UNLOAD,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::NONCHECK_UNLOAD,
    nntrainer::CheckPointType::NONCHECK_UNLOAD,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::CHECKPOINTED,
    nntrainer::CheckPointType::NONCHECK_UNLOAD,
    nntrainer::CheckPointType::NONCHECK_UNLOAD,
    nntrainer::CheckPointType::CHECKPOINTED,
  });
}

/**
 * @brief Check training wich checkpoint length = 0
 */
TEST_F(CheckPointTest, train_length_0) {
  EXPECT_CALL(*dynamic_cast<MockInputLayer *>(graph[0]->getLayer()), forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[1]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[2]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[3]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[4]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[5]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[6]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[7]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[8]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[9]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[10]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockMSELossLayer *>(graph[11]->getLayer()),
              forwarding)
    .Times(1);

  generateModel(0);

  auto inputs = generateTensors(nn->getInputDimension());
  auto labels = generateTensors(nn->getOutputDimension());

  auto out = nn->forwarding(inputs, labels);
  nn->backwarding(1);
}

/**
 * @brief Check training wich checkpoint length = 1
 */
TEST_F(CheckPointTest, train_length_1) {
  EXPECT_CALL(*dynamic_cast<MockInputLayer *>(graph[0]->getLayer()), forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[1]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[2]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[3]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[4]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[5]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[6]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[7]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[8]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[9]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[10]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockMSELossLayer *>(graph[11]->getLayer()),
              forwarding)
    .Times(1);

  generateModel(1);

  auto inputs = generateTensors(nn->getInputDimension());
  auto labels = generateTensors(nn->getOutputDimension());

  auto out = nn->forwarding(inputs, labels);
  nn->backwarding(1);
}

/**
 * @brief Check training wich checkpoint length = 2
 */
TEST_F(CheckPointTest, train_length_2) {
  EXPECT_CALL(*dynamic_cast<MockInputLayer *>(graph[0]->getLayer()), forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[1]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[2]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[3]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[4]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[5]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[6]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[7]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[8]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[9]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[10]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockMSELossLayer *>(graph[11]->getLayer()),
              forwarding)
    .Times(1);

  generateModel(2);

  auto inputs = generateTensors(nn->getInputDimension());
  auto labels = generateTensors(nn->getOutputDimension());

  auto out = nn->forwarding(inputs, labels);
  nn->backwarding(1);
}

/**
 * @brief Check training wich checkpoint length = 3
 */
TEST_F(CheckPointTest, train_length_3) {
  EXPECT_CALL(*dynamic_cast<MockInputLayer *>(graph[0]->getLayer()), forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[1]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[2]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[3]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[4]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[5]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[6]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[7]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[8]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[9]->getLayer()),
              forwarding)
    .Times(2);
  EXPECT_CALL(*dynamic_cast<MockFullyConnectedLayer *>(graph[10]->getLayer()),
              forwarding)
    .Times(1);
  EXPECT_CALL(*dynamic_cast<MockMSELossLayer *>(graph[11]->getLayer()),
              forwarding)
    .Times(1);

  generateModel(3);

  auto inputs = generateTensors(nn->getInputDimension());
  auto labels = generateTensors(nn->getOutputDimension());

  auto out = nn->forwarding(inputs, labels);
  nn->backwarding(1);
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
