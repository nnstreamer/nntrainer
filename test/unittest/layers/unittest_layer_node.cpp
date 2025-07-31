// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file        unittest_layer_node.cpp
 * @date        02 July 2021
 * @brief       Unit test utility for layer node
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>

#include <identity_layer.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <tensor_dim.h>
#include <var_grad.h>

/**
 * @brief createLayerNode with unknown layer type
 */
TEST(nntrainer_LayerNode, createLayerNode_01_n) {
  EXPECT_THROW(nntrainer::createLayerNode(ml::train::LayerType::LAYER_UNKNOWN),
               std::invalid_argument);
}

/**
 * @brief createLayerNode with unknown layer string
 */
TEST(nntrainer_LayerNode, createLayerNode_02_n) {
  EXPECT_THROW(nntrainer::createLayerNode("unknown_layer"),
               std::invalid_argument);
}

/**
 * @brief createLayerNode with empty string
 */
TEST(nntrainer_LayerNode, createLayerNode_03_n) {
  EXPECT_THROW(nntrainer::createLayerNode(""), std::invalid_argument);
}

/**
 * @brief test distribute property
 */
TEST(nntrainer_LayerNode, setDistribute_01_p) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_EQ(false, lnode->getDistribute());

  EXPECT_NO_THROW(lnode->setProperty({"distribute=true"}));

  EXPECT_EQ(true, lnode->getDistribute());
}

/**
 * @brief test flatten property
 */
TEST(nntrainer_LayerNode, setFlatten_01_p) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"flatten=true"}));
}

/**
 * @brief finalize with empty input shape and input dimension
 */
TEST(nntrainer_LayerNode, finalize_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_ANY_THROW(lnode->finalize());
}

/**
 * @brief finalize with empty input info
 */
TEST(nntrainer_LayerNode, finalize_02_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"name=abc"}));

  EXPECT_THROW(lnode->finalize(), std::invalid_argument);
}

/**
 * @brief finalize with empty name
 */
TEST(nntrainer_LayerNode, finalize_03_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"input_shape=1:1:1"}));

  EXPECT_ANY_THROW(lnode->finalize());
}

/**
 * @brief finalize
 */
TEST(nntrainer_LayerNode, finalize_04_p) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"input_shape=1:1:1", "name=abc"}));
  EXPECT_NO_THROW(lnode->finalize());
}

/**
 * @brief finalize twice
 */
TEST(nntrainer_LayerNode, finalize_05_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;
  nntrainer::Var_Grad input =
    nntrainer::Var_Grad(nntrainer::TensorDim({1, 1, 1, 1}),
                        nntrainer::Initializer::NONE, true, false, "dummy");

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"input_shape=1:1:1", "name=abc"}));
  EXPECT_NO_THROW(lnode->finalize());
  EXPECT_NO_THROW(lnode->configureRunContext({}, {&input}, {}, {}));
  EXPECT_THROW(lnode->finalize(), std::runtime_error);
}

/**
 * @brief finalize with invalid input_shape (1D)
 */
TEST(nntrainer_LayerNode, finalize_06_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"input_shape=1"}));

  EXPECT_THROW(lnode->finalize(), std::invalid_argument);
}

/**
 * @brief finalize with invalid input_shape (2D)
 */
TEST(nntrainer_LayerNode, finalize_07_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"input_shape=1:1"}));

  EXPECT_THROW(lnode->finalize(), std::invalid_argument);
}

/**
 * @brief finalize with missing input_shape
 */
TEST(nntrainer_LayerNode, finalize_08_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({}));

  EXPECT_THROW(lnode->finalize(), std::invalid_argument);
}

/**
 * @brief finalize with invalid input_shape (negative value)
 */
TEST(nntrainer_LayerNode, finalize_09_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"input_shape=-1:1:1"}));

  EXPECT_THROW(lnode->finalize(), std::invalid_argument);
}

/**
 * @brief getRunContext for empty run_context
 */
TEST(nntrainer_LayerNode, getRunContext_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getRunContext(), std::runtime_error);
}

/**
 * @brief getInputDimensions for empty run_context
 */
TEST(nntrainer_LayerNode, getInputDimensions_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getInputDimensions(), std::runtime_error);
}

/**
 * @brief getOutputDimensions for empty run_context
 */
TEST(nntrainer_LayerNode, getOutputDimensions_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getOutputDimensions(), std::runtime_error);
}

/**
 * @brief getNumInputs for empty run_context
 */
TEST(nntrainer_LayerNode, getNumInputs_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getNumInputs(), std::runtime_error);
}

/**
 * @brief getInput for empty run_context
 */
TEST(nntrainer_LayerNode, getInput_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getInput(0), std::runtime_error);
}

/**
 * @brief getInputGrad for empty run_context
 */
TEST(nntrainer_LayerNode, getInputGrad_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getInputGrad(0), std::runtime_error);
}

/**
 * @brief getNumOutputs for empty run_context
 */
TEST(nntrainer_LayerNode, getNumOutputs_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getNumOutputs(), std::runtime_error);
}

/**
 * @brief getOutput for empty run_context
 */
TEST(nntrainer_LayerNode, getOutput_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getOutput(0), std::runtime_error);
}

/**
 * @brief getOutputGrad for empty run_context
 */
TEST(nntrainer_LayerNode, getOutputGrad_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getOutputGrad(0), std::runtime_error);
}

/**
 * @brief getNumWeights for empty run_context
 */
TEST(nntrainer_LayerNode, getNumWeights) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getNumWeights(), std::runtime_error);
}

/**
 * @brief getWeights for empty run_context
 */
TEST(nntrainer_LayerNode, getWeights_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getWeights(), std::runtime_error);
}

/**
 * @brief setWeights for empty run_context
 */
TEST(nntrainer_LayerNode, setWeights_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;
  const std::vector<float *> weights({new float});

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_THROW(lnode->setWeights(weights), std::runtime_error);
}

/**
 * @brief setWeights with non matched size of weights
 */
TEST(nntrainer_LayerNode, setWeights_02_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;
  nntrainer::Weight weight = nntrainer::Weight(
    nntrainer::TensorDim({1, 1, 1, 1}), nntrainer::Initializer::XAVIER_UNIFORM,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, 0.0f, true, false,
    "weight");
  float *float_ptr[2] = {nullptr, nullptr};
  const std::vector<float *> new_weights({float_ptr[0], float_ptr[1]});
  nntrainer::Var_Grad input =
    nntrainer::Var_Grad(nntrainer::TensorDim({1, 1, 1, 1}),
                        nntrainer::Initializer::NONE, true, false, "dummy");

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));
  EXPECT_NO_THROW(lnode->setProperty({"input_shape=1:1:1", "name=abc"}));
  EXPECT_NO_THROW(lnode->configureRunContext({&weight}, {&input}, {}, {}));

  EXPECT_THROW(lnode->setWeights(new_weights), std::runtime_error);
}

/**
 * @brief getWeight for empty run_context
 */
TEST(nntrainer_LayerNode, getWeight_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getWeight(0), std::runtime_error);
}

/**
 * @brief getWeightObject for empty run_context
 */
TEST(nntrainer_LayerNode, getWeightObject_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getWeightObject(0), std::runtime_error);
}

/**
 * @brief getWeightName for empty run_context
 */
TEST(nntrainer_LayerNode, getWeightName_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getWeightName(0), std::runtime_error);
}

/**
 * @brief getWeightWrapper for empty run_context
 */
TEST(nntrainer_LayerNode, getWeightWrapper_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->getWeightWrapper(0), std::runtime_error);
}

/**
 * @brief getWeightGrad for empty run_context
 */
TEST(nntrainer_LayerNode, getWeightGrad_01_n) {
  auto lnode = nntrainer::createLayerNode(nntrainer::IdentityLayer::type);
  EXPECT_THROW(lnode->getWeightGrad(0), std::runtime_error);
}

/**
 * @brief clearOptVar for empty run_context
 */
TEST(nntrainer_LayerNode, clearOptVar_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->clearOptVar(), std::runtime_error);
}

/**
 * @brief read for empty run_context
 */
TEST(nntrainer_LayerNode, read_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;
  std::ifstream ifstream("filename", std::ios_base::in | std::ios_base::binary);

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->read(ifstream), std::runtime_error);
}

/**
 * @brief save for empty run_context
 */
TEST(nntrainer_LayerNode, save_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;
  std::ofstream ofstream("filename", std::ios_base::in | std::ios_base::binary);

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->save(ofstream), std::runtime_error);
}

/**
 * @brief setBatch for empty run_context
 */
TEST(nntrainer_LayerNode, setBatch_01_n) {
  std::unique_ptr<nntrainer::LayerNode> lnode;

  EXPECT_NO_THROW(lnode =
                    nntrainer::createLayerNode(nntrainer::IdentityLayer::type));

  EXPECT_THROW(lnode->setBatch(0), std::invalid_argument);
}
