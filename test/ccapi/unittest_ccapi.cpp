// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file        unittest_ccapi.cc
 * @date        19 October 2020
 * @brief       cc API Unit tests.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */

#include <gtest/gtest.h>
#include <iostream>

#include <dataset.h>
#include <layer.h>
#include <ml-api-common.h>
#include <model.h>
#include <nntrainer_test_util.h>
#include <optimizer.h>

/**
 * @brief Neural Network Model Contruct Test
 */
TEST(ccapi_model, construct_01_n) {
  EXPECT_THROW(ml::train::createModel(ml::train::ModelType::UNKNOWN),
               std::invalid_argument);

  EXPECT_THROW(ml::train::createModel((ml::train::ModelType)100),
               std::invalid_argument);
}

/**
 * @brief Neural Network Model Contruct Test
 */
TEST(ccapi_model, construct_02_p) {
  EXPECT_NO_THROW(ml::train::createModel(ml::train::ModelType::NEURAL_NET));
}

/**
 * @brief Neural Network Layer Contruct Test
 */
TEST(ccapi_layer, construct_01_n) {
  EXPECT_THROW(ml::train::createLayer("unknown type"), std::invalid_argument);
}

/**
 * @brief Neural Network Layer Contruct Test
 */
TEST(ccapi_layer, construct_02_p) {
  auto layer = ml::train::createLayer("input");
  EXPECT_EQ(layer->getType(), "input");

  layer = ml::train::createLayer("fully_connected");
  EXPECT_EQ(layer->getType(), "fully_connected");

  layer = ml::train::createLayer("batch_normalization");
  EXPECT_EQ(layer->getType(), "batch_normalization");

  layer = ml::train::createLayer("conv2d");
  EXPECT_EQ(layer->getType(), "conv2d");

  layer = ml::train::createLayer("pooling2d");
  EXPECT_EQ(layer->getType(), "pooling2d");

  layer = ml::train::createLayer("flatten");
  EXPECT_EQ(layer->getType(), "flatten");

  layer = ml::train::createLayer("activation");
  EXPECT_EQ(layer->getType(), "activation");

  layer = ml::train::createLayer("addition");
  EXPECT_EQ(layer->getType(), "addition");

  layer = ml::train::createLayer("loss");
  EXPECT_EQ(layer->getType(), "loss");

  layer = ml::train::createLayer("Loss");
  EXPECT_EQ(layer->getType(), "loss");
}

/**
 * @brief Neural Network Optimizer Contruct Test
 */
TEST(ccapi_optimizer, construct_01_n) {
  EXPECT_THROW(ml::train::createOptimizer("Not existing type"),
               std::invalid_argument);
}

/**
 * @brief Neural Network Optimizer Contruct Test
 */
TEST(ccapi_optimizer, construct_02_p) {
  EXPECT_NO_THROW(ml::train::createOptimizer("adam"));
  EXPECT_NO_THROW(ml::train::createOptimizer("sgd"));
}

/**
 * @brief Neural Network Dataset Contruct Test
 */
TEST(ccapi_dataset, construct_01_n) {
  EXPECT_THROW(ml::train::createDataset(ml::train::DatasetType::UNKNOWN),
               std::invalid_argument);
}

/**
 * @brief Neural Network Dataset Contruct Test
 */
TEST(ccapi_dataset, construct_02_p) {
  EXPECT_NO_THROW(ml::train::createDataset(ml::train::DatasetType::GENERATOR));
  EXPECT_NO_THROW(ml::train::createDataset(ml::train::DatasetType::FILE));
}

/**
 * @brief Neural Network Model Training
 */
TEST(nntrainer_ccapi, train_with_config_01_p) {
  std::unique_ptr<ml::train::Model> model;

  std::string config_file = "./test_train_01_p.ini";
  RESET_CONFIG(config_file.c_str());

  replaceString("Input_Shape = 1:1:62720", "Input_Shape=1:1:62720", config_file,
                config_str);
  replaceString("batch_size = 32", "batch_size = 16", config_file, config_str);
  replaceString("BufferSize=100", "", config_file, config_str);

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_EQ(model->loadFromConfig(config_file), ML_ERROR_NONE);
  EXPECT_EQ(model->init(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  EXPECT_FLOAT_EQ(model->getTrainingLoss(), 4.434051);
  EXPECT_FLOAT_EQ(model->getValidationLoss(), 2.9646113);
  EXPECT_FLOAT_EQ(model->getLoss(), 2.8036859);
}

/**
 * @brief Neural Network Model Training
 */
TEST(nntrainer_ccapi, train_dataset_with_file_01_p) {
  std::unique_ptr<ml::train::Model> model;
  std::shared_ptr<ml::train::Layer> layer;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::shared_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_NO_THROW(layer = ml::train::createLayer(
                    "input", {"input_shape=1:1:62720", "normalization=true",
                              "bias_initializer=zeros"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::createLayer(
      "fully_connected",
      {"unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    optimizer = ml::train::createOptimizer(
      "adam", {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
               "beta1=0.002", "beta2=0.001", "epsilon=1e-7"}));
  EXPECT_NO_THROW(model->setOptimizer(optimizer));

  EXPECT_NO_THROW(
    dataset = ml::train::createDataset(
      ml::train::DatasetType::FILE, "trainingSet.dat", "valSet.dat", nullptr));
  EXPECT_EQ(dataset->setProperty({"label_data=label.dat", "buffer_size=100"}),
            ML_ERROR_NONE);
  EXPECT_EQ(model->setDataset(dataset), ML_ERROR_NONE);

  EXPECT_EQ(model->setProperty({"loss=cross", "batch_size=16", "epochs=2",
                                "save_path=model.bin"}),
            ML_ERROR_NONE);
  EXPECT_EQ(model->init(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  EXPECT_FLOAT_EQ(model->getTrainingLoss(), 2.1934659);
  EXPECT_FLOAT_EQ(model->getValidationLoss(), 2.2051108);
  EXPECT_FLOAT_EQ(model->getLoss(), 2.1751201);
}

/**
 * @brief Neural Network Model Training
 */
TEST(nntrainer_ccapi, train_dataset_with_generator_01_p) {
  std::unique_ptr<ml::train::Model> model;
  std::shared_ptr<ml::train::Layer> layer;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::shared_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_NO_THROW(layer = ml::train::createLayer(
                    "input", {"input_shape=1:1:62720", "normalization=true",
                              "bias_initializer=zeros"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::createLayer(
      "fully_connected",
      {"unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    optimizer = ml::train::createOptimizer(
      "adam", {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
               "beta1=0.002", "beta2=0.001", "epsilon=1e-7"}));
  EXPECT_NO_THROW(model->setOptimizer(optimizer));

  EXPECT_NO_THROW(
    dataset = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                       getBatch_train, getBatch_val, nullptr));
  EXPECT_EQ(dataset->setProperty({"buffer_size=100"}), ML_ERROR_NONE);
  EXPECT_EQ(model->setDataset(dataset), ML_ERROR_NONE);

  EXPECT_EQ(model->setProperty({"loss=cross", "batch_size=16", "epochs=2",
                                "save_path=model.bin"}),
            ML_ERROR_NONE);
  EXPECT_EQ(model->init(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  EXPECT_FLOAT_EQ(model->getTrainingLoss(), 2.2109976);
  EXPECT_FLOAT_EQ(model->getValidationLoss(), 1.995334);
  EXPECT_FLOAT_EQ(model->getLoss(), 1.9843285);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cout << "Failed to init gtest" << std::endl;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cout << "Failed to run the tests" << std::endl;
  }

  return result;
}
