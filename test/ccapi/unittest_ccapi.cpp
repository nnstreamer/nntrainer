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

static const std::string getTestResPath(const std::string &file) {
  return getResPath(file, {"test"});
}

/**
 * @brief Neural Network Model Construct Test
 */
TEST(ccapi_model, construct_01_n) {
  EXPECT_THROW(ml::train::createModel(ml::train::ModelType::UNKNOWN),
               std::invalid_argument);

  EXPECT_THROW(ml::train::createModel((ml::train::ModelType)100),
               std::invalid_argument);
}

/**
 * @brief Neural Network Model Construct Test
 */
TEST(ccapi_model, construct_02_p) {
  EXPECT_NO_THROW(ml::train::createModel(ml::train::ModelType::NEURAL_NET));
}

/**
 * @brief Neural Network Layer Construct Test
 */
TEST(ccapi_layer, construct_01_n) {
  EXPECT_THROW(ml::train::createLayer("unknown type"), std::invalid_argument);
}

/**
 * @brief Neural Network Layer Contruct Test
 */
TEST(ccapi_layer, construct_02_p) {
  std::shared_ptr<ml::train::Layer> layer;

  EXPECT_NO_THROW(layer = ml::train::layer::Input());
  EXPECT_EQ(layer->getType(), "input");

  EXPECT_NO_THROW(layer = ml::train::layer::FullyConnected());
  EXPECT_EQ(layer->getType(), "fully_connected");

  EXPECT_NO_THROW(layer = ml::train::layer::BatchNormalization());
  EXPECT_EQ(layer->getType(), "batch_normalization");

  EXPECT_NO_THROW(layer = ml::train::layer::Convolution2D());
  EXPECT_EQ(layer->getType(), "conv2d");

  EXPECT_NO_THROW(layer = ml::train::layer::Pooling2D());
  EXPECT_EQ(layer->getType(), "pooling2d");

  EXPECT_NO_THROW(layer = ml::train::layer::Flatten());
  EXPECT_EQ(layer->getType(), "flatten");

  EXPECT_NO_THROW(layer = ml::train::layer::Addition());
  EXPECT_EQ(layer->getType(), "addition");

  EXPECT_NO_THROW(layer = ml::train::layer::Concat());
  EXPECT_EQ(layer->getType(), "concat");

  EXPECT_NO_THROW(layer = ml::train::layer::MultiOut());
  EXPECT_EQ(layer->getType(), "output");

#ifdef ENABLE_NNSTREAMER_BACKBONE
  EXPECT_NO_THROW(layer = ml::train::layer::BackboneNNStreamer());
  EXPECT_EQ(layer->getType(), "backbone_nnstreamer");
#endif

#ifdef ENABLE_TFLITE_BACKBONE
  EXPECT_NO_THROW(layer = ml::train::layer::BackboneTFLite());
  EXPECT_EQ(layer->getType(), "backbone_tflite");
#endif

  EXPECT_NO_THROW(layer = ml::train::layer::ReLU());
  EXPECT_EQ(layer->getType(), "activation");

  EXPECT_NO_THROW(layer = ml::train::layer::Tanh());
  EXPECT_EQ(layer->getType(), "activation");

  EXPECT_NO_THROW(layer = ml::train::layer::Sigmoid());
  EXPECT_EQ(layer->getType(), "activation");

  EXPECT_NO_THROW(layer = ml::train::layer::Softmax());
  EXPECT_EQ(layer->getType(), "activation");
}

/**
 * @brief Neural Network Loss Layer Contruct Test
 */
TEST(ccapi_layer, construct_03_p) {
  std::shared_ptr<ml::train::Layer> layer;

  EXPECT_NO_THROW(layer = ml::train::loss::MSE());
  EXPECT_EQ(layer->getType(), "loss");

  EXPECT_NO_THROW(layer = ml::train::loss::CrossEntropy());
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
  EXPECT_NO_THROW(ml::train::optimizer::Adam());
  EXPECT_NO_THROW(ml::train::optimizer::SGD());
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

  static IniSection model_base("Model", "Type = NeuralNetwork"
                                        " | Learning_rate = 0.0001"
                                        " | Decay_rate = 0.96"
                                        " | Decay_steps = 1000"
                                        " | Epochs = 1"
                                        " | Optimizer = adam"
                                        " | Loss = cross"
                                        " | Weight_Regularizer = l2norm"
                                        " | weight_regularizer_constant = 0.005"
                                        " | Save_Path = 'model.bin'"
                                        " | batch_size = 32"
                                        " | beta1 = 0.9"
                                        " | beta2 = 0.9999"
                                        " | epsilon = 1e-7");

  static IniSection dataset("Dataset", "BufferSize=100"
                                       " | TrainData = trainingSet.dat"
                                       " | ValidData = valSet.dat"
                                       " | LabelData = label.dat");

  static IniSection inputlayer("inputlayer", "Type = input"
                                             "| Input_Shape = 1:1:62720"
                                             "| bias_initializer = zeros"
                                             "| Normalization = true"
                                             "| Activation = sigmoid");

  static IniSection outputlayer("outputlayer", "Type = fully_connected"
                                               "| input_layers = inputlayer"
                                               "| Unit = 10"
                                               "| bias_initializer = zeros"
                                               "| Activation = softmax");

  ScopedIni s("test_train_01_p",
              {model_base + "batch_size = 16", dataset + "-BufferSize",
               inputlayer, outputlayer});

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_EQ(model->loadFromConfig(s.getIniName()), ML_ERROR_NONE);
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
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

  EXPECT_NO_THROW(layer = ml::train::layer::Input({"input_shape=1:1:62720",
                                                   "normalization=true",
                                                   "bias_initializer=zeros"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::layer::FullyConnected(
      {"unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform",
       "input_layers=" + layer->getName()}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    optimizer = ml::train::optimizer::Adam(
      {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
       "beta1=0.002", "beta2=0.001", "epsilon=1e-7"}));
  EXPECT_NO_THROW(model->setOptimizer(optimizer));

  EXPECT_NO_THROW(dataset = ml::train::createDataset(
                    ml::train::DatasetType::FILE,
                    getTestResPath("trainingSet.dat").c_str(),
                    getTestResPath("valSet.dat").c_str(), nullptr));
  EXPECT_EQ(dataset->setProperty(
              {"label_data=" + getTestResPath("label.dat"), "buffer_size=100"}),
            ML_ERROR_NONE);
  EXPECT_EQ(model->setDataset(dataset), ML_ERROR_NONE);

  EXPECT_EQ(model->setProperty({"loss=cross", "batch_size=16", "epochs=2",
                                "save_path=model.bin"}),
            ML_ERROR_NONE);
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
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

  EXPECT_NO_THROW(layer = ml::train::layer::Input({"input_shape=1:1:62720",
                                                   "normalization=true",
                                                   "bias_initializer=zeros"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::layer::FullyConnected(
      {"unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform",
       "input_layers=" + layer->getName()}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    optimizer = ml::train::optimizer::Adam(
      {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
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
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  EXPECT_FLOAT_EQ(model->getTrainingLoss(), 2.2109976);
  EXPECT_FLOAT_EQ(model->getValidationLoss(), 1.995334);
  EXPECT_FLOAT_EQ(model->getLoss(), 1.9843285);
}

/**
 * @brief Neural Network Model Training
 */
TEST(nntrainer_ccapi, train_batch_size_update_after) {
  std::unique_ptr<ml::train::Model> model;
  std::shared_ptr<ml::train::Layer> layer;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::shared_ptr<ml::train::Dataset> dataset;

  EXPECT_NO_THROW(model =
                    ml::train::createModel(ml::train::ModelType::NEURAL_NET));

  EXPECT_NO_THROW(layer = ml::train::layer::Input({"input_shape=1:1:62720",
                                                   "normalization=true",
                                                   "bias_initializer=zeros"}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    layer = ml::train::layer::FullyConnected(
      {"unit= 10", "activation=softmax", "bias_initializer=zeros",
       "weight_regularizer=l2norm", "weight_regularizer_constant=0.005",
       "weight_initializer=xavier_uniform",
       "input_layers=" + layer->getName()}));
  EXPECT_NO_THROW(model->addLayer(layer));

  EXPECT_NO_THROW(
    optimizer = ml::train::optimizer::Adam(
      {"learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
       "beta1=0.002", "beta2=0.001", "epsilon=1e-7"}));
  EXPECT_NO_THROW(model->setOptimizer(optimizer));

  EXPECT_NO_THROW(dataset = ml::train::createDataset(
                    ml::train::DatasetType::FILE,
                    getTestResPath("trainingSet.dat").c_str(),
                    getTestResPath("valSet.dat").c_str(), nullptr));
  EXPECT_EQ(dataset->setProperty(
              {"label_data=" + getTestResPath("label.dat"), "buffer_size=100"}),
            ML_ERROR_NONE);
  EXPECT_EQ(model->setDataset(dataset), ML_ERROR_NONE);

  EXPECT_EQ(model->setProperty({"loss=cross", "batch_size=16", "epochs=1"}),
            ML_ERROR_NONE);

  /** Update batch size after compile */
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->setProperty({"batch_size=128"}), ML_ERROR_NONE);

  /** Update batch size after initialize */
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
  EXPECT_EQ(model->setProperty({"batch_size=8"}), ML_ERROR_NONE);

  EXPECT_NO_THROW(model->train());

  /** Update batch size after train */
  EXPECT_EQ(model->setProperty({"batch_size=16"}), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  /** Update batch size after train */
  EXPECT_EQ(model->setProperty({"batch_size=32"}), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  /** Update batch size after train */
  EXPECT_EQ(model->setProperty({"batch_size=4"}), ML_ERROR_NONE);
  EXPECT_NO_THROW(model->train());

  EXPECT_FLOAT_EQ(model->getTrainingLoss(), 1.9582682);
  EXPECT_FLOAT_EQ(model->getValidationLoss(), 2.1831701);
  EXPECT_FLOAT_EQ(model->getLoss(), 2.1985414);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  nntrainer::AppContext::Global().setWorkingDirectory(getTestResPath(""));
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
