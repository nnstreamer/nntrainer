/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 *
 * @file	neuralnet.h
 * @date	04 December 2019
 * @brief	This is Neural Network Class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __NEURALNET_H__
#define __NEURALNET_H__
#ifdef __cplusplus

#include <bn_layer.h>
#include <conv2d_layer.h>
#include <databuffer.h>
#include <fc_layer.h>
#include <fstream>
#include <input_layer.h>
#include <iostream>
#include <layer.h>
#include <loss_layer.h>
#include <optimizer.h>
#include <tensor.h>
#include <vector>

namespace nntrainer {

/**
 * @brief     Enumeration of Network Type
 *            0. KNN ( k Nearest Neighbor )
 *            1. REG ( Logistic Regression )
 *            2. NEU ( Neural Network )
 *            3. Unknown
 */
typedef enum { NET_KNN, NET_REG, NET_NEU, NET_UNKNOWN } NetType;

/**
 * @class   NeuralNetwork Class
 * @brief   NeuralNetwork Class which has Network Configuration & Layers
 */
class NeuralNetwork {
public:
  /**
   * @brief     Constructor of NeuralNetwork Class
   */
  NeuralNetwork();

  /**
   * @brief     Constructor of NeuralNetwork Class with Configuration file
   * path
   */
  NeuralNetwork(std::string config_path);

  /**
   * @brief     Destructor of NeuralNetwork Class
   */
  ~NeuralNetwork(){};

  /**
   * @brief     Get Loss
   * @retval    loss value
   */
  float getLoss();

  /**
   * @brief     Set Optimizer
   * @retval    Optimizer
   */
  void setOptimizer(Optimizer optimizer) { opt = optimizer; };

  /**
   * @brief     Get Learning rate
   * @retval    Learning rate
   */
  float getLearningRate() { return learning_rate; };

  /**
   * @brief     Set Loss
   * @param[in] l loss value
   */
  void setLoss(float l);

  /**
   * @brief     Initialize Network. This should be called after set all hyper
   * parmeters.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int init();

  /**
   * @brief     set Property of Network
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     Initialize Network
   * @param[in] opimizer optimizer instance
   * @param[in] arg_list argument list
   *            "loss = cross | msr"
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int init(std::shared_ptr<Optimizer> optimizer,
           std::vector<std::string> arg_list);

  /**
   * @brief     forward propagation
   * @param[in] input Input Tensor X
   * @retval    Output Tensor Y
   */
  Tensor forwarding(Tensor input, int &status);

  /**
   * @brief     forward propagation
   * @param[in] input Input Tensor X
   * @param[in] label Input Tensor Y2
   * @retval    Output Tensor Y
   */
  Tensor forwarding(Tensor input, Tensor output, int &status);

  /**
   * @brief     back propagation to update W & B
   * @param[in] input Input Tensor X
   * @param[in] expectedOutput Lable Tensor Y
   * @param[in] iteration Epoch Number for ADAM
   */
  int backwarding(Tensor input, Tensor expected_output, int iteration);

  /**
   * @brief     save model and training parameters into file
   */
  void saveModel();

  /**
   * @brief     read model and training parameters from file
   */
  void readModel();

  /**
   * @brief     set configuration file
   * @param[in] config_path configuration file path
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setConfig(std::string config_path);

  /**
   * @brief     get Epoch
   * @retval    epoch
   */
  unsigned int getEpoch() { return epoch; };

  /**
   * @brief     Copy Neural Network
   * @param[in] from NeuralNetwork Object to copy
   * @retval    NeuralNewtork Object copyed
   */
  NeuralNetwork &copy(NeuralNetwork &from);

  /**
   * @brief     finalize NeuralNetwork Object
   */
  void finalize();

  /**
   * @brief     Run NeuralNetwork train
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train_run();

  /**
   * @brief     Run NeuralNetwork train
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train();

  /**
   * @brief     Run NeuralNetwork train
   * @param[in] values hyper parmeters
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train(std::vector<std::string> values);

  /**
   * @brief     Run NeuralNetwork train with callback function by user
   * @param[in] train_func callback function to get train data. This provides
   * mini batch size data per every call.
   * @param[in] val_func callback function to get validation data. This provides
   * mini batch size data per every call.
   * @param[in] test_func callback function to get test data. This provides
   * mini batch size data per every call.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train(std::function<bool(float *, float *, int *)> function_train,
            std::function<bool(float *, float *, int *)> function_val,
            std::function<bool(float *, float *, int *)> function_test);

  /**
   * @brief     Run NeuralNetwork train with callback function by user
   * @param[in] train_func callback function to get train data. This provides
   * mini batch size data per every call.
   * @param[in] val_func callback function to get validation data. This provides
   * mini batch size data per every call.
   * @param[in] test_func callback function to get test data. This provides
   * mini batch size data per every call.
   * @param[in] values hyper-parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train(std::function<bool(float *, float *, int *)> function_train,
            std::function<bool(float *, float *, int *)> function_val,
            std::function<bool(float *, float *, int *)> function_test,
            std::vector<std::string> values);

  /**
   * @brief     check neural network whether the hyper-parameters are set.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int checkValidation();

  /**
   * @brief     add layer into neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int addLayer(std::shared_ptr<Layer> layer);

  enum class PropertyType {
    loss = 0,
    cost = 1,
    train_data = 2,
    val_data = 3,
    test_data = 4,
    label_data = 5,
    buffer_size = 6,
    batch_size = 7,
    epochs = 8,
    model_file = 9,
    continue_train = 10,
    unknown = 11,
  };

private:
  /**
   * @brief     batch size
   */
  int batch_size;

  /**
   * @brief     learning rate
   */
  float learning_rate;

  /**
   * @brief     decay_rate for decayed learning rate
   */
  float decay_rate;

  /**
   * @brief     decay_step for decayed learning rate
   */
  float decay_steps;

  /**
   * @brief     Maximum Epoch
   */
  unsigned int epoch;

  /**
   * @brief     loss
   */
  float loss;

  /**
   * @brief     Cost Function type
   */
  CostType cost;

  /**
   * @brief     Weight Initialization type
   */
  WeightIniType weight_ini;

  /**
   * @brief     Model path to save or read
   */
  std::string model;

  /**
   * @brief     Configuration file path
   */
  std::string config;

  /**
   * @brief     Optimizer
   * @note      This gets copied into each layer, do not use this directly
   */
  Optimizer opt;

  /**
   * @brief     Network Type
   */
  NetType net_type;

  /**
   * @brief     vector for store layer pointers.
   */
  std::vector<std::shared_ptr<Layer>> layers;

  /**
   * @brief     Data Buffer to get Input
   */
  std::shared_ptr<DataBuffer> data_buffer;

  /**
   * @brief    Continue train from the previous state of optimizer and
   * iterations
   */
  bool continue_train;

  /**
   * @brief     Number of iterations trained
   */
  uint64_t iter;

  /**
   * @brief     Network is initialized
   */
  bool initialized;

  /**
   * @brief     Sets up and initialize the loss layer
   */
  int initLossLayer();
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __NEURALNET_H__ */
