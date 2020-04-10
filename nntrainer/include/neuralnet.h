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

#include <fstream>
#include <iostream>
#include <vector>
#include "layers.h"
#include "optimizer.h"
#include "tensor.h"

/**
 * @Namespace   Namespace of Network
 * @brief       Namespace for Network
 */
namespace Network {

/**
 * @brief     Enumeration of Network Type
 *            0. KNN ( k Nearest Neighbor )
 *            1. REG ( Logistic Regression )
 *            2. NEU ( Neural Network )
 *            3. Unknown
 */
typedef enum { NET_KNN, NET_REG, NET_NEU, NET_UNKNOWN } net_type;

/**
 * @brief     Enumeration for input configuration file parsing
 *            0. OPT     ( Optimizer Token )
 *            1. COST    ( Cost Function Token )
 *            2. NET     ( Network Token )
 *            3. ACTI    ( Activation Token )
 *            4. LAYER   ( Layer Token )
 *            5. WEIGHTINI  ( Weight Initialization Token )
 *            7. WEIGHT_DECAY  ( Weight Decay Token )
 *            8. UNKNOWN
 */
typedef enum {
  TOKEN_OPT,
  TOKEN_COST,
  TOKEN_NET,
  TOKEN_ACTI,
  TOKEN_LAYER,
  TOKEN_WEIGHTINI,
  TOKEN_WEIGHT_DECAY,
  TOKEN_UNKNOWN
} input_type;

/**
 * @class   NeuralNetwork Class
 * @brief   NeuralNetwork Class which has Network Configuration & Layers
 */
class NeuralNetwork {
 public:
  /**
   * @brief     Constructor of NeuralNetwork Class
   */
  NeuralNetwork(){};

  /**
   * @brief     Constructor of NeuralNetwork Class with Configuration file path
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
    * @brief     Get Optimizer
    * @retval    Optimizer
    */
  Optimizer getOptimizer() { return opt; };

  /**
    * @brief     Get Learing rate
    * @retval    Learning rate
    */
  float getLearningRate() { return learning_rate; };

  /**
   * @brief     Set Loss
   * @param[in] l loss value
   */
  void setLoss(float l);

  /**
   * @brief     Initialize Network
   */
  void init();

  /**
   * @brief     forward propagation
   * @param[in] input Input Tensor X
   * @retval    Output Tensor Y
   */
  Tensors::Tensor forwarding(Tensors::Tensor input);

  /**
   * @brief     forward propagation
   * @param[in] input Input Tensor X
   * @param[in] label Input Tensor Y2
   * @retval    Output Tensor Y
   */
  Tensors::Tensor forwarding(Tensors::Tensor input, Tensors::Tensor output);

  /**
   * @brief     back propagation to update W & B
   * @param[in] input Input Tensor X
   * @param[in] expectedOutput Lable Tensor Y
   * @param[in] iteration Epoch Number for ADAM
   */
  void backwarding(Tensors::Tensor input, Tensors::Tensor expectedOutput, int iteration);

  /**
   * @brief     save W & B into file
   */
  void saveModel();

  /**
   * @brief     read W & B from file
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

 private:
  /**
   * @brief     batch size
   */
  int batchsize;

  /**
   * @brief     function pointer for activation
   */
  float (*activation)(float);

  /**
   * @brief     function pointer for derivative of activation
   */
  float (*activationPrime)(float);

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
   * @brief     boolean to set the Bias zero
   */
  bool init_zero;

  /**
   * @brief     Cost Function type
   */
  Layers::cost_type cost;

  /**
   * @brief     Weight Initialization type
   */
  Layers::weightIni_type weightini;

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
   */
  Optimizer opt;

  /**
   * @brief     Network Type
   */
  net_type nettype;

  /**
   * @brief     vector for store layer pointers.
   */
  std::vector<Layers::Layer *> layers;
};
}

#endif
