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
 * @file	layers.h
 * @date	04 December 2019
 * @brief	This is Layer classes of Neural Network
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <fstream>
#include <iostream>
#include <vector>
#include "tensor.h"

/**
 * @Namespace   Namespace of Layers
 * @brief       Namespace for Layers
 */
namespace Layers {

/**
 * @brief     Enumeration of optimizer type
 *            0. SGD ( Stocastic Gradient Descent )
 *            1. ADAM ( Adaptive Moment Estimation )
 *            2. Unknown
 */
typedef enum { OPT_SGD, OPT_ADAM, OPT_UNKNOWN } opt_type;

/**
 * @brief     Enumeration of cost(loss) function type
 *            0. CATEGORICAL ( Categorical Cross Entropy )
 *            1. MSR ( Mean Squared Roots )
 *            2. ENTROPY ( Cross Entropy )
 *            3. Unknown
 */
typedef enum { COST_CATEGORICAL, COST_MSR, COST_ENTROPY, COST_UNKNOWN } cost_type;

/**
 * @brief     Enumeration of activation function type
 *            0. tanh
 *            1. sigmoid
 *            2. Unknown
 */
typedef enum { ACT_TANH, ACT_SIGMOID, ACT_UNKNOWN } acti_type;

/**
 * @brief     Enumeration of layer type
 *            0. Input Layer type
 *            1. Fully Connected Layer type
 *            2. Output Layer type
 *            3. Unknown
 */
typedef enum { LAYER_IN, LAYER_FC, LAYER_OUT, LAYER_UNKNOWN } layer_type;

/**
 * @brief     type for the Optimizor to save hyper-parameter
 */
typedef struct {
  opt_type type;
  float learning_rate;
  double beta1;
  double beta2;
  double epsilon;
  acti_type activation;
  float decay_rate;
  float decay_steps;
} Optimizer;

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 */
class Layer {
 public:
  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~Layer(){};

  /**
   * @brief     Forward Propation of neural Network
   * @param[in] input Input Tensor taken by upper layer
   * @retval    Output Tensor
   */
  virtual Tensor forwarding(Tensor input) = 0;

  /**
   * @brief     Forward Propation of neural Network
   * @param[in] input Input Tensor taken by upper layer
   * @retval    Output Tensor
   */
  virtual Tensor forwarding(Tensor input, Tensor output) = 0;

  /**
   * @brief     Back Propation of neural Network
   * @param[in] input Input Tensor taken by lower layer
   * @param[in] iteration Epoch value for the ADAM Optimizer
   * @retval    Output Tensor
   */
  virtual Tensor backwarding(Tensor input, int iteration) = 0;

  /**
   * @brief     Initialize the layer
   *            - Weight(Height, Width), Bias(1, Width)
   * @param[in] b batch
   * @param[in] h Height
   * @param[in] w Width
   * @param[in] id index of this layer
   * @param[in] init_zero Bias initialization with zero
   */
  virtual void initialize(int b, int h, int w, int id, bool init_zero) = 0;

  /**
   * @brief     read layer Weight & Bias data from file
   * @param[in] file input file stream
   */
  virtual void read(std::ifstream &file) = 0;

  /**
   * @brief     save layer Weight & Bias data from file
   * @param[in] file output file stream
   */
  virtual void save(std::ofstream &file) = 0;

  /**
   * @brief     Optimizer Setter
   * @param[in] opt Optimizer
   */
  virtual void setOptimizer(Optimizer opt) = 0;

  /**
   * @brief     Layer type Setter
   * @param[in] type layer type
   */
  void setType(layer_type type) { this->type = type; }

  /**
   * @brief     Copy Layer
   * @param[in] l Layer to be copied
   */
  virtual void copy(Layer *l) = 0;

  /**
   * @brief     Input Tensor
   */
  Tensor Input;

  /**
   * @brief     Hidden Layer Tensor which store the
   *            forwading result
   */
  Tensor hidden;

  /**
   * @brief     Layer index
   */
  unsigned int index;

  /**
   * @brief     batch size of Weight Data
   */
  unsigned int batch;

  /**
   * @brief     width size of Weight Data
   */
  unsigned int width;

  /**
   * @brief     height size of Weight Data
   */
  unsigned int height;

  /**
   * @brief     Optimizer for this layer
   */
  Optimizer opt;

  /**
   * @brief     Boolean for the Bias to set zero
   */
  bool init_zero;

  /**
   * @brief     Layer type
   */
  layer_type type;

  /**
   * @brief     Activation function pointer
   */
  float (*activation)(float);

  /**
   * @brief     Activation Derivative function pointer
   */
  float (*activationPrime)(float);
};

/**
 * @class   Input Layer
 * @brief   Just Handle the Input of Network
 */
class InputLayer : public Layer {
 public:
  /**
   * @brief     Constructor of InputLayer
   */
  InputLayer(){};

  /**
   * @brief     Destructor of InputLayer
   */
  ~InputLayer(){};

  /**
   * @brief     No Weight data for this Input Layer
   */
  void read(std::ifstream &file){};

  /**
   * @brief     No Weight data for this Input Layer
   */
  void save(std::ofstream &file){};

  /**
   * @brief     It is back propagation of input layer.
   *            It return Input as it is.
   * @param[in] input input Tensor from lower layer.
   * @param[in] iteration Epoch Number for ADAM
   * @retval
   */
  Tensor backwarding(Tensor input, int iteration) { return Input; };

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] input input Tensor from lower layer.
   * @retval    return Input Tensor
   */
  Tensor forwarding(Tensor input);

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] input input Tensor from lower layer.
   * @param[in] output label Tensor.
   * @retval    return Input Tensor
   */
  Tensor forwarding(Tensor input, Tensor output){
    return forwarding(input);
  };

  /**
   * @brief     Set Optimizer
   * @param[in] opt optimizer
   */
  void setOptimizer(Optimizer opt);

  /**
   * @brief     Initializer of Input Layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] id index of this layer
   * @param[in] init_zero boolean to set Bias zero
   */
  void initialize(int b, int h, int w, int id, bool init_zero);

  /**
   * @brief     Copy Layer
   * @param[in] l layer to copy
   */
  void copy(Layer *l);
};

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayer : public Layer {
 public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayer(){};

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedLayer(){};

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file);

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file);

  /**
   * @brief     forward propagation with input
   * @param[in] input Input Tensor from upper layer
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor input);

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] input input Tensor from lower layer.
   * @param[in] output label Tensor.
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor input, Tensor output){
    return forwarding (input);
  };

  /**
   * @brief     back propagation
   *            Calculate dJdB & dJdW & Update W & B
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    dJdB x W Tensor
   */
  Tensor backwarding(Tensor input, int iteration);

  /**
   * @brief     set optimizer
   * @param[in] opt Optimizer
   */
  void setOptimizer(Optimizer opt);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(Layer *l);

  /**
   * @brief     initialize layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] id layer index
   * @param[in] init_zero boolean to set Bias zero
   */
  void initialize(int b, int h, int w, int id, bool init_zero);

 private:
  Tensor Weight;
  Tensor Bias;

  /**
   * @brief     First Momentum Tensor for the ADAM
   */
  Tensor M;

  /**
   * @brief     Second Momentum Tensor for the ADAM
   */
  Tensor V;
};

/**
 * @class   OutputLayer
 * @brief   OutputLayer (has Cost Function & Weight, Bias)
 */
class OutputLayer : public Layer {
 public:
  /**
   * @brief     Constructor of OutputLayer
   */
  OutputLayer(){};

  /**
   * @brief     Destructor of OutputLayer
   */
  ~OutputLayer(){};

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file);

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &flle);

  /**
   * @brief     forward propagation with input
   * @param[in] input Input Tensor from upper layer
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor input);

  /**
   * @brief     forward propagation with input and set loss
   * @param[in] input Input Tensor from upper layer
   * @param[in] output Label Tensor
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor input, Tensor output);

  /**
   * @brief     back propagation
   *            Calculate dJdB & dJdW & Update W & B
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    dJdB x W Tensor
   */
  Tensor backwarding(Tensor label, int iteration);

  /**
   * @brief     set optimizer
   * @param[in] opt Optimizer
   */
  void setOptimizer(Optimizer opt);

  /**
   * @brief     initialize layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] id layer index
   * @param[in] init_zero boolean to set Bias zero
   */
  void initialize(int b, int w, int h, int id, bool init_zero);

  /**
   * @brief     get Loss value
   */
  float getLoss() { return loss; }

  /**
   * @brief     set cost function
   * @param[in] c cost function type
   */
  void setCost(cost_type c) { this->cost = c; };

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(Layer *l);

 private:
  Tensor Weight;
  Tensor Bias;
  Tensor M;
  Tensor V;
  float loss;
  cost_type cost;
};
}

#endif
