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
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __LAYERS_H__
#define __LAYERS_H__
#ifdef __cplusplus

#include "optimizer.h"
#include "tensor.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace nntrainer {

/**
 * @brief     Enumeration of cost(loss) function type
 *            0. MSR ( Mean Squared Roots )
 *            1. ENTROPY ( Cross Entropy )
 *            2. Unknown
 */
typedef enum { COST_MSR, COST_ENTROPY, COST_UNKNOWN } CostType;

/**
 * @brief     Enumeration of activation function type
 *            0. tanh
 *            1. sigmoid
 *            2. relu
 *            3. Unknown
 */
typedef enum {
  ACT_TANH,
  ACT_SIGMOID,
  ACT_RELU,
  ACT_SOFTMAX,
  ACT_UNKNOWN
} ActiType;

/**
 * @brief     Enumeration of layer type
 *            0. Input Layer type
 *            1. Fully Connected Layer type
 *            2. Unknown
 */
typedef enum { LAYER_IN, LAYER_FC, LAYER_BN, LAYER_UNKNOWN } LayerType;

/**
 * @brief     Enumeration of Weight Initialization Type
 *            0. WEIGHT_LECUN_NORMAL ( LeCun normal initialization )
 *            1. WEIGHT_LECUN_UNIFORM (LeCun uniform initialization )
 *            2. WEIGHT_XAVIER_NORMAL ( Xavier normal initialization )
 *            3. WEIGHT_XAVIER_UNIFORM ( Xavier uniform initialization )
 *            4. WEIGHT_HE_NORMAL ( He normal initialization )
 *            5. WEIGHT_HE_UNIFORM ( He uniform initialization )
 */
typedef enum {
  WEIGHT_LECUN_NORMAL,
  WEIGHT_LECUN_UNIFORM,
  WEIGHT_XAVIER_NORMAL,
  WEIGHT_XAVIER_UNIFORM,
  WEIGHT_HE_NORMAL,
  WEIGHT_HE_UNIFORM,
  WEIGHT_UNKNOWN
} WeightIniType;

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 */
class Layer {
public:
  Layer();
  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~Layer(){};

  /**
   * @brief     Forward Propation of neural Network
   * @param[in] in Input Tensor taken by upper layer
   * @retval    Output Tensor
   */
  virtual Tensor forwarding(Tensor in, int &status) = 0;

  /**
   * @brief     Forward Propation of neural Network
   * @param[in] in Input Tensor taken by upper layer
   * @retval    Output Tensor
   */
  virtual Tensor forwarding(Tensor in, Tensor output, int &status) = 0;

  /**
   * @brief     Back Propation of neural Network
   * @param[in] in Input Tensor taken by lower layer
   * @param[in] iteration Epoch value for the ADAM Optimizer
   * @retval    Output Tensor
   */
  virtual Tensor backwarding(Tensor in, int iteration) = 0;

  /**
   * @brief     Initialize the layer
   *            - Weight(Height, Width), Bias(1, Width)
   * @param[in] b batch
   * @param[in] h Height
   * @param[in] w Width
   * @param[in] last last layer
   * @param[in] init_zero Bias initialization with zero
   * @param[in] wini Weight Initialization Scheme
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize(int b, int h, int w, bool last, bool init_zero,
                         WeightIniType wini) = 0;

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
   * @brief     set Property of layer
   * @param[in] key key of property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setProperty(const char *key, std::vector<std::string> values) = 0;

  /**
   * @brief     Optimizer Setter
   * @param[in] opt Optimizer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setOptimizer(Optimizer &opt);

  /**
   * @brief     Activation Setter
   * @param[in] activation activation type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setActivation(ActiType activation);

  /**
   * @brief     Layer type Setter
   * @param[in] type layer type
   */
  void setType(LayerType type) { this->type = type; }

  /**
   * @brief     Layer type Getter
   * @retval type LayerType
   */
  LayerType getType() { return type; }

  /**
   * @brief     Copy Layer
   * @param[in] l Layer to be copied
   */
  virtual void copy(std::shared_ptr<Layer> l) = 0;

  void setBNfallow(bool ok) { this->bn_fallow = ok; }

  int checkValidation();

  void setWeightDecay(WeightDecayParam w) { weight_decay = w; }

protected:
  /**
   * @brief     Input Tensor
   */
  Tensor input;

  /**
   * @brief     Hidden Layer Tensor which store the
   *            forwading result
   */
  Tensor hidden;

  /**
   * @brief     last layer
   */
  bool last_layer;

  /**
   * @brief     Dimension of this layer
   */
  TensorDim dim;

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
  LayerType type;

  /**
   * @brief     Activation function pointer
   */
  float (*activation)(float);

  /**
   * @brief     Activation Derivative function pointer
   */
  float (*activation_prime)(float);

  ActiType activation_type;

  bool bn_fallow;

  WeightDecayParam weight_decay;
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
  InputLayer() : normalization(false), standardization(false){};

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
  Tensor backwarding(Tensor in, int iteration) { return input; };

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] in input Tensor from lower layer.
   * @retval    return Input Tensor
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] in input Tensor from lower layer.
   * @param[in] output label Tensor.
   * @retval    return Input Tensor
   */
  Tensor forwarding(Tensor in, Tensor output, int &status) {
    return forwarding(in, status);
  };

  /**
   * @brief     Set Optimizer
   * @param[in] opt optimizer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setOptimizer(Optimizer &opt);

  /**
   * @brief     Initializer of Input Layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] last last layer
   * @param[in] init_zero boolean to set Bias zero
   * @param[in] wini Weight Initialization Scheme
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(int b, int h, int w, bool last, bool init_zero,
                 WeightIniType wini);

  /**
   * @brief     Copy Layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     set normalization
   * @param[in] enable boolean
   */
  void setNormalization(bool enable) { this->normalization = enable; };

  /**
   * @brief     set standardization
   * @param[in] enable boolean
   */
  void setStandardization(bool enable) { this->standardization = enable; };

  /**
   * @brief     set Property of layer
   * @param[in] key key of property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(const char *key, std::vector<std::string> values);

  /**
   * @brief     Property Enumeration
   *            0. input shape : string
   *            1. bias zero : bool
   *            2. normalization : bool
   *            3. normalization : bool
   */
  enum class PropertyType {
    input_shape = 0,
    bias_zero = 1,
    normalization = 2,
    standardization = 3
  };

private:
  bool normalization;
  bool standardization;
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
  FullyConnectedLayer() : loss(0.0), cost(COST_UNKNOWN){};

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
   * @param[in] in Input Tensor from upper layer
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] input input Tensor from lower layer.
   * @param[in] output label Tensor.
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor in, Tensor output, int &status);

  /**
   * @brief     back propagation
   *            Calculate dJdB & dJdW & Update W & B
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    dJdB x W Tensor
   */
  Tensor backwarding(Tensor in, int iteration);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     initialize layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] last last layer
   * @param[in] init_zero boolean to set Bias zero
   * @param[in] wini Weight Initialization Scheme
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(int b, int h, int w, bool last, bool init_zero,
                 WeightIniType wini);
  /**
   * @brief     get Loss value
   */
  float getLoss() { return loss; }

  /**
   * @brief     set cost function
   * @param[in] c cost function type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setCost(CostType c);

  /**
   * @brief     set Property of layer
   * @param[in] key key of property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(const char *key, std::vector<std::string> values);

  /**
   * @brief     Property Enumeration
   *            0. input shape : string
   *            1. bias zero : bool
   *            4. activation : bool
   *            6. weight_decay : string (type), float
   */
  enum class PropertyType {
    input_shape = 0,
    bias_zero = 1,
    activation = 4,
    weight_decay = 6
  };

private:
  Tensor weight;
  Tensor bias;
  float loss;
  CostType cost;
};

/**
 * @class   BatchNormalizationLayer
 * @brief   Batch Noramlization Layer
 */
class BatchNormalizationLayer : public Layer {
public:
  /**
   * @brief     Constructor of Batch Noramlization Layer
   */
  BatchNormalizationLayer() : epsilon(0.0){};

  /**
   * @brief     Destructor of BatchNormalizationLayer
   */
  ~BatchNormalizationLayer(){};

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
   * @param[in] in Input Tensor from upper layer
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] input input Tensor from lower layer.
   * @param[in] output label Tensor.
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor in, Tensor output, int &status) {
    return forwarding(in, status);
  };

  /**
   * @brief     back propagation
   *            Calculate dJdB & dJdW & Update W & B
   * @param[in] in Input Tensor from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    dJdB x W Tensor
   */
  Tensor backwarding(Tensor in, int iteration);

  /**
   * @brief     set optimizer
   * @param[in] opt Optimizer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setOptimizer(Optimizer &opt);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     initialize layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] last last layer
   * @param[in] init_zero boolean to set Bias zero
   * @param[in] wini Weight Initialization Scheme
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(int b, int h, int w, bool last, bool init_zero,
                 WeightIniType wini);

  /**
   * @brief     set Property of layer
   * @param[in] key key of property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(const char *key, std::vector<std::string> values);

  /**
   * @brief     Property Enumeration
   *            0. input shape : string
   *            1. bias zero : bool
   *            5. epsilon : float
   */
  enum class PropertyType {
    input_shape = 0,
    bias_zero = 1,
    epsilon = 5,
  };

private:
  Tensor weight;
  Tensor bias;
  Tensor mu;
  Tensor var;
  Tensor gamma;
  Tensor beta;
  float epsilon;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYERS_H__ */
