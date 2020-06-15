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
 * @file	layer.h
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

#include <fstream>
#include <iostream>
#include <optimizer.h>
#include <tensor.h>
#include <tensor_dim.h>
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
 *            2. Batch Normalization Layer type
 *            3. Convolution 2D Layer type
 *            4. Pooling 2D Layer type
 *            5. Unknown
 */
typedef enum {
  LAYER_IN,
  LAYER_FC,
  LAYER_BN,
  LAYER_CONV2D,
  LAYER_POOLING2D,
  LAYER_UNKNOWN
} LayerType;

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
  Layer() :
    last_layer(false),
    init_zero(false),
    type(LAYER_UNKNOWN),
    activation(NULL),
    activation_prime(NULL),
    activation_type(ACT_UNKNOWN),
    bn_follow(false),
    weight_decay(),
    weight_ini_type(WEIGHT_UNKNOWN) {}

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
   * @param[in] output label Tensor
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
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize(bool last) = 0;

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
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setProperty(std::vector<std::string> values) = 0;

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

  /**
   * @brief     set Batch Normalization Layer followed
   * @param[in] ok true/false
   */
  void setBNfollow(bool ok) { this->bn_follow = ok; }

  /**
   * @brief     check hyper parameter for the layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int checkValidation();

  /**
   * @brief     set weight decay parameters
   * @param[in] w struct for weight decay
   */
  void setWeightDecay(WeightDecayParam w) { weight_decay = w; }

  /**
   * @brief  get Tensor Dimension
   * @retval TensorDim Tensor Dimension
   */
  TensorDim &getTensorDim() { return dim; }

  /**
   * @brief  set if this is last layer of Network
   * @param[in] last true/false
   */
  void setLast(bool last) { last_layer = last; }

  /**
   * @brief  set bias initialize with zero
   * @param[in] zero true/false
   */
  void setBiasZero(bool zero) { init_zero = zero; }

  /**
   * @brief  set Weight Initialization Type
   * @param[in] wini WeightIniType
   */
  void setWeightInit(WeightIniType wini) { weight_ini_type = wini; }

  /**
   * @brief  initialize Weight
   * @param[in] w_dim TensorDim
   * @param[in] init_type Weight Initialization Type
   * @param[out] status Status
   * @retval Tensor Initialized Tensor
   */
  Tensor initializeWeight(TensorDim w_dim, WeightIniType init_type,
                          int &status);

  void setInputDimension(TensorDim d) { input_dim = d; }

  TensorDim getOutputDimension() { return output_dim; }

  TensorDim getInputDimension() { return input_dim; }

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
   * @brief     Dimension of input activation
   */
  TensorDim input_dim;

  /**
   * @brief     Dimension of output activation
   */
  TensorDim output_dim;

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

  bool bn_follow;

  WeightDecayParam weight_decay;

  WeightIniType weight_ini_type;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYERS_H__ */
