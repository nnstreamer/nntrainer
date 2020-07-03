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
 *            2. ENTROPY_SIGMOID (Cross Entropy amalgamated with sigmoid for
 * stability)
 *            3. ENTROPY_SOFTMAX (Cross Entropy amalgamated with softmax for
 * stability)
 *            4. Unknown
 */
typedef enum {
  COST_MSR,
  COST_ENTROPY,
  COST_ENTROPY_SIGMOID,
  COST_ENTROPY_SOFTMAX,
  COST_UNKNOWN
} CostType;

/**
 * @brief     Enumeration of activation function type
 *            0. tanh
 *            1. sigmoid
 *            2. relu
 *            3. softmax
 *            4. none
 *            5. Unknown
 */
typedef enum {
  ACT_TANH,
  ACT_SIGMOID,
  ACT_RELU,
  ACT_SOFTMAX,
  ACT_NONE,
  ACT_UNKNOWN
} ActiType;

/**
 * @brief     Enumeration of layer type
 *            0. Input Layer type
 *            1. Fully Connected Layer type
 *            2. Batch Normalization Layer type
 *            3. Convolution 2D Layer type
 *            4. Pooling 2D Layer type
 *            5. Flatten Layer type
 *            6. Loss Layer type
 *            7. Activation Layer type
 *            8. Unknown
 */
typedef enum {
  LAYER_IN,
  LAYER_FC,
  LAYER_BN,
  LAYER_CONV2D,
  LAYER_POOLING2D,
  LAYER_FLATTEN,
  LAYER_LOSS,
  LAYER_ACTIVATION,
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
  Layer()
    : last_layer(false),
      bias_init_zero(false),
      type(LAYER_UNKNOWN),
      loss(0.0),
      cost(COST_UNKNOWN),
      activation_type(ACT_NONE),
      bn_follow(false),
      weight_decay(),
      weight_ini_type(WEIGHT_UNKNOWN),
      flatten(false) {}

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
   * @brief     Activation Type Getter
   * @retval    Activation Type.
   */
  ActiType getActivationType() { return this->activation_type; }

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
   * @brief  get if this is last layer of Network
   * @retval last true/false
   */
  bool getLast() { return last_layer; }

  /**
   * @brief  set bias initialize with zero
   * @param[in] zero true/false
   */
  void setBiasZero(bool zero) { bias_init_zero = zero; }

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

  /**
   * @brief  get the loss value added by this layer
   * @retval loss value
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
   * @brief     get gradients
   * @retval    shared ptr of vector of all tensors
   */
  std::shared_ptr<std::vector<Tensor>> getGradients() { return getObjFromRef(gradients); }

  /**
   * @brief     get weights
   * @retval    shared ptr of vector of all tensors
   */
  std::shared_ptr<std::vector<Tensor>> getWeights() { return getObjFromRef(weights); }

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  bool getFlatten() { return flatten; }

  /**
   * @brief     Property Enumeration
   *            0. input shape : string
   *            1. bias zero : bool
   *            2. normalization : bool
   *            3. standardization : bool
   *            4. activation : string (type)
   *            5. epsilon : float
   *            6. weight_decay : string (type)
   *            7. weight_decay_lambda : float
   *            8. unit : int
   *            9. weight_ini : string (type)
   *            10. filter_size : int
   *            11. kernel_size : ( n , m )
   *            12. stride : ( n, m )
   *            13. padding : ( n, m )
   *            14, pooling_size : ( n,m )
   *            15, pooling : max, average, global_max, global_average
   */
  enum class PropertyType {
    input_shape = 0,
    bias_init_zero = 1,
    normalization = 2,
    standardization = 3,
    activation = 4,
    epsilon = 5,
    weight_decay = 6,
    weight_decay_lambda = 7,
    unit = 8,
    weight_ini = 9,
    filter = 10,
    kernel_size = 11,
    stride = 12,
    padding = 13,
    pooling_size = 14,
    pooling = 15,
    flatten = 16,
    unknown = 17
  };

protected:

/**
 * @brief        check if current layer's weight decay type is l2norm
 * @return       bool is weightdecay type is L2 Norm
 */
  bool isWeightDecayL2Norm() {
    return weight_decay.type == WeightDecayType::l2norm;
  }
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
  bool bias_init_zero;

  /**
   * @brief     Layer type
   */
  LayerType type;

  /**
   * @brief     Loss value added by this layer
   */
  float loss;

  /**
   * @brief     Cost type for this network consisting of this layer
   */
  CostType cost;

  ActiType activation_type;

  bool bn_follow;

  WeightDecayParam weight_decay;

  WeightIniType weight_ini_type;

  /**
   * @brief   Output of this layer should be flattened
   */
  bool flatten;

  /**
   * @brief     Gradient for the weights in this layer
   * @note      The order of gradients should match the order in weights
   */
  std::vector<std::reference_wrapper<Tensor>> gradients;

  /**
   * @brief     weights in this layer
   * @note      The weights are combined with their corresponding bias
   *            For example- with W0, W1, B0 and B1, weights would be of format
   *            {W0, B0, W1, B1}.
   */
  std::vector<std::reference_wrapper<Tensor>> weights;

private:
  /**
   * @brief     Convert vector of reference to vector of objects
   */
  std::shared_ptr<std::vector<Tensor>> getObjFromRef(
      std::vector<std::reference_wrapper<Tensor>> &elements);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYERS_H__ */
