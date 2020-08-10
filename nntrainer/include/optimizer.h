/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	optimizer.h
 * @date	08 April 2020
 * @brief	This is Optimizer classes of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#ifdef __cplusplus

#include <iostream>
#include <memory>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief UpdatableParam that could be updated thorugh optimizer
 */
struct UpdatableParam {
  Tensor weight;    /**<  weight to be updated and used */
  Tensor grad;      /**<  gradient for the weight */
  std::string name; /**< name of the parameter */
};

/**
 * @brief     Enumeration of Optimizer
 *            0. SGD
 *            1. ADAM
 *            2. Unknown
 */
enum class OptType { sgd = 0, adam = 1, unknown = 2 };

/**
 * @brief     Enumeration of Weight Decay type
 *            0. L2Norm
 *            1. Regression
 *            2. Unknown (equivalent to none)
 */
enum class WeightDecayType { l2norm = 0, regression = 1, unknown = 2 };

/**
 * @brief     type for the Weight Decay hyper-parameter
 */
typedef struct WeightDecayParam_ {
  WeightDecayType type;
  float lambda;

  WeightDecayParam_() : type(WeightDecayType::unknown), lambda(0.0f) {}
} WeightDecayParam;

/**
 * @brief     type for the Optimizor to save hyper-parameter
 */
typedef struct _OptParam {
  float learning_rate;
  double beta1;
  double beta2;
  double epsilon;
  float decay_rate;
  float decay_steps;
  bool continue_train; /** Continue training with previous tensors for adam */

  _OptParam(OptType type = OptType::adam) :
    learning_rate(0.001f),
    beta1(0.9f),
    beta2(0.999f),
    epsilon(1.0e-7f),
    decay_rate(1.0f),
    decay_steps(-1.0f),
    continue_train(false) {
    if (type == OptType::sgd) {
      learning_rate = 0.01f;
    }
  }
} OptParam;

class Optimizer {
public:
  /**
   * @brief     Constructor of Optimizer Class
   */
  Optimizer() : type(OptType::unknown), popt() {}

  Optimizer(const OptType type, OptParam popt);

  /**
   * @brief     Destructor of Optimizer Class
   */
  ~Optimizer() {}

  /**
   * @brief  copy constructor
   * @param[in] rhs Optimizer to be copied
   */
  Optimizer(const Optimizer &rhs) = default;

  /**
   * @brief  copy assignment operator
   * @param[in] rhs Optimizer to be copied
   */
  Optimizer &operator=(const Optimizer &rhs) = default;

  /**
   *  @brief  Move constructor of Conv 2D Layer.
   *  @param[in] Conv2dLayer &&
   */
  Optimizer(Optimizer &&rhs) = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Optimizer to be moved.
   */
  Optimizer &operator=(Optimizer &&rhs) = default;

  /**
   * @brief     set Optimizer Type
   * @param[in] t Optimizer type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setType(OptType t);

  /**
   * @brief     get Optimizer Type
   * @retval    Optimizer type
   */
  OptType getType() { return type; };

  /**
   * @brief     get Learning Rate
   * @retval    Learning rate
   */
  float getLearningRate() { return popt.learning_rate; };

  /**
   * @brief     get Decay Rate for learning rate decay
   * @retval    decay rate
   */
  float getDecayRate() { return popt.decay_rate; };

  /**
   * @brief     get Decay Steps for learning rate decay
   * @retval    decay steps
   */
  float getDecaySteps() { return popt.decay_steps; };

  /**
   * @brief     set Optimizer Parameters
   * @param[in] p Optimizer Parameter : OptParam
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setOptParam(OptParam p);

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     get Optimizer Parameters
   * @retval OptParam
   */
  OptParam getOptParam() { return popt; };

  /**
   * @brief     initialize optimizer. Initialize Weight if it is adam
   * @param[in] params UpdatableParam list
   * @param[in] param_size size of the array
   * @param[in] setTensor true if the layer need weight update.
   *            Input Layer and Batch Normalization layer won't need it.
   *            Therefore, it sets false.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(std::shared_ptr<UpdatableParam> params,
                 unsigned int param_size, bool setTensor);

  /**
   * @brief     apply gradient to weights
   * @param[in] params array of updatable params.
   * @param[in] param_size size of the array
   * @param[in] iteration nth epoch number
   */
  void apply_gradients(std::shared_ptr<UpdatableParam> params,
                       unsigned int param_size, int iteration);

  /**
   * @brief     Property Enumeration
   * learning_rate : float ,
   * decay_rate : float,
   * decay_steps : float,
   * beta1 : float,
   * beta2 : float,
   * epsilon : float,
   */
  enum class PropertyType {
    learning_rate = 0,
    decay_rate = 1,
    decay_steps = 2,
    beta1 = 3,
    beta2 = 4,
    epsilon = 5,
    continue_train = 6,
    unknown = 7,
  };

  /**
   * @brief     Read Training optimizer paramters from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file);

  /**
   * @brief     Save Training optimizer paramters from file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file);

  /**
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "Optimizer"; };

private:
  /**
   * @brief Optimizer Type
   */
  OptType type;

  /**
   * @brief Optimizer Hyper Parmeters
   */
  OptParam popt;

  /**
   * @brief Internal Tensors for adam Optimizer
   */
  std::vector<std::pair<Tensor, Tensor>> weight_mv;
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __OPTIMIZER_H__ */
