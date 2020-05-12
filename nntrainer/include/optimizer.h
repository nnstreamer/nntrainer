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

#include "tensor.h"
#include <iostream>

namespace nntrainer {

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
 *            2. Unknown
 */
enum class WeightDecayType { l2norm = 0, regression = 1, unknown = 2 };

/**
 * @brief     type for the Weight Decay hyper-parameter
 */
typedef struct {
  WeightDecayType type;
  float lambda;
} WeightDecayParam;

/**
 * @brief     type for the Optimizor to save hyper-parameter
 */
typedef struct {
  float learning_rate;
  double beta1;
  double beta2;
  double epsilon;
  float decay_rate;
  float decay_steps;
} OptParam;

class Optimizer {
public:
  /**
   * @brief     Constructor of Optimizer Class
   */
  Optimizer();

  /**
   * @brief     Destructor of Optimizer Class
   */
  ~Optimizer(){};

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
   * @param[in] height height of Weight
   * @param[in] width width of Weight
   * @param[in] setTensor true if the layer need wieght update.
   *            Input Layer and Batch Noramlization layer won't need it.
   *            Therefore, it sets false.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(unsigned int height, unsigned int width, bool setTensor);

  /**
   * @brief     calculate optimizer and Update Weight & Bais
   * @param[in] dJdW Weight derivative
   * @param[in] dJdB Bias derivative
   * @param[in] Weight Weight Tensor
   * @param[in] Bias Bais Tensor
   * @param[in] iteration nth epoch number
   * @param[in] init_zero bool it is true if bias sets zero.
   * @param[in] weight_decay weight decay type & lambda
   */
  void calculate(Tensor &djdw, Tensor &djdb, Tensor &weight, Tensor &bias,
                 int iteration, bool init_zero, WeightDecayParam weight_decay);

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
  };

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
   * @brief Internal Tesnors for adam Optimizer
   */
  Tensor wm;
  Tensor bm;
  Tensor wv;
  Tensor bv;
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __OPTIMIZER_H__ */
