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
 * @file	optimizer_internal.h
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

#include <memory>
#include <optimizer.h>
#include <tensor.h>
#include <weight.h>

namespace nntrainer {

/**
 * @class   Optimizer Base class for optimizers
 * @brief   Base class for all optimizers
 */
class Optimizer : public ml::train::Optimizer {

  /** Allow layer to initialize optimizer with itself */
  friend class Layer;

public:
  /**
   * @brief     Default Constructor of Optimizer Class
   */
  Optimizer(float lr, float decay_rate = 1.0f, unsigned int decay_steps = 0,
            float continue_train = false) :
    learning_rate(lr),
    decay_rate(decay_rate),
    decay_steps(decay_steps),
    continue_train(continue_train) {
    checkValidation();
  }

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
  Optimizer(Optimizer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Optimizer to be moved.
   */
  Optimizer &operator=(Optimizer &&rhs) = default;

  /**
   * @brief     get Learning Rate
   * @retval    Learning rate
   */
  float getLearningRate() { return learning_rate; };

  /**
   * @brief     get Decay Rate for learning rate decay
   * @retval    decay rate
   */
  float getDecayRate() { return decay_rate; };

  /**
   * @brief     get Decay Steps for learning rate decay
   * @retval    decay steps
   */
  float getDecaySteps() { return decay_steps; };

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     apply gradient to weight_list
   * @param[in] params Weight list
   * @param[in] iteration nth epoch number
   */
  void apply_gradients(std::vector<Weight> &params, int iteration);

  /**
   * @brief     Read Training optimizer paramters from file
   * @param[in] file input stream file
   */
  virtual void read(std::ifstream &file);

  /**
   * @brief     Save Training optimizer paramters from file
   * @param[in] file output stream file
   */
  virtual void save(std::ofstream &file);

  /**
   * @brief setProperty by PropertyType
   * @note By passing empty string, this can validate if @a type is valid
   * @param[in] type property type to be passed
   * @param[in] value value to be passed, if empty string is passed, do nothing
   * but throws error when @a type is invalid
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  virtual void setProperty(const PropertyType type,
                           const std::string &value = "");

  /**
   * @brief     validate the optimizer
   */
  virtual void checkValidation();

protected:
  /**
   * @brief     get Learning Rate for the given iteration
   * @param[in] iteration Iteration for the learning rate
   * @retval    Learning rate
   */
  virtual double getLearningRate(int iteration);

  float learning_rate;      /** learning rate */
  float decay_rate;         /** decay rate for learning rate */
  unsigned int decay_steps; /** decay steps for learning rate */
  bool continue_train; /** Continue training with previous tensors for adam */

private:
  /**
   * @brief     initialize optimizer. Initialize Weight if it is adam
   * @param[in] params Weight list
   * @param[in] num_weights size of the array
   * @param[in] setTensor true if the layer need weight update.
   *            Input Layer and Batch Normalization layer won't need it.
   *            Therefore, it sets false.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize(std::vector<Weight> &params, bool setTensor);

  /**
   * @brief     apply gradient to the given weight
   * @param[in] weight Weight and gradient set to be updated
   * @param[in] tensor_idx Idx of this tensor in the tensors list
   * @param[in] num_weights size of the array
   * @param[in] iteration nth epoch number
   * @note weight which is called upon can be assumed to be trainable
   */
  virtual void apply_gradient(Weight &weight, int tensor_idx, double updated_lr,
                              int iteration) = 0;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __OPTIMIZER_H__ */
