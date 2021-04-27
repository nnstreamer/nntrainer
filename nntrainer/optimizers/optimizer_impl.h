// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_impl.h
 * @date   18 March 2021
 * @brief  This is base Optimizer implementation class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __OPTIMIZER_IMPL_H__
#define __OPTIMIZER_IMPL_H__
#ifdef __cplusplus

#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @class   Optimizer Base class for optimizers
 * @brief   Basic implementation class for nntrainer supported optimizers
 */
class OptimizerImpl : public Optimizer {

public:
  /**
   * @brief     Default Constructor of Optimizer Class
   */
  OptimizerImpl(float lr, float decay_rate = 1.0f, unsigned int decay_steps = 0,
                float continue_train = false) :
    Optimizer(),
    learning_rate(lr),
    decay_rate(decay_rate),
    decay_steps(decay_steps),
    continue_train(continue_train) {}

  /**
   * @brief  copy constructor
   * @param[in] rhs OptimizerImpl to be copied
   */
  OptimizerImpl(const OptimizerImpl &rhs) = default;

  /**
   * @brief  copy assignment operator
   * @param[in] rhs OptimizerImpl to be copied
   */
  OptimizerImpl &operator=(const OptimizerImpl &rhs) = default;

  /**
   *  @brief  Move constructor operator.
   * @param[in] rhs OptimizerImpl to be moved
   */
  OptimizerImpl(OptimizerImpl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs OptimizerImpl to be moved.
   */
  OptimizerImpl &operator=(OptimizerImpl &&rhs) = default;

  /**
   * @brief     get Learning Rate
   * @retval    Learning rate in float
   */
  float getLearningRate() const { return learning_rate; };

  /**
   * @brief     get Decay Rate for learning rate decay
   * @retval    decay rate
   */
  float getDecayRate() const { return decay_rate; };

  /**
   * @brief     get Decay Steps for learning rate decay
   * @retval    decay steps
   */
  float getDecaySteps() const { return decay_steps; };

  /**
   * @brief     get Learning Rate for the given iteration
   * @param[in] iteration Iteration for the learning rate
   * @retval    Learning rate
   */
  double getLearningRate(size_t iteration) const;

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
   * @brief     initialize optimizer.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize();

  /**
   * @brief     Add extra variables per weight if the optimizer needs any.
   * @param[in] params Weight list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual void addOptimizerVariable(std::vector<Weight> &params) {}

protected:
  float learning_rate;      /**< learning rate */
  float decay_rate;         /** decay rate for learning rate */
  unsigned int decay_steps; /** decay steps for learning rate */
  bool continue_train; /** Continue training with previous tensors for adam */

private:
  /**
   * @brief     apply gradient to the given weight
   * @param[in] weight Weight and gradient set to be updated
   * @param[in] num_weights size of the array
   * @param[in] iteration nth epoch number
   * @note weight which is called upon can be assumed to be trainable
   */
  virtual void applyGradient(Weight &weight, double updated_lr,
                             int iteration) = 0;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __OPTIMIZER_IMPL_H__ */
