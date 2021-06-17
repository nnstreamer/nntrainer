// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_devel.h
 * @date   08 April 2020
 * @brief  This is Optimizer internal interface class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __OPTIMIZER_DEVEL_H__
#define __OPTIMIZER_DEVEL_H__
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

public:
  /**
   * @brief     get Learning Rate
   * @retval    Learning rate in float
   */
  virtual float getLearningRate() const { return getLearningRate(0); }

  /**
   * @brief     get Learning Rate for the given iteration
   * @param[in] iteration Iteration for the learning rate
   * @retval    Learning rate in double
   * @detail    the return value of this function and getLearningRate() must
   * match for iteration == 0.
   */
  virtual double getLearningRate(size_t iteration) const = 0;

  /**
   * @brief     apply gradient to weight_list
   * @param[in] params Weight list
   * @param[in] iteration nth epoch number
   */
  virtual void applyGradients(std::vector<Weight> &params, int iteration);

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setProperty(std::vector<std::string> values);

  /**
   * @brief     Default allowed properties
   * Available for all optimizers
   * - learning_rate : float
   *
   * Available for SGD and Adam optimizers
   * - decay_rate : float,
   * - decay_steps : float,
   *
   * Available for Adam optimizer
   * - beta1 : float,
   * - beta2 : float,
   * - epsilon : float,
   *
   * @todo: convert to string
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
   * @brief setProperty individually
   * @note By passing empty string, this can validate if @a type is valid
   * @param[in] key key to be passed as string
   * @param[in] value value to be passed, if empty string is passed, do nothing
   * but throws error when @a type is invalid
   * @exception exception::not_supported     when string type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  virtual void setProperty(const std::string &key,
                           const std::string &value) = 0;

  /**
   * @brief     initialize optimizer.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize() = 0;

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
   * @brief     validate the optimizer
   */
  virtual void checkValidation() const;

  /**
   * @brief     Get dimension of extra variables if the optimizer needs any.
   * @param dim Dimension of tensor to be added as a optimizer variable
   * @return    Vector of dimensions
   */
  virtual std::vector<TensorDim>
  getOptimizerVariableDim(const TensorDim &dim) = 0;

  /**
   * @brief     get Optimizer Type
   * @retval    Optimizer type
   */
  virtual const std::string getType() const = 0;

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

using CreateOptimizerFunc = ml::train::Optimizer *(*)();
using DestroyOptimizerFunc = void (*)(ml::train::Optimizer *);

/**
 * @brief  Optimizer Pluggable struct that enables pluggable layer
 *
 */
typedef struct {
  CreateOptimizerFunc createfunc;   /**< create function */
  DestroyOptimizerFunc destroyfunc; /**< destory function */
} OptimizerPluggable;

/**
 * @brief pluggable optimizer must have this structure defined
 */
extern "C" OptimizerPluggable ml_train_optimizer_pluggable;

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __OPTIMIZER_DEVEL_H__ */
