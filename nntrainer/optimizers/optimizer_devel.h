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
#include <optimizer_context.h>
#include <tensor.h>

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
   * @brief     apply gradient to weight
   * @param[in] context Optimizer context
   */
  virtual void applyGradient(RunOptimizerContext &context) = 0;

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   */
  virtual void setProperty(const std::vector<std::string> &values);

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
   * @brief     initialize optimizer.
   */
  virtual void initialize(){};

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
