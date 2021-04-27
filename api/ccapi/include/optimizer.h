// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer.h
 * @date   14 October 2020
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is optimizers interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_OPTIMIZER_H__
#define __ML_TRAIN_OPTIMIZER_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <string>
#include <vector>

#include <ml-api-common.h>
#include <nntrainer-api-common.h>

namespace ml {
namespace train {

/**
 * @brief     Enumeration of optimizer type
 */
enum OptimizerType {
  ADAM = ML_TRAIN_OPTIMIZER_TYPE_ADAM,      /** adam */
  SGD = ML_TRAIN_OPTIMIZER_TYPE_SGD,        /** sgd */
  UNKNOWN = ML_TRAIN_OPTIMIZER_TYPE_UNKNOWN /** unknown */
};

/**
 * @class   Optimizer Base class for optimizers
 * @brief   Base class for all optimizers
 */
class Optimizer {
public:
  /**
   * @brief     Destructor of Optimizer Class
   */
  virtual ~Optimizer() = default;

  /**
   * @brief     get Optimizer Type
   * @retval    Optimizer type
   */
  virtual const std::string getType() const = 0;

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
   */

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual int setProperty(std::vector<std::string> values) = 0;
};

/**
 * @brief Factory creator with constructor for optimizer
 */
std::unique_ptr<Optimizer>
createOptimizer(const std::string &type,
                const std::vector<std::string> &properties = {});

/**
 * @brief Factory creator with constructor for optimizer
 */
std::unique_ptr<Optimizer>
createOptimizer(const OptimizerType &type,
                const std::vector<std::string> &properties = {});

/*
 * @brief General Optimizer Factory function to register optimizer
 *
 * @param props property representation
 * @return std::unique_ptr<ml::train::Optimizer> created object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<Optimizer, T>::value, T> * = nullptr>
std::unique_ptr<Optimizer>
createOptimizer(const std::vector<std::string> &props = {}) {
  std::unique_ptr<Optimizer> ptr = std::make_unique<T>();

  if (ptr->setProperty(props) != ML_ERROR_NONE) {
    throw std::invalid_argument("Set properties failed for optimizer");
  }
  return ptr;
}

namespace optimizer {

/**
 * @brief Helper function to create adam optimizer
 */
inline std::unique_ptr<Optimizer>
Adam(const std::vector<std::string> &properties = {}) {
  return createOptimizer(OptimizerType::ADAM, properties);
}

/**
 * @brief Helper function to create sgd optimizer
 */
inline std::unique_ptr<Optimizer>
SGD(const std::vector<std::string> &properties = {}) {
  return createOptimizer(OptimizerType::SGD, properties);
}

} // namespace optimizer

} // namespace train
} // namespace ml

#else
#error "CPP versions c++14 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_OPTIMIZER_H__
