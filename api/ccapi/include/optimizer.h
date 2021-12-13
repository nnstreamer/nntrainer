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

#include <nntrainer-api-common.h>

namespace ml {
namespace train {

/** forward declaration */
class LearningRateScheduler;

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
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

  /**
   * @brief Set the Learning Rate Scheduler object
   *
   * @param lrs the learning rate scheduler object
   */
  virtual void setLearningRateScheduler(
    std::unique_ptr<ml::train::LearningRateScheduler> &&lrs) = 0;
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

/**
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

  ptr->setProperty(props);
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

/**
 * @brief     Enumeration of learning type
 */
enum LearningRateType {
  CONSTANT = 0, /**< constant */
  EXPONENTIAL,  /**< exponentially decay */
  STEP          /**< step wise decay */
};

/**
 * @class   Learning Rate Schedulers Base class
 * @brief   Base class for all Learning Rate Schedulers
 */
class LearningRateScheduler {

public:
  /**
   * @brief     Destructor of learning rate scheduler Class
   */
  virtual ~LearningRateScheduler() = default;

  /**
   * @brief     Default allowed properties
   * Constant Learning rate scheduler
   * - learning_rate : float
   *
   * Exponential Learning rate scheduler
   * - learning_rate : float
   * - decay_rate : float,
   * - decay_steps : float,
   *
   * Step Learning rate scheduler
   * - learing_rate : float, float, ...
   * - iteration : uint, uint, ...
   *
   * more to be added
   */

  /**
   * @brief     set learning rate scheduler properties
   * @param[in] values learning rate scheduler properties list
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name = std::string property_val, ...}
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

  /**
   * @brief     get learning rate scheduler Type
   * @retval    learning rate scheduler type
   */
  virtual const std::string getType() const = 0;
};

/**
 * @brief Factory creator with constructor for learning rate scheduler type
 */
std::unique_ptr<ml::train::LearningRateScheduler>
createLearningRateScheduler(const LearningRateType &type,
                            const std::vector<std::string> &properties = {});

/**
 * @brief Factory creator with constructor for learning rate scheduler
 */
std::unique_ptr<ml::train::LearningRateScheduler>
createLearningRateScheduler(const std::string &type,
                            const std::vector<std::string> &properties = {});

/**
 * @brief General LR Scheduler Factory function to create LR Scheduler
 *
 * @param props property representation
 * @return std::unique_ptr<nntrainer::LearningRateScheduler> created object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<LearningRateScheduler, T>::value, T>
            * = nullptr>
std::unique_ptr<LearningRateScheduler>
createLearningRateScheduler(const std::vector<std::string> &props = {}) {
  std::unique_ptr<LearningRateScheduler> ptr = std::make_unique<T>();
  ptr->setProperty(props);
  return ptr;
}

namespace optimizer {
namespace learning_rate {

/**
 * @brief Helper function to create constant learning rate scheduler
 */
inline std::unique_ptr<LearningRateScheduler>
Constant(const std::vector<std::string> &properties = {}) {
  return createLearningRateScheduler(LearningRateType::CONSTANT, properties);
}

/**
 * @brief Helper function to create exponential learning rate scheduler
 */
inline std::unique_ptr<LearningRateScheduler>
Exponential(const std::vector<std::string> &properties = {}) {
  return createLearningRateScheduler(LearningRateType::EXPONENTIAL, properties);
}

/**
 * @brief Helper function to create step learning rate scheduler
 */
inline std::unique_ptr<LearningRateScheduler>
Step(const std::vector<std::string> &properties = {}) {
  return createLearningRateScheduler(LearningRateType::STEP, properties);
}

} // namespace learning_rate
} // namespace optimizer

} // namespace train
} // namespace ml

#else
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_OPTIMIZER_H__
