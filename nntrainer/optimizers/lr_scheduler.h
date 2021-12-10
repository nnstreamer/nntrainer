// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lr_scheduler.h
 * @date   09 December 2021
 * @brief  This is Learning Rate Scheduler interface class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LEARNING_RATE_SCHEDULER__
#define __LEARNING_RATE_SCHEDULER__
#ifdef __cplusplus

#include <string>

namespace nntrainer {

class Exporter;
enum class ExportMethods;

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
   * @brief     Finalize creating the learning rate scheduler
   *
   * @details   Verify that all the needed properties have been and within the
   * valid range.
   * @note      After calling this it is not allowed to
   * change properties.
   */
  virtual void finalize() = 0;

  /**
   * @brief     get Learning Rate for the given iteration
   * @param[in] iteration Iteration for the learning rate
   * @retval    Learning rate in double
   * @detail    the return value of this function and getInitialLearningRate()
   * may not match for iteration == 0 (warmup can lead to different initial
   * learning rates).
   *
   * @note this is non-const function intentionally.
   */
  virtual double getLearningRate(size_t iteration) = 0;

  /**
   * @brief this function helps exporting the learning rate in a predefined
   * format, while workarounding issue caused by templated function type eraser
   *
   * @param     exporter exporter that conatins exporting logic
   * @param     method enum value to identify how it should be exported to
   */
  virtual void exportTo(Exporter &exporter, const ExportMethods &method) const {
  }

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

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LEARNING_RATE_SCHEDULER__ */
