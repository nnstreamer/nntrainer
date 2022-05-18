// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lr_scheduler_exponential.h
 * @date   09 December 2021
 * @brief  This is Exponential Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LEARNING_RATE_SCHEDULER_EXPONENTIAL__
#define __LEARNING_RATE_SCHEDULER_EXPONENTIAL__
#ifdef __cplusplus

#include <string>

#include <lr_scheduler_constant.h>

namespace nntrainer {

/**
 * @class   Constant Learning Rate Scheduler class
 * @brief   class for constant Learning Rate Schedulers
 */
class ExponentialLearningRateScheduler final
  : public ConstantLearningRateScheduler {

public:
  /**
   * @brief Construct a new exponential learning rate scheduler object
   *
   */
  ExponentialLearningRateScheduler();

  /**
   * @copydoc LearningRateScheduler::getLearningRate(size_t iteration) const
   *
   */
  double getLearningRate(size_t iteration) override;

  /**
   * @copydoc LearningRateScheduler::finalize()
   *
   */
  void finalize() override;

  /**
   * @copydoc LearningRateScheduler::exportTo(Exporter &exporter, const
   * ml::train::ExportMethods& method)
   *
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc LearningRateScheduler::setProperty(const std::vector<std::string>
   * &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc LearningRateScheduler::getType() const
   *
   */
  const std::string getType() const override {
    return ExponentialLearningRateScheduler::type;
  }

  inline static const std::string type = "exponential";

private:
  std::tuple<props::DecayRate, props::DecaySteps> lr_props;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LEARNING_RATE_SCHEDULER_EXPONENTIAL__ */
