// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lr_scheduler_step.h
 * @date   13 December 2021
 * @brief  This is Step Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LEARNING_RATE_SCHEDULER_STEP__
#define __LEARNING_RATE_SCHEDULER_STEP__
#ifdef __cplusplus

#include <string>
#include <vector>

#include <lr_scheduler.h>

namespace nntrainer {

namespace props {
class LearningRate;
class Iteration;
} // namespace props

/**
 * @class   Step Learning Rate Scheduler class
 * @brief   class for Step Learning Rate Schedulers
 */
class StepLearningRateScheduler final : public LearningRateScheduler {

public:
  /**
   * @brief Construct a new step learning rate scheduler object
   *
   */
  StepLearningRateScheduler();

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
   * ExportMethods& method)
   *
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const override;

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
    return StepLearningRateScheduler::type;
  }

  inline static const std::string type = "step";

private:
  std::tuple<std::vector<props::LearningRate>, std::vector<props::Iteration>>
    lr_props;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LEARNING_RATE_SCHEDULER_STEP__ */
