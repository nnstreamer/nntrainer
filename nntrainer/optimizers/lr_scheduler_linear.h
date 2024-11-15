// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Hyunwoo LEE <dlgusdn0414@snu.ac.kr>
 *
 * @file   lr_scheduler_linear.h
 * @date   11 November 2024
 * @brief  This is Linear Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyunwoo LEE <dlgusdn0414@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LEARNING_RATE_SCHEDULER_LINEAR__
#define __LEARNING_RATE_SCHEDULER_LINEAR__
#ifdef __cplusplus

#include <string>

#include <common_properties.h>
#include <lr_scheduler.h>

namespace nntrainer {

/**
 * @class   Linear Learning Rate Scheduler class
 * @brief   class for Linear Learning Rate Schedulers
 */
class LinearLearningRateScheduler : public LearningRateScheduler {

public:
  /**
   * @brief Construct a new Linear Learning Rate Scheduler object
   */
  LinearLearningRateScheduler();

  /**
   * @copydoc LearningRateScheduler::getLearningRate(size_t iteration) const
   */
  double getLearningRate(size_t iteration) override;

  /**
   * @copydoc LearningRateScheduler::finalize()
   */
  void finalize() override;

  /**
   * @copydoc LearningRateScheduler::exportTo(Exporter &exporter, const
   * ml::train::ExportMethods& method)
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
   */
  const std::string getType() const override {
    return LinearLearningRateScheduler::type;
  }

  inline static const std::string type = "linear";

private:
  std::tuple<props::MaxLearningRate, props::MinLearningRate, props::DecaySteps>
    lr_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LEARNING_RATE_SCHEDULER_LINEAR__ */
