// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lr_scheduler_constant.h
 * @date   09 December 2021
 * @brief  This is Constant Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LEARNING_RATE_SCHEDULER_CONSTANT__
#define __LEARNING_RATE_SCHEDULER_CONSTANT__
#ifdef __cplusplus

#include <string>

#include <common_properties.h>
#include <lr_scheduler.h>

namespace nntrainer {

/**
 * @class   Constant Learning Rate Scheduler class
 * @brief   class for constant Learning Rate Schedulers
 */
class ConstantLearningRateScheduler : public LearningRateScheduler {

public:
  /**
   * @brief Construct a new constant learning rate scheduler object
   *
   */
  ConstantLearningRateScheduler();

  /**
   * @copydoc LearningRateScheduler::getLearningRate(size_t iteration) const
   *
   */
  virtual double getLearningRate(size_t iteration) override;

  /**
   * @copydoc LearningRateScheduler::finalize()
   *
   */
  virtual void finalize() override;

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
    return ConstantLearningRateScheduler::type;
  }

  inline static const std::string type = "constant";

private:
  std::tuple<props::LearningRate> lr_props;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LEARNING_RATE_SCHEDULER_CONSTANT__ */
