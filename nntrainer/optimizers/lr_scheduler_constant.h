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
  NNTR_API ConstantLearningRateScheduler();

  /**
   * @copydoc LearningRateScheduler::getLearningRate(size_t iteration) const
   *
   */
  NNTR_API virtual double getLearningRate(size_t iteration) override;

  /**
   * @copydoc LearningRateScheduler::finalize()
   *
   */
  NNTR_API virtual void finalize() override;

  /**
   * @copydoc LearningRateScheduler::exportTo(Exporter &exporter, const
   * ml::train::ExportMethods& method)
   *
   */
  NNTR_API void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc LearningRateScheduler::setProperty(const std::vector<std::string>
   * &values)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc LearningRateScheduler::getType() const
   *
   */
  NNTR_API const std::string getType() const override {
    return ConstantLearningRateScheduler::type;
  }

  static constexpr const char *type = "constant";

private:
  std::tuple<props::LearningRate> lr_props;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LEARNING_RATE_SCHEDULER_CONSTANT__ */
