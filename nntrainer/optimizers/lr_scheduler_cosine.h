// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Hyunwoo LEE <dlgusdn0414@snu.ac.kr>
 *
 * @file   lr_scheduler_cosine.h
 * @date   7 October 2024
 * @brief  This is CosineAnnealing Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyunwoo LEE <dlgusdn0414@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LEARNING_RATE_SCHEDULER_COSINE__
#define __LEARNING_RATE_SCHEDULER_COSINE__
#ifdef __cplusplus

#include <string>

#include <common_properties.h>
#include <lr_scheduler.h>

namespace nntrainer {

/**
 * @class   CosineAnnealing Learning Rate Scheduler class
 * @brief   class for CosineAnnealing Learning Rate Schedulers
 */
class CosineAnnealingLearningRateScheduler : public LearningRateScheduler {

public:
  /**
   * @brief Construct a new Cosine Annealing learning rate scheduler object
   *
   */
  CosineAnnealingLearningRateScheduler();

  /**
   * @copydoc LearningRateScheduler::getLearningRate(size_t iteration) const
   */
  virtual double getLearningRate(size_t iteration) override;

  /**
   * @copydoc LearningRateScheduler::finalize()
   *
   */
  virtual void finalize() override;

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
   *
   */
  const std::string getType() const override {
    return CosineAnnealingLearningRateScheduler::type;
  }

  inline static const std::string type = "cosine";

private:
  std::tuple<props::MaxLearningRate, props::MinLearningRate, props::DecaySteps>
    lr_props;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LEARNING_RATE_SCHEDULER_COSINE__ */