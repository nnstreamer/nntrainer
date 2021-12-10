// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_wrapped.cpp
 * @date   10 December 2021
 * @brief  This is Optimizer Wrapped interface class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details wraps the optimizer and learning rate scheduler together
 */

#include <app_context.h>
#include <common_properties.h>
#include <lr_scheduler_constant.h>
#include <lr_scheduler_exponential.h>
#include <node_exporter.h>
#include <optimizer_wrapped.h>

namespace nntrainer {

/**
 * @brief Optimizer wrapped creator with constructor for optimizer
 */
std::unique_ptr<OptimizerWrapped>
createOptimizerWrapped(const ml::train::OptimizerType &type,
                       const std::vector<std::string> &properties) {
  auto &ac = nntrainer::AppContext::Global();
  return createOptimizerWrapped(ac.createObject<OptimizerCore>(type),
                                properties);
}

/**
 * @brief Optimizer wrapped creator with constructor for optimizer
 */
std::unique_ptr<OptimizerWrapped>
createOptimizerWrapped(const std::string &type,
                       const std::vector<std::string> &properties) {
  auto &ac = nntrainer::AppContext::Global();
  return createOptimizerWrapped(ac.createObject<OptimizerCore>(type),
                                properties);
}

/**
 * @brief Optimizer wrapped creator with constructor for optimizer
 */
std::unique_ptr<OptimizerWrapped>
createOptimizerWrapped(std::unique_ptr<OptimizerCore> &&opt,
                       const std::vector<std::string> &properties) {
  auto opt_wrapped = std::make_unique<OptimizerWrapped>(std::move(opt));

  opt_wrapped->setProperty(properties);
  return opt_wrapped;
}

OptimizerWrapped::OptimizerWrapped(std::unique_ptr<OptimizerCore> &&opt) :
  optimizer(std::move(opt)),
  lr_sched(),
  props(props::LearningRate(), props::DecayRate(), props::DecaySteps()) {
  std::get<props::LearningRate>(props).set(optimizer->getDefaultLearningRate());
}

const std::string OptimizerWrapped::getType() const {
  return optimizer->getType();
}

void OptimizerWrapped::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, props);
  optimizer->setProperty(remain_props);
}

double OptimizerWrapped::getLearningRate(size_t iteration) {
  return lr_sched->getLearningRate(iteration);
}

void OptimizerWrapped::applyGradient(RunOptimizerContext &context) {
  optimizer->applyGradient(context);
}

void OptimizerWrapped::exportTo(Exporter &exporter,
                                const ExportMethods &method) const {
  optimizer->exportTo(exporter, method);
  lr_sched->exportTo(exporter, method);
}

void OptimizerWrapped::finalize() {
  auto const &props_lr = std::get<props::LearningRate>(props);
  auto const &props_dr = std::get<props::DecayRate>(props);
  auto const &props_ds = std::get<props::DecaySteps>(props);

  /** if lr_sched already set and property not empty, error */
  bool props_empty = props_lr.empty() & props_dr.empty() & props_ds.empty();

  NNTR_THROW_IF(props_empty && !lr_sched, std::invalid_argument)
    << "Learning rate scheduler not set for the optimizer " << getType();
  NNTR_THROW_IF(!props_empty && lr_sched, std::invalid_argument)
    << "Multiple learning rate schedulers set for the optimizer " << getType();

  /** if lr_sched not set, make lr_sched from properties */
  if (!props_empty) {
    if (!props_dr.empty() || !props_ds.empty()) {
      lr_sched = std::make_unique<ExponentialLearningRateScheduler>();
      if (!props_dr.empty())
        lr_sched->setProperty({"decay_rate=" + std::to_string(props_dr.get())});
      if (!props_ds.empty())
        lr_sched->setProperty(
          {"decay_steps=" + std::to_string(props_ds.get())});
    } else {
      lr_sched = std::make_unique<ConstantLearningRateScheduler>();
    }

    if (!props_lr.empty())
      lr_sched->setProperty(
        {"learning_rate=" + std::to_string(props_lr.get())});
  }

  lr_sched->finalize();
  optimizer->finalize();
}

void OptimizerWrapped::read(std::ifstream &file) { optimizer->read(file); }

void OptimizerWrapped::save(std::ofstream &file) { optimizer->save(file); }

std::vector<TensorDim>
OptimizerWrapped::getOptimizerVariableDim(const TensorDim &dim) {
  return optimizer->getOptimizerVariableDim(dim);
}

void OptimizerWrapped::setLearningRateScheduler(
  std::unique_ptr<nntrainer::LearningRateScheduler> &&lrs) {
  lr_sched = std::move(lrs);
}

nntrainer::LearningRateScheduler *OptimizerWrapped::setLearningRateScheduler() {
  return lr_sched.get();
}

} // namespace nntrainer