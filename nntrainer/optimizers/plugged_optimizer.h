// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kpaoor <pk.kapoor@samsung.com>
 *
 * @file   plugged_optimizer.h
 * @date   1 June 2021
 * @brief  This file contains a wrapper for a plugged optimizer, INTERNAL USE
 * ONLY
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __PLUGGED_OPTIMIZER_H__
#define __PLUGGED_OPTIMIZER_H__

#include <nntrainer_error.h>
#include <optimizer.h>
#include <optimizer_devel.h>

namespace nntrainer {
namespace internal {

/**
 * @brief   Plugged optimizer class
 */
class PluggedOptimizer : public nntrainer::Optimizer {
public:
  /**
   * @brief Construct a new Plugged Optimizer object
   *
   * @param pluggable OptimizerPluggable structure from the symbol
   */
  PluggedOptimizer(const nntrainer::OptimizerPluggable *pluggable) :
    optimizer_devel(
      dynamic_cast<nntrainer::Optimizer *>(pluggable->createfunc())),
    destroy_func(pluggable->destroyfunc) {
    NNTR_THROW_IF(optimizer_devel == nullptr, std::invalid_argument)
      << "create_func_ for plugged optimizer failed";
  }

  /**
   * @brief Destroy the Plugged Optimizer object
   *
   */
  ~PluggedOptimizer() override { destroy_func(optimizer_devel); }

  /**
   * @brief Move Contruct Plugged Optimizer object
   *
   * @param rhs optimizer to move
   */
  PluggedOptimizer(PluggedOptimizer &&rhs) noexcept = default;

  /**
   * @brief Move assign Plugged Optimizer Object
   *
   * @param rhs optimizer to move
   * @return PluggedOptimizer& *this
   */
  PluggedOptimizer &operator=(PluggedOptimizer &&rhs) = default;

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   *
   */
  double getDefaultLearningRate() const override {
    return optimizer_devel->getDefaultLearningRate();
  }
  /**
   * @brief     apply gradient to weight
   * @param[in] context Optimizer context
   */
  void applyGradient(RunOptimizerContext &context) override {
    optimizer_devel->applyGradient(context);
  }

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  void setProperty(const std::vector<std::string> &values) override {
    optimizer_devel->setProperty(values);
  }

  /**
   * @brief     finalize optimizer.
   */
  void finalize() override { optimizer_devel->finalize(); }

  /**
   * @brief     Read Training optimizer paramters from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file) override { optimizer_devel->read(file); }

  /**
   * @brief     Save Training optimizer paramters from file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file) override { optimizer_devel->save(file); }

  /**
   * @brief     Get dimension of extra variables if the optimizer needs any.
   * @param dim Dimension of tensor to be added as a optimizer variable
   * @return    Vector of dimensions
   */
  virtual std::vector<TensorDim>
  getOptimizerVariableDim(const TensorDim &dim) override {
    return optimizer_devel->getOptimizerVariableDim(dim);
  }

  /**
   * @brief     get Optimizer Type
   * @retval    Optimizer type
   */
  const std::string getType() const override {
    return optimizer_devel->getType();
  }

private:
  nntrainer::Optimizer *optimizer_devel;
  nntrainer::DestroyOptimizerFunc destroy_func;
};

} // namespace internal
} // namespace nntrainer

#endif // __PLUGGED_OPTIMIZER_H__
