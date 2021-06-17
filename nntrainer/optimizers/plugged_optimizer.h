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
   * @copydoc OptimizerDevel::getLearningRate()
   *
   */
  float getLearningRate() const override {
    return optimizer_devel->getLearningRate();
  }

  /**
   * @brief     get Learning Rate for the given iteration
   * @param[in] iteration Iteration for the learning rate
   * @retval    Learning rate in double
   * @detail    the return value of this function and getLearningRate() must
   * match for iteration == 0.
   */
  double getLearningRate(size_t iteration) const override {
    return optimizer_devel->getLearningRate(iteration);
  }

  /**
   * @brief     apply gradient to weight_list
   * @param[in] params Weight list
   * @param[in] iteration nth epoch number
   */
  void applyGradients(std::vector<Weight> &params, int iteration) override {
    optimizer_devel->applyGradients(params, iteration);
  }

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values) override {
    return optimizer_devel->setProperty(values);
  }

  /**
   * @brief setProperty individually
   * @note By passing empty string, this can validate if @a type is valid
   * @param[in] key key to be passed as string
   * @param[in] value value to be passed, if empty string is passed, do nothing
   * but throws error when @a type is invalid
   * @exception exception::not_supported     when string type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  void setProperty(const std::string &key, const std::string &value) override {
    optimizer_devel->setProperty(key, value);
  }

  /**
   * @brief     initialize optimizer.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize() override { return optimizer_devel->initialize(); }

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
   * @brief     validate the optimizer
   */
  void checkValidation() const override { optimizer_devel->checkValidation(); }

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

protected:
  /**
   * @brief     apply gradient to the given weight
   * @param[in] weight Weight and gradient set to be updated
   * @param[in] num_weights size of the array
   * @param[in] iteration nth epoch number
   * @note weight which is called upon can be assumed to be trainable
   */
  void applyGradient(Weight &weight, double updated_lr,
                     int iteration) override {
    throw std::runtime_error(
      "this is a protected function and must not be called");
  }

private:
  nntrainer::Optimizer *optimizer_devel;
  nntrainer::DestroyOptimizerFunc destroy_func;
};

} // namespace internal
} // namespace nntrainer

#endif // __PLUGGED_OPTIMIZER_H__
