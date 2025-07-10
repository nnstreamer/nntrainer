// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_wrapped.h
 * @date   10 December 2021
 * @brief  This is Optimizer Wrapped interface class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details wraps the optimizer and learning rate scheduler together
 */

#ifndef __OPTIMIZER_WRAPPER_H__
#define __OPTIMIZER_WRAPPER_H__

#if __cplusplus

#include <string>
#include <vector>

#include <common_properties.h>
#include <lr_scheduler.h>
#include <optimizer.h>
#include <optimizer_devel.h>

namespace nntrainer {

using OptimizerCore = nntrainer::Optimizer;

/**
 * @class   Optimizer Base class for optimizers
 * @brief   Base class for all optimizers
 */
class OptimizerWrapped : public ml::train::Optimizer {
public:
  /**
   * @brief Constructor of OptimizerWrapped class
   * @param opt optimizer to wrap
   *
   */
  NNTR_EXPORT OptimizerWrapped(std::unique_ptr<OptimizerCore> &&opt);

  /**
   * @brief     Destructor of Optimizer Class
   */
  NNTR_EXPORT ~OptimizerWrapped() = default;

  /**
   * Support all the interface requirements by ml::train::Optimizer
   */

  /**
   * @brief     get Optimizer Type
   * @retval    Optimizer type
   */
  NNTR_EXPORT const std::string getType() const override;

  /**
   * @brief     Default allowed properties
   * Available for all optimizers
   * - learning_rate : float
   *
   * Available for SGD and Adam optimizers
   * - decay_rate : float,
   * - decay_steps : float,
   *
   * Available for Adam optimizer
   * - beta1 : float,
   * - beta2 : float,
   * - epsilon : float,
   */

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  NNTR_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Set the Learning Rate Scheduler object
   *
   * @param lrs the learning rate scheduler object
   */
  NNTR_EXPORT int setLearningRateScheduler(
    std::shared_ptr<ml::train::LearningRateScheduler> lrs) override;

  /**
   * Support all the interface requirements by nntrainer::Optimizer
   */

  /**
   * @brief     get Learning Rate for the given iteration
   * @param[in] iteration Iteration for the learning rate
   * @retval    Learning rate in double
   * @detail    the return value of this function and getLearningRate() must
   * match for iteration == 0.
   */
  NNTR_EXPORT double getLearningRate(size_t iteration);

  /**
   * @brief     apply gradient to weight
   * @param[in] context Optimizer context
   */
  NNTR_EXPORT void applyGradient(RunOptimizerContext &context);

  /**
   * @brief this function helps exporting the optimizer in a predefined format,
   * while workarounding issue caused by templated function type eraser
   *
   * @param     exporter exporter that contains exporting logic
   * @param     method enum value to identify how it should be exported to
   */
  NNTR_EXPORT void exportTo(Exporter &exporter,
                            const ml::train::ExportMethods &method) const;

  /**
   * @brief     finalize optimizer.
   */
  NNTR_EXPORT void finalize();

  /**
   * @brief     Read Training optimizer parameters from file
   * @param[in] file input stream file
   */
  NNTR_EXPORT void read(std::ifstream &file);

  /**
   * @brief     Save Training optimizer parameters from file
   * @param[in] file output stream file
   */
  NNTR_EXPORT void save(std::ofstream &file);

  /**
   * @brief     Get dimension of extra variables if the optimizer needs any.
   * @param dim Dimension of tensor to be added as a optimizer variable
   * @return    Vector of dimensions
   */
  NNTR_EXPORT std::vector<TensorDim>
  getOptimizerVariableDim(const TensorDim &dim);

  /**
   * @brief Get the Learning Rate Scheduler object
   *
   * @return the learning rate scheduler object
   */
  NNTR_EXPORT nntrainer::LearningRateScheduler *getLearningRateScheduler();

private:
  std::unique_ptr<OptimizerCore> optimizer; /**< the underlying optimizer */
  std::shared_ptr<nntrainer::LearningRateScheduler>
    lr_sched; /**< the underlying learning rate scheduler */

  /** @todo remove DecayRate, DecaySteps*/
  std::tuple<props::LearningRate, props::DecayRate, props::DecaySteps>
    props; /**< lr scheduler props for backward compatibility */
};

/**
 * @brief Optimizer wrapped creator with constructor for optimizer
 *
 * @params[in] type Type of the optimizer to be constructed
 * @params[in] properties Properties of the optimizer
 */
NNTR_EXPORT std::unique_ptr<OptimizerWrapped>
createOptimizerWrapped(const ml::train::OptimizerType &type,
                       const std::vector<std::string> &properties = {});

/**
 * @brief Optimizer wrapped creator with constructor for optimizer
 *
 * @params[in] type Type of the optimizer to be constructed
 * @params[in] properties Properties of the optimizer
 */
NNTR_EXPORT std::unique_ptr<OptimizerWrapped>
createOptimizerWrapped(const std::string &type,
                       const std::vector<std::string> &properties = {});

/**
 * @brief Optimizer wrapped creator with constructor for optimizer
 *
 * @params[in] type Type of the optimizer to be constructed
 * @params[in] properties Properties of the optimizer
 */
NNTR_EXPORT std::unique_ptr<OptimizerWrapped>
createOptimizerWrapped(std::unique_ptr<OptimizerCore> &&opt,
                       const std::vector<std::string> &properties = {});

} // namespace nntrainer

#endif // __cpluscplus
#endif // __OPTIMIZER_WRAPPER_H__
