// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_impl.h
 * @date   18 March 2021
 * @brief  This is base Optimizer implementation class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __OPTIMIZER_IMPL_H__
#define __OPTIMIZER_IMPL_H__
#ifdef __cplusplus

#include <tuple>

#include <base_properties.h>
#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @brief Learning Rate props
 *
 */
class PropsLR : public Property<float> {
public:
  static constexpr const char *key =
    "learning_rate";               /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @brief Decay rate property
 *
 */
class PropsDecayRate : public Property<float> {
public:
  static constexpr const char *key = "decay_rate"; /**< unique key to access */
  using prop_tag = float_prop_tag;                 /**< property type */
};

/**
 * @brief decay steps property
 *
 */
class PropsDecaySteps : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "decay_steps"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                   /**< property type */
};

/**
 * @class   Optimizer Base class for optimizers
 * @brief   Basic implementation class for nntrainer supported optimizers
 */
class OptimizerImpl : public Optimizer {

public:
  /**
   * @brief Construct a new Optimizer Impl object
   *
   */
  OptimizerImpl();

  /**
   * @brief  copy constructor
   * @param[in] rhs OptimizerImpl to be copied
   */
  OptimizerImpl(const OptimizerImpl &rhs) = default;

  /**
   * @brief  copy assignment operator
   * @param[in] rhs OptimizerImpl to be copied
   */
  OptimizerImpl &operator=(const OptimizerImpl &rhs) = default;

  /**
   *  @brief  Move constructor operator.
   * @param[in] rhs OptimizerImpl to be moved
   */
  OptimizerImpl(OptimizerImpl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs OptimizerImpl to be moved.
   */
  OptimizerImpl &operator=(OptimizerImpl &&rhs) noexcept = default;

  /**
   * @brief     get Learning Rate for the given iteration
   * @param[in] iteration Iteration for the learning rate
   * @retval    Learning rate
   */
  double getLearningRate(size_t iteration) const;

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  virtual void setProperty(const std::vector<std::string> &values);

  /**
   * @copydoc Optimizer::exportTo(Exporter &exporter, const ExportMethods&
   * method)
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const override;

  /**
   * @brief     Get dimension of extra variables if the optimizer needs any.
   * @param dim Dimension of tensor to be added as a optimizer variable
   * @return    Vector of dimensions
   */
  virtual std::vector<TensorDim>
  getOptimizerVariableDim(const TensorDim &dim) override {
    return {};
  }

protected:
  std::tuple<PropsLR, PropsDecayRate, PropsDecaySteps> optimizer_impl_props;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __OPTIMIZER_IMPL_H__ */
