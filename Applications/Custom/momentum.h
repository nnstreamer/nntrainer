// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   momentum.h
 * @date   1 June 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Momentum optimizer.
 */
#ifndef __MOMENTUM_H__
#define __MOMENTUM_H__
#include "tensor_dim.h"
#ifdef __cplusplus

#include <optimizer_impl.h>

#include <base_properties.h>

namespace ml::train {
class TensorDim;
}

namespace nntrainer {
class RunOptimizerContext;
}

namespace custom {

class PropMomentum final : public nntrainer::Property<float> {
public:
  static constexpr const char *key = "momentum";
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @class   Momentum optimizer class for custom optimizer sample
 * @brief   Momentum optimizer
 */
class Momentum final : public nntrainer::OptimizerImpl {
public:
  /**
   * @brief     Constructor of Optimizer Class
   */
  Momentum();

  /**
   * @copydoc applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(nntrainer::RunOptimizerContext &context) override;

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc   getLearningRate(int iteration)
   * @note      this function should be supported if learning rate is depending
   * on iteration. If not, it is recommended to use default implementation in
   * OptimizerImpl
   */
  double getLearningRate(size_t iteration) const override;

  /**
   * @note this function is used to add some additional variable for each
   * parameter
   * @copydoc std::vector<ml::train::TensorDim>
   * getOptimizerVariableDim(const ml::train::TensorDim &dim) override
   */
  std::vector<ml::train::TensorDim>
  getOptimizerVariableDim(const ml::train::TensorDim &dim) override;

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return Momentum::type; }

  inline static const std::string type = "custom_momentum";

private:
  std::tuple<PropMomentum>
    momentum_props; /**< underlying properties for momentum */
};
} /* namespace custom */

#endif /* __cplusplus */
#endif /* __MOMENTUM_H__ */
