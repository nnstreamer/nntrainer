// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   momentum.h
 * @date   1 June 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Momentum optimizer.
 */
#ifndef __MOMENTUM_H__
#define __MOMENTUM_H__
#ifdef __cplusplus

#include <optimizer_impl.h>

namespace nntrainer {

/**
 * @class   Momentum optimizer class
 * @brief   Momentum optimizer
 */
class Momentum : public OptimizerImpl {
public:
  /**
   * @brief     Constructor of Optimizer Class
   */
  template <typename... Args>
  Momentum(float lr = 0.001f, double m = 0.9f, Args... args) :
    OptimizerImpl(lr, args...),
    momentum(m) {}

  /**
   * @copydoc applyGradient(Weight &weight, int tensor_idx, double updated_lr,
   * int iteration)
   */
  void applyGradient(Weight &weight, double updated_lr, int iteration);

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const { return Momentum::type; }

  /**
   * @copydoc setProperty(const std::string &key,
                           const std::string &value)
   */
  void setProperty(const std::string &key, const std::string &value);

  /**
   * @copydoc Optimizer::addOptimizerVariable(std::vector<Weight> &params)
   */
  void addOptimizerVariable(std::vector<Weight> &params);

  /**
   * @brief get momentum
   */
  double getMomentum() { return momentum; };

  static const std::string type;

private:
  double momentum; /** momentum for grad */
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __MOMENTUM_H__ */
