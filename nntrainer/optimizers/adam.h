// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   adam.h
 * @date   6 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Adam optimizer.
 */
#ifndef __ADAM_H__
#define __ADAM_H__
#ifdef __cplusplus

#include <optimizer_impl.h>

namespace nntrainer {

/**
 * @class   Adam optimizer class
 * @brief   Adam optimizer
 */
class Adam : public OptimizerImpl {
public:
  /**
   * @brief     Constructor of Optimizer Class
   */
  template <typename... Args>
  Adam(float lr = 0.001f, double b1 = 0.9f, double b2 = 0.999f,
       double ep = 1.0e-7f, Args... args) :
    OptimizerImpl(lr, args...),
    beta1(b1),
    beta2(b2),
    epsilon(ep) {}

  /**
   * @copydoc applyGradient(Weight &weight, int tensor_idx, double updated_lr,
   * int iteration)
   */
  void applyGradient(Weight &weight, double updated_lr, int iteration);

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const { return Adam::type; }

  /**
   * @copydoc   getLearningRate(int iteration)
   */
  double getLearningRate(size_t iteration) const;

  /**
   * @copydoc setProperty(const std::string &key,
                           const std::string &value)
   */
  void setProperty(const std::string &key, const std::string &value);

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim> getOptimizerVariableDim(const TensorDim &dim) override;

  /**
   * @brief get beta1
   */
  double getBeta1() { return beta1; };

  /**
   * @brief get beta2
   */
  double getBeta2() { return beta2; };

  /**
   * @brief get epsilon
   */
  double getEpsilon() { return epsilon; }

  inline static const std::string type = "adam";

private:
  double beta1;   /** momentum for grad */
  double beta2;   /** momentum for grad**2 */
  double epsilon; /** epsilon to protect overflow */
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __ADAM_H__ */
