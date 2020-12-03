// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	adam.h
 * @date	6 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is the Adam optimizer.
 */
#ifndef __ADAM_H__
#define __ADAM_H__
#ifdef __cplusplus

#include <optimizer_internal.h>

namespace nntrainer {

/**
 * @class   Adam optimizer class
 * @brief   Adam optimizer
 */
class Adam : public Optimizer {
public:
  /**
   * @brief     Constructor of Optimizer Class
   */
  template <typename... Args>
  Adam(float lr = 0.001f, double b1 = 0.9f, double b2 = 0.999f,
       double ep = 1.0e-7f, Args... args) :
    Optimizer(lr, args...),
    beta1(b1),
    beta2(b2),
    epsilon(ep) {}

  /**
   * @copydoc apply_gradient(Weight &weight, int tensor_idx, double updated_lr,
   * int iteration)
   */
  void apply_gradient(Weight &weight, int tensor_idx, double updated_lr,
                      int iteration);

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const { return Adam::type; }

  /**
   * @copydoc   getLearningRate(int iteration)
   */
  double getLearningRate(int iteration);

  /**
   * @copydoc setProperty(const PropertyType type,
                           const std::string &value = "")
   */
  void setProperty(const PropertyType type, const std::string &value = "");

  /**
   * @copydoc Optimizer::initialize(std::vector<Weight> params, bool setTensor)
   */
  int initialize(std::vector<Weight> &params, bool setTensor);

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

  static const std::string type;

private:

  double beta1;   /** momentum for grad */
  double beta2;   /** momentum for grad**2 */
  double epsilon; /** epsilon to protect overflow */
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __ADAM_H__ */
