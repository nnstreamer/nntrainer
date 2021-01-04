// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   activation_layer.cpp
 * @date   4 January 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Dynamic Training Optimization for Neural Network
 *
 */

#ifndef __DYNAMIC_TRAINING_OPT_H__
#define __DYNAMIC_TRAINING_OPT_H__
#ifdef __cplusplus

#include <random>
#include <vector>

#include <layer_internal.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

/**
 * @class   DynamicTraining Optimization
 * @brief   Dynamic Training Optimization
 */
class DynamicTrainingOptimization {
public:
  /**
   * @brief     Constructor of DynamicFineTuning Optimization
   */
  DynamicTrainingOptimization(int threshold_ = 1, int skip_n_iter = 1) :
    threshold(threshold_),
    enabled(false),
    epsilon(1e-7),
    skip_n_iterations(skip_n_iter) {
    reduce_op = reduce_by_norm;
    rng.seed(getSeed());
    dist = std::uniform_real_distribution<float>(0.0, 1.0);
  }

  /**
   * @brief     Set threshold for optimization
   */
  void setThreshold(float threshold_) { threshold = threshold_; };

  /**
   * @brief     Set the reduce operation for dynamic optimization
   */
  void setOp(std::string op) {
    enabled = true;
    if (op == dft_opt_max)
      reduce_op = reduce_by_max;
    else if (op == dft_opt_norm)
      reduce_op = reduce_by_norm;
    else
      enabled = false;
  };

  /**
   * @brief     Set initial iteraions to skip from optimization
   */
  void setSkipIterations(int skip_n_iter) { skip_n_iterations = skip_n_iter; }

  /**
   * @brief     Check if the given weights can skip updating
   */
  std::vector<bool> checkIfApply(const std::vector<Weight> &weights,
                                 const std::shared_ptr<Var_Grad> input,
                                 const std::shared_ptr<Var_Grad> output,
                                 const std::shared_ptr<Optimizer> opt,
                                 int iteration) {
    if (!enabled)
      return std::vector<bool>(weights.size(), true);

    std::vector<bool> apply;
    apply.reserve(weights.size());

    for (auto const &weight : weights)
      apply.push_back(checkIfApply(weight, input, output, opt, iteration));

    return apply;
  }

  /**
   * @brief     Check if the given weight can skip updating
   */
  bool checkIfApply(const Weight &weight,
                    const std::shared_ptr<Var_Grad> &input,
                    const std::shared_ptr<Var_Grad> &output,
                    const std::shared_ptr<Optimizer> &opt, int iteration) {
    // by gradient
    if (iteration < skip_n_iterations)
      return true;

    Tensor &weight_grad = weight.getGradientRef();
    Tensor &weight_var = weight.getVariableRef();

    if (!weight.getTrainable() || weight_grad.uninitialized())
      return true;

    Tensor ratio = weight_grad.divide(weight_var);

    // by derivative
    // Tensor ratio = output.getGradientRef().divide(weight.getVariableRef());
    // ratio.multiply_i(input.getVariableRef());

    /**
     * If the reduced update ratio is higher than 1, then always apply update.
     * If the reduced update raito is less than 1, then apply it with
     * probability = update ratio
     */
    if (dist(rng) <
        reduce_op(ratio) * ((float)opt->getLearningRate(iteration)) / threshold)
      return false;

    return true;
  }

  /**
   * @brief     Operation to decide if update should be skipped
   * @note      Calculate l0 norm of the tensor
   */
  static float reduce_by_max(Tensor const &ratio) { return ratio.max_abs(); }

  /**
   * @brief     Operation to decide if update should be skipped
   * @note      Calcalate l2 norm of the tensor averaged by its size
   */
  static float reduce_by_norm(Tensor const &ratio) {
    float l2norm = ratio.l2norm();
    return (l2norm * l2norm) / ratio.length();
  }

  /**< Different types of reduce operations */
  static const std::string dft_opt_off;
  static const std::string dft_opt_max;
  static const std::string dft_opt_norm;

private:
  std::mt19937 rng; /**< random number generator */
  std::uniform_real_distribution<float>
    dist;                /**< uniform random distribution */
  float threshold;       /**< threshold to decide when to skip updating */
  bool enabled;          /**< if optimization is enabled */
  float epsilon;         /**< epsilon to skip overflow */
  int skip_n_iterations; /**< skip initial iterations from optimization */
  std::function<float(Tensor const &)>
    reduce_op; /**< operation to reduce update ratio to value */
};

/**< Different types of reduce operations */
const std::string dft_opt_off = "off";
const std::string dft_opt_max = "max";
const std::string dft_opt_norm = "norm";

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __DYNAMIC_TRAINING_OPT_H__ */
