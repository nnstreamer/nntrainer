// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   dynamic_training_optimization.h
 * @date   4 January 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Dynamic Training Optimization for Neural Network
 *
 * Dynamic training aims to optimize the cost of applying the gradient.
 * The cost of applying the gradient includes the cost of the optimizer (adam,
 * etc) where the optimizer variables are updated, and the cost of actually
 * updating the weights (which can be non-trivial with bigger weights and
 * distributed training).
 *
 * There are two supported modes:
 * 1. Gradient Mode: The already calculated gradient is used to estimate if this
 * gradient must be used to update the weight, or if this update must be
 * skipped.
 *
 * 2. Derivative Mode: This mode tries to estimate an approximate gradient with
 * low cost in order to save the cost of calculating gradient. This cost of
 * calculating gradient is wasted if the gradient is not going to be applied.
 *
 * There are two supported reduction operations which reduce the gradient and
 * the weight to a single value in order to compare it with a threshold.
 * If the reduced value is less than threshold, the update is performed with
 * some probabilty proportional to the value. If the reduced value is higher
 * than threshold, then the update is always performed.
 *
 */

#ifndef __DYNAMIC_TRAINING_OPT_H__
#define __DYNAMIC_TRAINING_OPT_H__
#ifdef __cplusplus

#include <random>
#include <vector>

#include <layer_internal.h>
#include <tensor.h>

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
  DynamicTrainingOptimization(int threshold_ = 1, int skip_n_iter = 1);

  /**
   * @brief     Set threshold for optimization
   */
  void setThreshold(float threshold_) {
    if (threshold_ < epsilon)
      throw std::invalid_argument("Threshold is too small or negative");

    threshold = threshold_;
  };

  /**
   * @brief     Set the reduce operation for dynamic optimization
   */
  void setOp(const std::string &op) {
    if (op == dft_opt_max)
      reduce_op = reduceByMax;
    else if (op == dft_opt_norm)
      reduce_op = reduceByNorm;
    else
      throw std::invalid_argument(
        "Unsupported reduction op in dynamic training");
  };

  /**
   * @brief     Enable the optimization
   */
  void enable() { enabled = true; }

  /**
   * @brief     Disable the optimization
   */
  void disable() { enabled = false; }

  /**
   * @brief     Set the mode for optimization
   */
  void setMode(const std::string &mode_) {
    calc_ratio_mode = mode_;
    if (mode_ == dft_opt_mode_derivative)
      calc_ratio_op = ratioUsingDerivative;
    else if (mode_ == dft_opt_mode_gradient)
      calc_ratio_op = ratioUsingGradient;
    else
      throw std::invalid_argument("Unsupported mode in dynamic training");
  }

  /**
   * @brief     Check if the derivative mode is used for optimization
   * @note Use the derivative to calculate an approximate gradient to estimate
   * if the actual gradient needs applying
   */
  bool isDerivativeMode() {
    if (enabled && calc_ratio_mode == dft_opt_mode_derivative)
      return true;
    return false;
  }

  /**
   * @brief     Check if the gradient mode is used for optimization
   * @note Use the gradient to estimate if this gradient needs applying
   */
  bool isGradientMode() {
    if (enabled && calc_ratio_mode == dft_opt_mode_gradient)
      return true;
    return false;
  }

  /**
   * @brief    Initial iterations to not perform dynamic training optimization
   * @note If the current iteration is less than skip_n_iterations, the weights
   * will updated and dynamic training optimization will not be performed.
   *
   */
  void setSkipIterations(int skip_n_iter) { skip_n_iterations = skip_n_iter; }

  /**
   * @brief     Check if the given weights can skip updating
   * @param[in] weights All the weight tensors for a layer
   * @param[in] input Input tensor for a layer
   * @param[in] output Output tensor for a layer, from forward operation
   * @param[in] opt Optimizer used to update the layer weights
   * @param[in] iteration Current iteration number in training
   * @note true if should be applied, else false
   */
  bool checkIfApply(const std::vector<Weight> &weights,
                    const std::shared_ptr<Var_Grad> &input,
                    const std::shared_ptr<Var_Grad> &output,
                    const std::shared_ptr<Optimizer> &opt, int iteration);

  /**
   * @brief     Check if the given weight can skip updating
   * @param[in] weight Weight tensor for a layer
   * @param[in] input Input tensor for a layer
   * @param[in] output Output tensor for a layer, from forward operation
   * @param[in] opt Optimizer used to update the layer weights
   * @param[in] iteration Current iteration number in training
   * @note true if should be applied, else false
   */
  bool checkIfApply(const Weight &weight,
                    const std::shared_ptr<Var_Grad> &input,
                    const std::shared_ptr<Var_Grad> &output,
                    const std::shared_ptr<Optimizer> &opt, int iteration);

  /**< Different types of reduce operations */
  static const std::string dft_opt_max;
  static const std::string dft_opt_norm;

  /**< Different types of optimization modes */
  static const std::string dft_opt_mode_gradient;
  static const std::string dft_opt_mode_derivative;

private:
  std::mt19937 rng; /**< random number generator */
  std::uniform_real_distribution<float>
    dist;                      /**< uniform random distribution */
  float threshold;             /**< threshold to decide when to skip updating */
  bool enabled;                /**< if optimization is enabled */
  float epsilon;               /**< epsilon to skip overflow */
  int skip_n_iterations;       /**< skip initial iterations from optimization */
  std::string calc_ratio_mode; /**< the mode to calc the ratio */

  std::function<float(Tensor const &)>
    reduce_op; /**< operation to reduce update ratio to value */
  std::function<float(const Weight &, const std::shared_ptr<Var_Grad> &,
                      const std::shared_ptr<Var_Grad> &,
                      std::function<float(Tensor const &)> reduce_op)>
    calc_ratio_op; /**< calculate the ratio of update to the weight */

  /**
   * @brief   Calculate the ratio of update to the weight using derivative
   * @param[in] weight Weight tensor for a layer
   * @param[in] input Input tensor for a layer
   * @param[in] output Output tensor for a layer, from forward operation
   * @param[in] reduce_op Operation to reduce the ratio
   */
  static float
  ratioUsingDerivative(const Weight &weight,
                       const std::shared_ptr<Var_Grad> &input,
                       const std::shared_ptr<Var_Grad> &output,
                       std::function<float(Tensor const &)> reduce_op);

  /**
   * @brief   Calculate the ratio of update to the weight using gradient
   * @param[in] weight Weight tensor for a layer
   * @param[in] input Input tensor for a layer
   * @param[in] output Output tensor for a layer, from forward operation
   * @param[in] reduce_op Operation to reduce the ratio
   */
  static float
  ratioUsingGradient(const Weight &weight,
                     const std::shared_ptr<Var_Grad> &input,
                     const std::shared_ptr<Var_Grad> &output,
                     std::function<float(Tensor const &)> reduce_op);

  /**
   * @brief   Check if the update should be applied or skipped
   * @note true if should be applied, else false
   */
  bool checkIfApply(float reduced_ratio, float learning_rate);

  /**
   * @brief     Operation to decide if update should be skipped
   * @note      Calculate l0 norm of the tensor
   */
  static float reduceByMax(Tensor const &ratio);

  /**
   * @brief     Operation to decide if update should be skipped
   * @note      Calcalate l2 norm of the tensor averaged by its size
   */
  static float reduceByNorm(Tensor const &ratio);
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __DYNAMIC_TRAINING_OPT_H__ */
