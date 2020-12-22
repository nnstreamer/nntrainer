// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	manager.h
 * @date	30 Nov 2020
 * @brief	This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __MANAGER_H__
#define __MANAGER_H__
#ifdef __cplusplus

#include <functional>
#include <vector>

#include <var_grad.h>
#include <weight.h>

namespace nntrainer {

/**
 * @class   Manager
 * @brief   manager of nntrainer
 */
class Manager {

public:
  /**
   * @brief     Constructor of Manager
   */
  Manager() :
    max_weight_size(0),
    max_derivative_size(0),
    enable_gradient_memory_opt(true),
    enable_derivative_memory_opt(true),
    enable_activation_memory_opt(true) {}

  /**
   * @brief     Destructor of Manager
   */
  ~Manager() {}

  /**
   * @brief     Add weight to be tracked and updated with nntrainer
   *
   * @param w   Weight to be tracked
   */
  void trackWeight(std::reference_wrapper<Weight> w);

  /**
   * @brief     Add weights to be tracked and updated with nntrainer
   *
   * @param ws  Weights to be tracked
   */
  void trackWeights(std::vector<Weight> &ws);

  /**
   * @brief     Get weights tracked with nntrainer
   *
   * @retval    list of weight references
   */
  std::vector<std::vector<std::reference_wrapper<Weight>>> getWeightRefs() {
    return weights;
  }

  /**
   * @brief Enable gradient memory sharing based optimization
   * @param opt True to enable, else false
   */
  void setGradientMemoryOptimization(bool opt) {
    enable_gradient_memory_opt = opt;
  }

  /**
   * @brief Enable derivative memory sharing based optimization
   * @param opt True to enable, else false
   */
  void setDerivativeMemoryOptimization(bool opt) {
    enable_derivative_memory_opt = opt;
  }

  /**
   * @brief Enable derivative memory sharing based optimization
   * @param opt True to enable, else false
   */
  void setInPlaceActivationOptimization(bool opt) {
    enable_activation_memory_opt = opt;
  }

  /**
   * @brief Allocate and initialize the weight variable
   */
  void initialize();

  /**
   * @brief Reset the manager state
   */
  void reset() {
    weights.clear();
    in_outs.clear();
    max_weight_size = 0;
  }

  /**
   * @brief Track the inputs/ouputs of the layer
   * @param[in] layer_name Name of the layer
   * @param[in] input_dim Dimension of the input for the layer
   * @param[in] trainable If the layer is trainable
   * @note Manager is kept independent from the layer object itself
   */
  std::vector<std::shared_ptr<Var_Grad>> &
  TrackLayerInOuts(const std::string &layer_type, const std::string &layer_name,
                   const std::vector<TensorDim> &input_dim);

  /**
   * @brief Track the inputs/ouputs of the layer
   * @param[in] layer_name Name of the layer
   * @note Manager is kept independent from the layer object itself
   */
  void untrackLayerInOuts(const std::string layer_name);

  /**
   * @brief Initialize the inputs/outputs for the layers
   * @param[in] trainable If true, initialize derivates, else, do not.
   */
  void initializeInOuts(bool trainable);

  /**
   * @brief Set the batch size for the inputs/outputs of the layers
   */
  void setBatchSize(unsigned int batch) {
    if (!in_outs.empty() && !in_outs[0].empty()) {
      max_derivative_size /= in_outs[0][0]->getDim().batch();
      max_derivative_size *= batch;
    }
    for (auto &in_out : in_outs)
      for (auto &vg : in_out)
        vg->setBatchSize(batch);
  }

private:
  // TODO: ensure that names of these weights are unique
  /**< Weights of all the layer in the model to be managed */
  std::vector<std::vector<std::reference_wrapper<Weight>>> weights;

  /**< Inputs/outputs of all the layer in the model */
  std::vector<std::vector<std::shared_ptr<Var_Grad>>> in_outs;
  std::vector<bool> is_act_type;

  size_t max_weight_size;     /**< max weight required by a layer */
  size_t max_derivative_size; /**< max derivative required by a layer */

  bool enable_gradient_memory_opt; /**< share memory among all the gradients */
  bool enable_derivative_memory_opt; /**< share memory among all the derivative
                                        and output of the next layer */
  bool enable_activation_memory_opt; /**< Let activation layer work in-place
                                        without allocating output layer for
                                        itself */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MANAGER_H__ */
