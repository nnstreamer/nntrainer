// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	weight.h
 * @date	22 September 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Weight Class for Neural Network
 *
 */

#ifndef __WEIGHT_H__
#define __WEIGHT_H__

#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Enumeration of Weight Decay type
 */
enum class WeightRegularizerType {
  l2norm, /** L2 norm regularizer */
  unknown /** Unknown */
};

/**
 * @brief     Enumeration of Weight Initialization Type
 */
enum class WeightInitializer {
  WEIGHT_ZEROS,          /** Zero initialization */
  WEIGHT_ONES,           /** One initialization */
  WEIGHT_LECUN_NORMAL,   /** LeCun normal initialization */
  WEIGHT_LECUN_UNIFORM,  /** uniform initialization */
  WEIGHT_XAVIER_NORMAL,  /** Xavier normal initialization */
  WEIGHT_XAVIER_UNIFORM, /** Xavier uniform initialization */
  WEIGHT_HE_NORMAL,      /** He normal initialization */
  WEIGHT_HE_UNIFORM,     /** He uniform initialization */
  WEIGHT_UNKNOWN         /** Unknown */
};

/**
 * @class   Weight
 * @brief   Weight with gradient, and its corresponding trainable property
 */
class Weight {

  /** Declare layers as friend to get variable/gradient reference */
  friend class Layer;
  friend class Conv2DLayer;
  friend class FullyConnectedLayer;
  friend class BatchNormalizationLayer;

  /** Declare opitmizer as friend to get variable/gradient reference */
  friend class Optimizer;
  friend class SGD;
  friend class Adam;

public:
  /**
   * @brief Weight default constructor
   */
  Weight() : initializer(WeightInitializer::WEIGHT_UNKNOWN), trainable(false) {}

  /**
   * @brief Construct a new Weight object
   *
   * @param dim Variable and gradient tensor dimension
   * @param init Initializer for the tensor
   * @param train If the variable is trainable
   * @param name Name for this weight
   */
  Weight(
    const TensorDim &dim,
    const WeightInitializer init = WeightInitializer::WEIGHT_XAVIER_UNIFORM,
    bool train = true, std::string name = "");

  /**
   * @brief Allocate and initialize the variable
   *
   * @param dim Dimension for the variable
   */
  void initializeVar(const TensorDim &dim);

  /**
   * @brief Swap for weight
   *
   * @param lhs Swap to
   * @param rhs Swap from
   * @note Only swap gradient if trainable
   */
  friend void swap(Weight &lhs, Weight &rhs) noexcept {
    using std::swap;

    swap(lhs.var, rhs.var);
    swap(lhs.initializer, rhs.initializer);
    swap(lhs.trainable, rhs.trainable);
    swap(lhs.grad, rhs.grad);
    swap(lhs.name, rhs.name);
  }

  /**
   * @brief copy assigment
   *
   * @param rhs copy from
   * @return Weight& Updated weight
   */
  Weight &operator=(const Weight &rhs) = default;

  /**
   * @brief move assignment
   *
   * @param rhs move from
   * @return Weight& Updated weight
   */
  Weight &operator=(Weight &&rhs) = default;

  /**
   * @brief Get the TensorDim
   *
   * @return TensorDim Dimension
   */
  TensorDim getDim() { return var.getDim(); }

  /**
   * @brief Get if the weight is trainable
   *
   * @return true if trainable
   * @return false is not trainable
   */
  bool getTrainable() { return trainable; }

  /**
   * @brief Get the name of the weight
   *
   * @return std::string name
   */
  std::string getName() { return name; }

  /**
   * @brief Get the variable tensor (by name)
   *
   * @return Tensor Variable tensor
   */
  Tensor getVariable() { return var; }

  /**
   * @brief Get the Gradient tensor (by name)
   *
   * @return Tensor Gradient tensor
   */
  Tensor getGradient() { return grad; }

private:
  /**
   * @brief Get the variable tensor (by reference)
   *
   * @return Tensor Variable tensor
   */
  Tensor &getVariableRef() { return var; }

  /**
   * @brief Get the Gradient tensor (by reference)
   *
   * @return Tensor Gradient tensor
   */
  Tensor &getGradientRef() { return grad; }

  Tensor var;                    /**< variable to be updated and used */
  Tensor grad;                   /**< gradient for the variable */
  WeightInitializer initializer; /**< initializer for this variable */
  bool trainable;                /**< if this variable is trainable */
  std::string name;              /**< name of the parameter */
};

} // namespace nntrainer

#endif /** __WEIGHT_H__ */
