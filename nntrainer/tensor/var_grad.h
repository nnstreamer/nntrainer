// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	var_grad.h
 * @date	13 November 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Var_Grad Class for Neural Network
 *
 */

#ifndef __VAR_GRAD_H__
#define __VAR_GRAD_H__

#include <tensor.h>

namespace nntrainer {

/**
 * @class   Var_Grad
 * @brief   Variable with Gradient, and its corresponding trainable property
 */
class Var_Grad {

public:
  /**
   * @brief Var_Grad default constructor
   */
  Var_Grad() : trainable(false) {}

  /**
   * @brief Construct a new Var_Grad object
   *
   * @param dim Variable and gradient tensor dimension
   * @param train If the variable is trainable
   * @param name Name for this Var_Grad
   */
  Var_Grad(const TensorDim &dim, bool train = true,
           const std::string &name = "");

  /**
   * @brief Swap for Var_Grad
   *
   * @param lhs Swap to
   * @param rhs Swap from
   */
  friend void swap(Var_Grad &lhs, Var_Grad &rhs) noexcept {
    using std::swap;

    swap(lhs.var, rhs.var);
    swap(lhs.trainable, rhs.trainable);
    swap(lhs.grad, rhs.grad);
    swap(lhs.name, rhs.name);
  }

  /**
   * @brief Copy constructor for Var_Grad
   *
   * @param rhs Var_Grad to construct from
   */
  Var_Grad(const Var_Grad &rhs);

  /**
   * @brief Move constructor for Var_Grad
   *
   * @param rhs Var_Grad to construct from
   */
  Var_Grad(Var_Grad &&rhs) = default;

  /**
   * @brief copy assigment
   *
   * @param rhs copy from
   * @return Var_Grad& Updated Var_Grad
   */
  Var_Grad &operator=(const Var_Grad &rhs);

  /**
   * @brief move assignment
   *
   * @param rhs move from
   * @return Var_Grad& Updated Var_Grad
   */
  Var_Grad &operator=(Var_Grad &&rhs) = default;

  /**
   * @brief Get the TensorDim
   *
   * @return TensorDim Dimension
   */
  TensorDim getDim() { return var.getDim(); }

  /**
   * @brief Get if the Var_Grad is trainable
   *
   * @return true if trainable
   * @return false is not trainable
   */
  bool getTrainable() { return trainable; }

  /**
   * @brief Get the name of the Var_Grad
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

  /**
   * @brief Allocate and initialize the weight variable
   *
   * @param dim Dimension for the weight variable
   */
  void resetGradient() { grad.setZero(); }

protected:
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

  Tensor var;       /**< variable to be updated and used */
  Tensor grad;      /**< gradient for the variable */
  bool trainable;   /**< if this variable is trainable */
  std::string name; /**< name of the parameter */
};

} // namespace nntrainer

#endif /** __VAR_GRAD_H__ */
