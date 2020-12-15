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
   * @note Default variable is not trainable as gradient is 0 dim tensor
   */
  Var_Grad() = default;

  /**
   * @brief Var_Grad default destructor
   */
  virtual ~Var_Grad() {}

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
    swap(lhs.grad, rhs.grad);
    swap(lhs.name, rhs.name);
  }

  /**
   * @brief Copy constructor for Var_Grad
   *
   * @param rhs Var_Grad to construct from
   */
  Var_Grad(const Var_Grad &rhs) = default;

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
  Var_Grad &operator=(const Var_Grad &rhs) = default;

  /**
   * @brief move assignment
   *
   * @param rhs move from
   * @return Var_Grad& Updated Var_Grad
   */
  Var_Grad &operator=(Var_Grad &&rhs) = default;

  /**
   * @brief Allocate and initialize the weight variable
   *
   * @param weight_preallocated if initialized, use this tensor for weight
   * @param grad_preallocated if initialized, use this tensor for grad
   */
  virtual void initialize(const Tensor &weight_preallocated = Tensor(),
                          const Tensor &grad_preallocated = Tensor());

  /**
   * @brief Get the TensorDim
   *
   * @return TensorDim Dimension
   */
  TensorDim getDim() const { return dim; }

  /**
   * @brief Get if the Var_Grad is trainable
   *
   * @return true if trainable
   * @return false is not trainable
   */
  bool getTrainable() const { return trainable; }

  /**
   * @brief set if the Var_Grad is trainable
   * @param train true if trainable, else false
   */
  void setTrainable(bool train);

  /**
   * @brief Get the name of the Var_Grad
   *
   * @return std::string name
   */
  std::string getName() const { return name; }

  /**
   * @brief Get the variable tensor (by name)
   *
   * @return Tensor Variable tensor
   */
  Tensor getVariable() const { return *var.get(); }

  /**
   * @brief Get the Gradient tensor (by name)
   *
   * @return Tensor Gradient tensor
   */
  Tensor getGradient() const { return *grad.get(); }

  /**
   * @brief Allocate and initialize the weight variable
   *
   * @param dim Dimension for the weight variable
   */
  void resetGradient() { grad->setZero(); }

  /**
   * @bried Clone the currnet object
   *
   * @return Cloned copy
   */
  Var_Grad clone() const {
    Var_Grad vg(*this);
    vg.var = std::make_shared<Tensor>(this->var->clone());
    vg.grad = std::make_shared<Tensor>(this->grad->clone());

    return vg;
  };

  /**
   * @brief Reset the weight
   *
   * @param dim Variable and gradient tensor dimension
   * @param train If the variable is trainable
   *
   * @note New dimension must maintain the shape of the variable
   */

  void reset(const TensorDim &tdim, bool train) {
    dim = tdim;
    if (!var->uninitialized())
      var->reshape(dim);
    if (!grad->uninitialized())
      grad->reshape(dim);
    trainable = train;
    resetGradient();
  }

protected:
  /**
   * @brief Get the variable tensor (by reference)
   *
   * @return Tensor Variable tensor
   */
  Tensor &getVariableRef() { return *var.get(); }

  /**
   * @brief Get the Gradient tensor (by reference)
   *
   * @return Tensor Gradient tensor
   */
  Tensor &getGradientRef() { return *grad.get(); }

  TensorDim dim;                /**< dimension of the tensor */
  std::shared_ptr<Tensor> var;  /**< variable to be updated and used */
  std::shared_ptr<Tensor> grad; /**< gradient for the variable */
  bool trainable;               /**< if this variable is trainable */
  std::string name;             /**< name of the parameter */
};

} // namespace nntrainer

#endif /** __VAR_GRAD_H__ */
