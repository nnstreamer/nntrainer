// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   var_grad.h
 * @date   13 November 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Var_Grad Class for Neural Network
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
  virtual ~Var_Grad() = default;

  /**
   * @brief Construct a new Var_Grad object
   *
   * @param dim Variable and gradient tensor dimension
   * @param train If the variable is trainable
   * @param alloc_now The memory for the var_grad tensors be allocated upon init
   * @param name Name for this Var_Grad
   */
  explicit Var_Grad(const TensorDim &dim, bool train = true,
                    bool alloc_now = false, const std::string &name = "");

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
   * @param var_preallocated if initialized, use this tensor for var
   * @param grad_preallocated if initialized, use this tensor for grad
   * @param gtrain If all the variables should be trainable
   */
  virtual void initialize(const Tensor &var_preallocated = Tensor(),
                          const Tensor &grad_preallocated = Tensor(),
                          bool gtrain = true) {
    initializeVariable(var_preallocated);
    if (gtrain)
      initializeGradient(grad_preallocated);
  }

  /**
   * @brief Initialize the variable
   * @param preallocated if initialized, use this tensor for variable memory
   */
  virtual void initializeVariable(const Tensor &preallocated = Tensor());

  /**
   * @brief Initialize the gradient for the variable
   * @param preallocated if initialized, use this tensor for gradient memory
   */
  virtual void initializeGradient(const Tensor &preallocated = Tensor());

  /**
   * @brief Allocate and initialize the variable and grad
   * @note Variable and grad share the memory in this case
   */
  virtual void initializeShared();

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
   * @brief Reset the gradient to 0
   */
  void resetGradient() {
    if (grad->isAllocated())
      grad->setZero();
  }

  /**
   * @brief Clone the currnet object
   *
   * @return Cloned copy
   */
  Var_Grad clone() const {
    Var_Grad vg(*this);

    /// @fixme even if var is not allocated, cloned var will have allocated
    /// memory
    vg.var = std::make_shared<Tensor>(this->var->clone());
    vg.grad = std::make_shared<Tensor>(this->grad->clone());

    return vg;
  };

  /**
   * @brief transpose and clone variable, gradient is set to null here
   * @warning returned var_grad has broken invariant, so the gradient should
   * never be used
   *
   * @param direction direction to transpose
   * @return Var_Grad new var_grad
   */
  Var_Grad cloneTransposeVariableOnly(const std::string &direction) const;

  /**
   * @brief Reset the variable and gradient
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

  /**
   * @brief Set batch size
   *
   * @param batch batch size
   */
  void setBatchSize(unsigned int batch) {
    dim.batch(batch);

    if (!var->uninitialized())
      var->updateBatch(batch);
    if (!grad->uninitialized())
      grad->updateBatch(batch);
  }

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

  /**
   * @brief Get the variable tensor (by reference)
   *
   * @return Tensor Variable tensor
   */
  const Tensor &getVariableRef() const { return *var.get(); }

  /**
   * @brief Get the Gradient tensor (by reference)
   *
   * @return Tensor Gradient tensor
   */
  const Tensor &getGradientRef() const { return *grad.get(); }

  /**
   * @brief Allocate memory for the variable
   */
  void allocateVariable() { var->allocate(); }

  /**
   * @brief Allocate memory for the gradient
   */
  void allocateGradient() {
    grad->allocate();
    resetGradient();
  }

  /**
   * @brief Allocate memory for the variable and gradient
   */
  void allocate() {
    allocateVariable();
    allocateGradient();
  }

  /**
   * @brief Deallocate memory for the variable and gradient
   */
  void deallocate() {
    deallocateVariable();
    deallocateGradient();
  }

  /**
   * @brief Deallocate memory for the variable
   */
  void deallocateVariable() { var->deallocate(); }

  /**
   * @brief Deallocate memory for the gradient
   */
  void deallocateGradient() { grad->deallocate(); }

  /**
   * @brief Update the variable to use the variable from the given param
   * @param vg Var_Grad whose variable must be updated with
   */
  void updateVariableByVariable(const Var_Grad &vg) { var = vg.var; }

  /**
   * @brief Update the gradient to use the variable from the given param
   * @param vg Var_Grad whose variable must be updated with
   */
  void updateGradientByVariable(const Var_Grad &vg) { grad = vg.var; }

protected:
  TensorDim dim;                /**< dimension of the tensor */
  std::shared_ptr<Tensor> var;  /**< variable to be updated and used */
  std::shared_ptr<Tensor> grad; /**< gradient for the variable */
  bool trainable;               /**< if this variable is trainable */
  bool alloc_now;   /**< if the tensor should be allocated instantly */
  std::string name; /**< name of the parameter */
};

} // namespace nntrainer

#endif /** __VAR_GRAD_H__ */
