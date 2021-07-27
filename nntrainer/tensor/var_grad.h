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

#include <tuple>

#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace nntrainer {

/**
 * @class   Var_Grad
 * @brief   Variable with Gradient, and its corresponding need_gradient property
 */
class Var_Grad {
public:
  /**
   * @brief Specification of the Var_Grad
   *
   * @details The tuple values are dimension, need_gradient property, and the
   * name of the Var_Grad object.
   */
  typedef VarGradSpec Spec;

  /**
   * @brief Var_Grad default constructor
   * @note Default variable is not need_gradient as gradient is 0 dim tensor
   */
  Var_Grad() : Var_Grad(TensorDim()) {}

  /**
   * @brief Var_Grad default destructor
   */
  virtual ~Var_Grad() = default;

  /**
   * @brief Construct a new Var_Grad object
   *
   * @param dim Variable and gradient tensor dimension
   * @param ng If the variable is need_gradient
   * @param alloc_now The memory for the var_grad tensors be allocated upon init
   * @param name Name for this Var_Grad
   */
  explicit Var_Grad(const TensorDim &dim,
                    const Tensor::Initializer init = Tensor::Initializer::NONE,
                    bool ng = true, bool alloc_now = false,
                    const std::string &name = "");

  /**
   * @brief Construct a new Var_Grad object
   *
   * @param spec Var_Grad specification
   */
  explicit Var_Grad(const Spec &spec) :
    Var_Grad(std::get<0>(spec), // TensorDim
             std::get<1>(spec), // initializer
             std::get<2>(spec), // need_gradient
             false,
             std::get<3>(spec) // Name
    ) {}

  /**
   * @brief Construct a new Var_Grad object
   *
   * @param v Already created variable object
   * @param g Already created gradient object
   * @param n Name for this Var_Grad
   *
   * @note This API is not recommended for usage and must be used for internal
   * uses only, as Var_Grad does not own the tensors v and g, and can go invalid
   * if the owner of these tensors free the tensors.
   */
  explicit Var_Grad(const Tensor &v, const Tensor &g,
                    const std::string &n = "") :
    dim(v.getDim()),
    var(std::make_shared<Tensor>(v.getSharedDataTensor(dim, 0, false))),
    grad(std::make_shared<Tensor>()),
    need_gradient(!g.empty()),
    alloc_now(v.isAllocated()),
    name(n) {
    if (need_gradient)
      grad = std::make_shared<Tensor>(g.getSharedDataTensor(dim, 0, false));
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
   * @param var_preallocated if initialized, use this tensor for var
   * @param grad_preallocated if initialized, use this tensor for grad
   * @param gtrain If the network is training or not
   */
  virtual void initialize(const Tensor &var_preallocated = Tensor(),
                          const Tensor &grad_preallocated = Tensor(),
                          bool gtrain = true) {
    initializeVariable(var_preallocated);
    if (gtrain)
      initializeGradient(grad_preallocated);
    else
      grad = std::make_shared<Tensor>();
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
   * @brief Get if the Var_Grad is need_gradient
   *
   * @retval true if need_gradient
   * @retval false is not need_gradient
   */
  bool needsGradient() const { return need_gradient; }

  /**
   * @brief set if the Var_Grad should need gradient
   * @param ng true if needs gradient, else false
   */
  void needsGradient(bool ng);

  /**
   * @brief Get the name of the Var_Grad
   *
   * @return std::string name
   */
  const std::string &getName() const { return name; }

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
    /** zero the gradient */
    grad->initialize();
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
   * @brief Reset the variable and gradient
   *
   * @param dim Variable and gradient tensor dimension
   * @param ng If the variable needs gradient
   *
   * @note New dimension must maintain the shape of the variable
   */
  void reset(const TensorDim &tdim, Tensor::Initializer init, bool ng) {
    dim = tdim;
    if (!var->empty())
      var->reshape(dim);
    var->initialize(init);

    if (!grad->empty())
      grad->reshape(dim);
    need_gradient = ng;
    resetGradient();
  }

  /**
   * @brief Set batch size
   *
   * @param batch batch size
   */
  void setBatchSize(unsigned int batch) {
    dim.batch(batch);

    if (!var->empty())
      var->updateBatch(batch);
    if (!grad->empty())
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
  void allocateGradient() { grad->allocate(); }

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

  /**
   * @brief If this variable has gradient
   *
   * @return true if the var_grad as gradient set, else false
   * @note this is can return is the var_grad needs gradient but it not
   * empty
   */
  bool hasGradient() const { return need_gradient && !grad->empty(); }

protected:
  TensorDim dim;                /**< dimension of the tensor */
  std::shared_ptr<Tensor> var;  /**< variable to be updated and used */
  std::shared_ptr<Tensor> grad; /**< gradient for the variable */
  bool need_gradient;           /**< if this variable needs gradient */
  bool alloc_now;   /**< if the tensor should be allocated instantly */
  std::string name; /**< name of the parameter */
};

} // namespace nntrainer

#endif /** __VAR_GRAD_H__ */
