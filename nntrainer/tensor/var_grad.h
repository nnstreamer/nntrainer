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
   * @param dim_v Variable tensor dimension
   * @param dim_g Gradient tensor dimension
   * @param ng If the variable is need_gradient
   * @param alloc_now The memory for the var_grad tensors be allocated upon init
   * @param name Name for this Var_Grad
   */
  explicit Var_Grad(const TensorDim &dim_v, const TensorDim &dim_g,
                    const Tensor::Initializer init = Tensor::Initializer::NONE,
                    bool ng = true, bool alloc_now = false,
                    const std::string &name = "");

  /**
   * @brief Construct a new Var_Grad object
   *
   * @param spec Var_Grad specification
   */
  explicit Var_Grad(const Spec &spec, bool alloc_now = false) :
    Var_Grad(std::get<0>(spec), // TensorDim
             std::get<1>(spec), // initializer
             std::get<2>(spec), // need_gradient
             alloc_now,
             std::get<3>(spec) // Name
    ) {}

  /**
   * @brief Construct a new Var_Grad object
   *
   * @param v Already created variable object
   * @param g Already created gradient object
   * @param n Name for this Var_Grad
   * @param is_dependent true if the var grad is dependent
   *
   * @note This API is not recommended for usage and must be used for internal
   * uses only, as Var_Grad does not own the tensors v and g, and can go invalid
   * if the owner of these tensors free the tensors.
   */
  explicit Var_Grad(const Tensor &v, const Tensor &g, const std::string &n = "",
                    bool is_dependent = false) :
    is_dependent(is_dependent),
    is_first_access_gradient(false),
    is_last_access_gradient(false),
    var(
      std::make_shared<Tensor>(v.getSharedDataTensor(v.getDim(), 0, false, n))),
    grad(std::make_shared<Tensor>(n + grad_suffix)) {
    if (!g.empty())
      grad = std::make_shared<Tensor>(
        g.getSharedDataTensor(v.getDim(), 0, false, n + grad_suffix));
  }

  /**
   * @brief Construct a new Var_Grad object
   *
   * @param v ptr to already created variable tensor
   * @param g ptr to already created gradient tensor
   * @param is_dependent true if the given var grad is dependent
   */
  explicit Var_Grad(Tensor *v, Tensor *g, bool is_dependent = false) :
    is_dependent(is_dependent),
    is_first_access_gradient(false),
    is_last_access_gradient(false),
    var(std::shared_ptr<Tensor>(v, [](void *) {})),
    grad(std::shared_ptr<Tensor>(g, [](void *) {})) {
    if (!v)
      var = std::make_shared<Tensor>();
    if (!g)
      grad = std::make_shared<Tensor>();
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
   * @brief Get the TensorDim
   *
   * @return TensorDim Dimension
   */
  TensorDim getDim() const { return var->getDim(); }

  /**
   * @brief Get the name of the variable
   *
   * @return std::string name of the variable
   */
  const std::string &getName() const { return var->getName(); }

  /**
   * @brief Get the name of the gradient
   *
   * @return std::string name of the gradient
   */
  const std::string &getGradientName() const { return grad->getName(); }

  /**
   * @brief Get the variable tensor
   *
   * @return Tensor Variable tensor
   */
  Tensor getVariable() const { return *var.get(); }

  /**
   * @brief Get the Gradient tensor
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
   * @brief Set batch size
   *
   * @param batch batch size
   */
  void setBatchSize(unsigned int batch) {
    if (!var->empty())
      var->updateBatch(batch);
    if (grad && !grad->empty())
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
   * @brief If this variable has gradient
   *
   * @return true if the var_grad as gradient set, else false
   * @note this is can return is the var_grad needs gradient but it not
   * empty
   */
  bool hasGradient() const {
    if (!grad)
      return false;
    if (var->isAllocated())
      return grad->isAllocated();
    return !grad->empty();
  }

  /**
   * @brief check if given weight is dependent to other weight
   *
   * @return bool return true if the weight is dependent to others
   */
  bool isDependent() const { return is_dependent; }

  /**
   * @brief Set the As First Gradient Access
   *
   */
  void setAsGradientFirstAccess() { is_first_access_gradient = true; }

  /**
   * @brief Set the As Gradient Last Access object
   *
   */
  void setAsGradientLastAccess() { is_last_access_gradient = true; }

  /**
   * @brief check if given weight at the last execution order
   * (first access of gradient)
   *
   * @return bool true if last access
   */
  bool isGradientFirstAccess() const { return is_first_access_gradient; }

  /**
   * @brief check if given weight at the first execution order (last access of
   * gradient)
   *
   * @return bool true if last access
   */
  bool isGradientLastAccess() const { return is_last_access_gradient; }

  /**
   * @brief Get the norm of the gradient
   *
   * @return float l2 norm of the gradient
   */
  float getGradientNorm() const { return grad->l2norm(); }

  inline static const std::string grad_suffix = ":grad";

protected:
  bool is_dependent; /**< check if the weight tensor is burrowed from somewhere
                        thus it is dependent */
  bool is_first_access_gradient; /**< check if current weight tensor is first
                                    access */
  bool is_last_access_gradient;  /**< check if current weight tensor is last
   access */

  std::shared_ptr<Tensor> var;  /**< variable to be updated and used */
  std::shared_ptr<Tensor> grad; /**< gradient for the variable */
};

} // namespace nntrainer

#endif /** __VAR_GRAD_H__ */
