// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   weight.h
 * @date   22 September 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Weight Class for Neural Network
 *
 */

#ifndef __WEIGHT_H__
#define __WEIGHT_H__

#include <tuple>

#include <tensor.h>
#include <tensor_wrap_specs.h>
#include <var_grad.h>

namespace nntrainer {

/**
 * @class   Weight
 * @brief   Weight extends over Var_Grad with regularization & optimizer updates
 */
class Weight : public Var_Grad {
public:
  /**
   * @brief Specification of the Weight
   *
   * @details The tuple values are dimension, initializer, regularizer,
   * regularizer_constant, need_gradient property amd name of the Weight object.
   */
  typedef WeightSpec Spec;

  /**
   * @brief Weight default constructor
   */
  Weight() :
    Var_Grad(),
    regularizer(WeightRegularizer::UNKNOWN),
    regularizer_constant(1.0f),
    decay(0.0f),
    clip_by_global_norm(0.0f),
    output_axis(3),
    loss_scale(1.0) {}

  /**
   * @brief Construct a new Weight object
   *
   * @param dim Variable and gradient tensor dimension
   * @param init Initializer for the weight
   * @param reg Regularizer for the weight
   * @param reg_const Constant multiplier for regularizer
   * @param ng If the variable needs gradient
   * @param alloc_now The memory for the weight tensors be allocated upon init
   * @param name Name for this weight
   */
  explicit Weight(
    const TensorDim &dim,
    const Tensor::Initializer init = Tensor::Initializer::XAVIER_UNIFORM,
    const WeightRegularizer reg = WeightRegularizer::NONE,
    const float reg_const = 1.0f, const float decay = 0.0f,
    const float clip_by_global_norm = 0.0f, bool ng = true,
    bool alloc_now = false, std::string name = "", unsigned int axis = 3,
    float loss_scale_ = 1.0);

  /**
   * @brief Construct a new Weight object
   *
   * @param dim_v Variable and gradient tensor dimension
   * @param dim_g Gradient tensor dimension
   * @param init Initializer for the weight
   * @param reg Regularizer for the weight
   * @param reg_const Constant multiplier for regularizer
   * @param ng If the variable needs gradient
   * @param alloc_now The memory for the weight tensors be allocated upon init
   * @param name Name for this weight
   */
  explicit Weight(
    const TensorDim &dim_v, const TensorDim &dim_g,
    const Tensor::Initializer init = Tensor::Initializer::XAVIER_UNIFORM,
    const WeightRegularizer reg = WeightRegularizer::NONE,
    const float reg_const = 1.0f, const float decay = 0.0f,
    const float clip_by_global_norm = 0.0f, bool ng = true,
    bool alloc_now = false, std::string name = "", unsigned int axis = 3,
    float loss_scale_ = 1.0);

  /**
   * @brief Construct a new Weight object
   *
   * @param spec Weight specification
   */
  explicit Weight(const Spec &spec, bool alloc_now = false) :
    Weight(std::get<0>(spec), // TensorDim for Variable
           std::get<1>(spec), // TensorDim for Gradient
           std::get<2>(spec), // Initializer
           std::get<3>(spec), // WeightRegularizer
           std::get<4>(spec), // WeightRegularizerConstant
           std::get<5>(spec), // weight decay constant
           std::get<6>(spec), // MaxNorm for clipping
           std::get<7>(spec), // need_gradient
           alloc_now,
           std::get<8>(spec), // Name
           std::get<9>(spec), // out axis
           std::get<10>(spec) // loss scale
    ) {}

  /**
   * @brief Construct a new Weight object
   *
   * @param v Already created variable object
   * @param g Already created gradient object
   * @param v32 Already created gradient object
   * @param n Name for this Weight
   *
   * @note This is primarily used to created wrapper of variable extracted from
   * context. If needed, add support for regularizer, and opt_vars.
   *
   * @note This API is not recommended for usage and must be used for internal
   * uses only, as Weight does not own the tensors v and g, and can go invalid
   * if the owner of these tensors free the tensors.
   */
  explicit Weight(const Tensor &v, const Tensor &g, const Tensor &v32,
                  const std::string &n = "", bool is_dependent = false,
                  unsigned int output_axis_ = 3);

  /**
   * @brief Construct a new Weight object
   *
   * @param v ptr to already created variable tensor
   * @param g ptr to already created gradient tensor
   * @param v32 ptr to already created variable32 tensor
   * @param reg Regularizer for the weight
   * @param reg_const Constant multiplier for regularizer
   */
  explicit Weight(Tensor *v, Tensor *g, Tensor *v32,
                  const WeightRegularizer reg, const float reg_const,
                  const float decay, bool is_dependent = false,
                  const float max_norm = 0.0f, unsigned int output_axis_ = 3,
                  float loss_scale_ = 1.0f);

  /**
   * @brief Swap for weight
   *
   * @param lhs Swap to
   * @param rhs Swap from
   * @note Only swap gradient if need gradient
   */
  friend void swap(Weight &lhs, Weight &rhs) noexcept {
    using std::swap;
    swap(static_cast<Var_Grad &>(lhs), static_cast<Var_Grad &>(rhs));
    swap(lhs.regularizer, rhs.regularizer);
    swap(lhs.regularizer_constant, rhs.regularizer_constant);
    swap(lhs.decay, rhs.decay);
    swap(lhs.clip_by_global_norm, rhs.clip_by_global_norm);
    swap(lhs.output_axis, rhs.output_axis);
    swap(lhs.opt_vars, rhs.opt_vars);
    swap(lhs.loss_scale, rhs.loss_scale);
    swap(lhs.var32, rhs.var32);
  }

  /**
   * @brief Copy constructor for weight
   *
   * @param rhs weight to construct from
   */
  Weight(const Weight &rhs) = default;

  /**
   * @brief Move constructor for weight
   *
   * @param rhs weight to construct from
   */
  Weight(Weight &&rhs) = default;

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
   * @brief Clone the currnet object
   *
   * @return Cloned copy
   */
  Weight clone() const {
    Weight w(*this);
    if (!this->var->empty())
      w.var = std::make_shared<Tensor>(this->var->clone());
    if (!this->grad->empty())
      w.grad = std::make_shared<Tensor>(this->grad->clone());
    if (!this->var32->empty())
      w.var32 = std::make_shared<Tensor>(this->var32->clone());

    return w;
  }

  /**
   * @brief Clear optimizer variables
   */
  void clearOptimizerVariables() { opt_vars.clear(); }

  /**
   * @brief Add optimizer variables
   * @param dim Optimizer variable dimension
   */
  void setOptimizerVariables(std::vector<Tensor *> tensors) {
    opt_vars = tensors;
  }

  /**
   * @brief Get optimizer variable reference
   * @param idx Index of the optimizer variable to get
   * @retval Reference of the optimizer variable
   */
  Tensor &getOptimizerVariableRef(unsigned int idx) { return *opt_vars[idx]; }

  /**
   * @brief Get number of optimizer variable
   * @retval number of optimizer variable
   */
  int getNumOptVariable() { return opt_vars.size(); }

  /**
   * @brief Get axis of Weight
   * @retval axis of Wegiht
   */
  unsigned int getOutputAxis() { return output_axis; }

  /**
   * @brief     check if weight regularizer type is l2norm
   * @return    bool is weight regrulatizer type is L2 Norm
   */
  bool isWeightRegularizerL2Norm() {
    return regularizer == WeightRegularizer::L2NORM;
  }

  /**
   * @brief     check if weight decay is enabled
   * @return    true if weight decay is enabled else false
   */
  bool isWeightDecay() { return decay > epsilon_decay; }

  /**
   * @brief     Get loss from the regularization of the weight
   */
  float getRegularizationLoss() {
    if (hasGradient() && isWeightRegularizerL2Norm())
      return regularizer_constant * 0.5f * var->l2norm();

    return 0;
  }

  /**
   * @brief     Calculate gradient from the regularization of the weight
   */
  void calcRegularizationGradient() {
    if (isWeightRegularizerL2Norm())
      grad->add_i(*var.get(), regularizer_constant);
  }

  /**
   * @brief     Calculate gradient from the decay of the weight
   */
  void calcWeightDecayGradient() {
    if (isWeightDecay())
      applyWeightDecay();
  }

  /**
   * @brief     Apply the gradient to the weight
   */
  void applyGradient(double lr) { var->add_i(*grad.get(), -lr); }

  /**
   * @brief     Apply the gradient to the weight with updated gradient
   * @param[in] updated_grad gradient tensor which is updated in optimizer
   * it might be different data type with gradient in weight. .eg : FP32
   */
  void applyGradient(double lr, Tensor &updated_grad);

  /**
   * @brief Check if the gradient is supposed to be clipped by global norm with
   * the given max_norm value
   *
   * @param max_norm
   * @return true if it is to be clipped
   * @return false otherwise
   */
  static bool isGradientClipByGlobalNorm(const float max_norm) {
    return max_norm > epsilon;
  }

  /**
   * @brief Check if the gradient is supposed to be clipped by global norm
   *
   * @return true if it is to be clipped
   * @return false otherwise
   */
  bool isGradientClipByGlobalNorm() const {
    return clip_by_global_norm > epsilon;
  }

  /**
   * @brief Check if the variable type is not full precision
   *
   * @return true if it is not full precsion
   * @return false otherwise
   */
  bool isMixedPrecision() const {
    return var->getDataType() != ml::train::TensorDim::DataType::FP32;
  }

  /**
   * @brief clip the gradient value based on the given global norm
   *
   * @param global_norm the global norm for all the weights
   */
  void clipGradientByGlobalNorm(const float global_norm) {
    if ((global_norm + epsilon) > clip_by_global_norm)
      grad->multiply_i(clip_by_global_norm / (global_norm + epsilon));
  }

  /**
   * @brief Get the variable FP32 tensor (by reference)
   *
   * @return Tensor Variable FP32 tensor
   */
  Tensor &getVariableFP32Ref() { return *var32.get(); }

  /**
   * @brief Quantize var32 to var
   *
   */
  void quantizeWeight();

private:
  static constexpr float epsilon = 1e-6; /**< epsilon for zero comparison */
  static constexpr float epsilon_decay =
    1e-8; /**< epsilon for zero comparison */

  WeightRegularizer regularizer; /**< regularizer for this variable */
  float regularizer_constant;    /**< constant factor for regularization */
  float decay;                   /**< constant factor for the weight decay */
  float clip_by_global_norm; /**< constant factor to clip gradient by L2 norm */
  unsigned int output_axis;
  float loss_scale;
  std::vector<Tensor *>
    opt_vars; /**< optimizer variables : We assume it is always full-precsion*/
  std::shared_ptr<Tensor> var32;

  /**
   * @brief     Apply the weight decay to the weight
   */
  void applyWeightDecay() { grad->add_i(*var.get(), decay); }
};

} // namespace nntrainer

#endif /** __WEIGHT_H__ */
