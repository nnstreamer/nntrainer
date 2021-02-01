// SPDX-License-Identifier: Apache-2.0
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
#include <var_grad.h>

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
class Weight : public Var_Grad {

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
  Weight() : Var_Grad(), initializer(WeightInitializer::WEIGHT_UNKNOWN) {}

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
    bool train = true, bool alloc_now = true, std::string name = "");

  /**
   * @copydoc var_grad::initializeVariable(const Tensor &)
   */
  void initializeVariable(const Tensor &preallocated = Tensor());

  /**
   * @copydoc var_grad::initializeGradient(const Tensor &, bool)
   */
  void initializeGradient(const Tensor &preallocated = Tensor(),
                          bool gtrain = true);

  /**
   * @brief Swap for weight
   *
   * @param lhs Swap to
   * @param rhs Swap from
   * @note Only swap gradient if trainable
   */
  friend void swap(Weight &lhs, Weight &rhs) noexcept {
    using std::swap;
    swap(static_cast<Var_Grad &>(lhs), static_cast<Var_Grad &>(rhs));
    swap(lhs.initializer, rhs.initializer);
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
    if (!this->var->uninitialized())
      w.var = std::make_shared<Tensor>(this->var->clone());
    if (!this->grad->uninitialized())
      w.grad = std::make_shared<Tensor>(this->grad->clone());

    return w;
  }

  /**
   * @brief Reset the weight
   *
   * @param dim Variable and gradient tensor dimension
   * @param init Initializer for the tensor
   * @param train If the variable is trainable
   *
   * @note New dimension must maintain the shape of the variable
   */
  void reset(const TensorDim &dim, const WeightInitializer init, bool train) {
    initializer = init;
    Var_Grad::reset(dim, train);
  }

  /**
   * @brief Clear optimizer variables
   */
  void clearOptimizerVariables() {
    opt_vars.clear();
    opt_vars_dim.clear();
  }

  /**
   * @brief Add optimizer variables
   * @param dim Optimizer variable dimension
   */
  void addOptimizerVariable(const TensorDim &dim) {
    opt_vars_dim.emplace_back(dim);
    // TODO: Move this out when an optimizer does not initialize with 0.
  }

  /**
   * @brief Get optimizer variable reference
   * @param idx Index of the optimizer variable to get
   * @retval Reference of the optimizer variable
   */
  Tensor &getOptimizerVariableRef(unsigned int idx) { return opt_vars[idx]; }

  /**
   * @brief Allocate and initialize the weight variable, if needed
   */
  void allocateVariable() {
    Var_Grad::allocateVariable();
    initializeVariable();
  }

  /**
   * @brief Allocate and initialize the weight gradient, if needed
   */
  void allocateGradient() {
    Var_Grad::allocateGradient();
    resetGradient();
    allocateOptimizerVariables();
  }

private:
  WeightInitializer initializer; /**< initializer for this variable */

  std::vector<Tensor> opt_vars;        /**< optimizer variables */
  std::vector<TensorDim> opt_vars_dim; /**< optimizer variables dimensions */

  /**
   * @brief Initialize the weight with the initializer
   */
  void runVariableInitializer();

  /**
   * @brief Allocate optimizer related variables for the given weights
   */
  void allocateOptimizerVariables();
};

} // namespace nntrainer

#endif /** __WEIGHT_H__ */
