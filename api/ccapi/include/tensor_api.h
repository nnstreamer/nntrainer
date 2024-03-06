// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api.h
 * @date   11 December 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Tensor interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_TENSOR_H__
#define __ML_TRAIN_TENSOR_H__

#if __cplusplus < MIN_CPP_VERSION
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus

#include <layer.h>
#include <tensor.h>
#include <tuple>
#include <var_grad.h>

using iTensor = nntrainer::Tensor;

namespace ml {
namespace train {

/**
 * @class   Tensor
 * @brief   Tensor extends over Var_Grad for the API
 */
class Tensor : public nntrainer::Var_Grad {
public:
  /**
   * @brief Weight default constructor
   */
  Tensor() : nntrainer::Var_Grad() {}

  /**
   * @brief Construct a new Tensor object
   *
   * @param dim Variable and gradient tensor dimension
   * @param init Initializer for the Tensor
   * @param needg If the tensor needs gradient
   * @param name Name for this tensor
   */
  explicit Tensor(
    const TensorDim &dim,
    const nntrainer::Initializer init = nntrainer::Initializer::ZEROS,
    bool ng = false, std::string name = ""){};

  /**
   * @brief Swap for weight
   *
   * @param lhs Swap to
   * @param rhs Swap from
   * @note Only swap gradient if need gradient
   */
  friend void swap(Tensor &lhs, Tensor &rhs) noexcept {
    using std::swap;
    swap(static_cast<Var_Grad &>(lhs), static_cast<Var_Grad &>(rhs));
  }

  /**
   * @brief Copy constructor for weight
   *
   * @param rhs weight to construct from
   */
  Tensor(const Tensor &rhs) = default;

  /**
   * @brief Move constructor for weight
   *
   * @param rhs weight to construct from
   */
  Tensor(Tensor &&rhs) = default;

  /**
   * @brief copy assigment
   *
   * @param rhs copy from
   * @return Tensor& Updated weight
   */
  Tensor &operator=(const Tensor &rhs) = default;

  /**
   * @brief move assignment
   *
   * @param rhs move from
   * @return Tensor& Updated weight
   */
  Tensor &operator=(Tensor &&rhs) = default;

  /**
   * @brief Clone the currnet object
   *
   * @return Cloned copy
   */
  Tensor clone() const {
    Tensor t(*this);
    if (!this->var->empty())
      t.var = std::make_shared<iTensor>(this->var->clone());
    if (!this->grad->empty())
      t.grad = std::make_shared<iTensor>(this->grad->clone());

    return t;
  }

  /**
   * @brief source layer setter
   *
   */
  void setSrcLayer(std::shared_ptr<Layer> l) { src_layer = l; }

  /**
   * @brief source layer getter
   *
   * @return Layer
   */
  std::shared_ptr<Layer> getSrcLayer() { return src_layer; }

private:
  std::shared_ptr<Layer>
    src_layer; /**< source layer which create this Tensor */
};

} // namespace train
} // namespace ml
#endif // __ML_TRAIN_TENSOR_H__
