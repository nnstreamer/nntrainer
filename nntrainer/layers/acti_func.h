// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   acti_func.cpp
 * @date   22 March 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Function Class for Neural Network
 *
 */

#ifndef __ACTI_FUNC_H__
#define __ACTI_FUNC_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Activation enum to string map
 */
const std::array<std::string, 6> ActivationTypeStr = {
  "tanh", "sigmoid", "relu", "softmax", "none", "unknown"};

/**
 * @class   ActiFunc Class
 * @brief   ActiFunc Class
 */
class ActiFunc {

public:
  /**
   * @brief     Constructor of ActiFunc
   */
  ActiFunc(ActivationType at = ActivationType::ACT_NONE,
           bool in_place_ = true) :
    in_place(in_place_) {
    setActiFunc(at);
  }

  /**
   * @brief     Destructor of ActiFunc
   */
  ~ActiFunc() = default;

  /**
   * @brief setActivation by preset ActivationType
   *
   * @param[in] ActivationType
   */
  void setActiFunc(ActivationType acti_type);

  /**
   * @brief run function
   *
   * @param[in] x : input
   * @param[out] output : output
   */
  void run_fn(Tensor const &x, Tensor &output);

  /**
   * @brief run prime function
   *
   * @param[in] in : input
   * @param[out] ret : output
   * @param[in] deriv : derivative
   * @retVal    Tensor
   */
  Tensor &run_prime_fn(Tensor &in, Tensor &ret, Tensor const &deriv);

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const;

  /**
   * @brief       Calculate softmax for Tensor Type
   * @param[in] x Tensor
   * @param[out] output output Tensor
   * @retval      Tensor
   */
  static Tensor &softmax(Tensor const &x, Tensor &output);

  /**
   * @brief     derivative softmax function for Tensor Type
   * @param[in] x Tensor
   * @param[out] output output Tensor
   * @param[in] derivative derivative Tensor from next layer
   * @retVal    Tensor
   */
  static Tensor &softmaxPrime(Tensor const &x, Tensor &output,
                              Tensor const &derivative = Tensor());

  /**
   * @brief     sigmoid activation function
   * @param[in] x input
   */
  static float sigmoid(float x);

  /**
   * @brief     derivative sigmoid function
   * @param[in] x input
   */
  static float sigmoidPrime(float x);

  /**
   * @brief     tanh function for float type
   * @param[in] x input
   */
  static float tanhFloat(float x);

  /**
   * @brief     derivative tanh function
   * @param[in] x input
   */
  static float tanhPrime(float x);

  /**
   * @brief     relu activation function
   * @param[in] x input
   */
  static float relu(float x);

  /**
   * @brief     derivative relu function
   * @param[in] x input
   */
  static float reluPrime(float x);

  /**
   * @brief     no_op function
   * @param[in] x input
   */
  static float no_op(float x);

  /**
   * @brief     no_op function
   * @param[in] x input
   */
  static float no_op_prime(float x);

  /**
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)> activation_fn
   * activation function to be used
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
    std::function<Tensor &(Tensor &, Tensor &)> const &activation_prime_fn);

  /**
   * @brief setActivation by custom activation function
   * @note  derivative not applied here as this activation_prime_fn applies
   * derivative itself
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)> activation_fn
   * activation function to be used
   * @param[in] std::function<Tensor(Tensor const &, Tensor &, Tensor const &)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
    std::function<Tensor &(Tensor &, Tensor &, Tensor const &)> const
      &activation_prime_fn);

  /**
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<float(float const &)> activation_fn activation
   *            function to be used
   * @param[in] std::function<float(float const &)> activation_prime_fn
   *            activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<float(float const)> const &activation_fn,
    std::function<float(float const)> const &activation_prime_fn);

  /**
   * @brief   Notify that this layer will execute in-place
   *
   * @param val True if execute in-place, else false
   */
  void executeInPlace(bool val) {
    if (val && !supportInPlace())
      throw std::runtime_error(
        "Error setting activation layer to work in-place");

    in_place = val;
  }

private:
  std::function<Tensor &(Tensor const &, Tensor &)> _act_fn;
  std::function<Tensor &(Tensor &, Tensor &, Tensor const &)> _act_prime_fn;

  ActivationType
    activation_type; /**< type of the activaiton represented by this */
  bool in_place;     /**< if this class should operate in_place */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ACTI_FUNC_H__ */
