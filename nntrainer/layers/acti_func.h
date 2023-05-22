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

namespace nntrainer {

class Tensor;

/**
 * @class   ActiFunc Class
 * @brief   ActiFunc Class
 */
class ActiFunc {

public:
  /**
   * @brief     Constructor of ActiFunc
   */
  ActiFunc(ActivationType at = ActivationType::ACT_NONE, bool in_place_ = true);

  /**
   * @brief     Destructor of ActiFunc
   */
  ~ActiFunc();

  /**
   * @brief setActivation by preset ActivationType
   *
   * @param[in] ActivationType
   */
  void setActiFunc(ActivationType acti_type);

  /**
   * @brief run function
   *
   * @param[in] input : input
   * @param[out] output : output
   */
  void run_fn(Tensor const &input, Tensor &output);

  /**
   * @brief run prime function
   *
   * @param[in] input input
   * @param[in] output output
   * @param[out] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   * @retVal    Tensor
   */
  Tensor &run_prime_fn(Tensor &input, Tensor &output,
                       Tensor &outgoing_derivative,
                       Tensor const &incoming_derivative);

  /**
   * @brief run prime function
   *
   * @param[in] output output
   * @param[out] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   * @retVal    Tensor
   */
  Tensor &run_prime_fn(Tensor &output, Tensor &outgoing_derivative,
                       Tensor const &incoming_derivative);

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const;

  /**
   * @brief       Calculate softmax for Tensor Type
   * @param[in] input input Tensor
   * @param[out] output output Tensor
   * @retval      Tensor
   */
  static Tensor &softmax(Tensor const &input, Tensor &output);

  /**
   * @brief     Calculate derivative of softmax function
   * @param[in] output output tensor
   * @param[out] outgoing_derivative result of calculated derivative of softmax
   * @param[in] incoming_derivative incoming derivative tensor from next layer
   * @retVal    Tensor
   */
  static Tensor &softmaxPrime(Tensor const &output, Tensor &outgoing_derivative,
                              Tensor const &incoming_derivative = Tensor());

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
   * @brief leaky relu function
   * @note slope parameter is needed for leaky relu, but supporting property on
   * this class will need extensive refactoring. For now 0.01 is used for
   * negative slope.
   *
   * @param x input
   * @return float output
   */
  static float leakyRelu(float x);

  /**
   * @brief leaky relu prime function
   * @note slope parameter is needed for leaky relu, but supporting property on
   * this class will need extensive refactoring. For now 0.01 is used for
   * negative slope.
   *
   * @param x input
   * @return float output
   */
  static float leakyReluPrime(float x);

  /**
   * @brief     swish activation function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   */
  static Tensor &swish(Tensor const &t_in, Tensor &t_out);

  /**
   * @brief     derivative swish function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   * @param[in] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   */
  static Tensor &swishPrime(Tensor const &t_in, Tensor const &t_out,
                            Tensor &outgoing_derivative,
                            Tensor const &incoming_derivative = Tensor());

  /**
   * @brief     gelu activation function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   */
  static Tensor &gelu(Tensor const &t_in, Tensor &t_out);

  /**
   * @brief     derivative gelu function
   * @param[in] t_in input tensor
   * @param[in] t_out output tensor
   * @param[in] outgoing_derivative outgoing derivative
   * @param[in] incoming_derivative incoming derivative
   */
  static Tensor &geluPrime(Tensor const &t_in, Tensor const &t_out,
                           Tensor &outgoing_derivative,
                           Tensor const &incoming_derivative = Tensor());

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
   * @note  derivative not applied here as this activation_prime_fn applies
   * derivative itself
   * @param[in] activation_fn activation function to be used
   * @param[in] activtion_prime_fn activation prime function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
    std::function<Tensor &(Tensor const &, Tensor const &, Tensor &,
                           Tensor const &)> const &activation_prime_fn);

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
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<float(float const)> activation_fn activation
   * function to be used
   * @param[in] std::function<float(float const, float const)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<float(float const)> const &activation_fn,
    std::function<float(float const, float const)> const &activation_prime_fn);

  /**
   * @brief   Notify that this layer will execute in-place
   *
   * @param val True if execute in-place, else false
   */
  void executeInPlace(bool val);

private:
  std::function<Tensor &(Tensor const &, Tensor &)> _act_fn;
  std::function<Tensor &(Tensor const &, Tensor &, Tensor &, Tensor const &)>
    _act_prime_fn; /**< prime function with input and output*/

  ActivationType
    activation_type; /**< type of the activation represented by this */
  bool in_place;     /**< if this class should operate in_place */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ACTI_FUNC_H__ */
