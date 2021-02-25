// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.h
 * @date   17 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Activation Layer
 * @brief   Activation Layer
 */
class ActivationLayer : public Layer {

public:
  /**
   * @brief     Constructor of Activation Layer
   */
  template <typename... Args>
  ActivationLayer(ActivationType at = ActivationType::ACT_NONE, Args... args) :
    Layer(args...) {
    setTrainable(false);
    setActivation(at);
  }

  /**
   * @brief     Destructor of Activation Layer
   */
  ~ActivationLayer(){};

  /**
   * @brief     Initialize the layer
   *
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @brief     Read Activation layer params. This is essentially noops for now.
   * @param[in] file input stream file
   */
  void read(std::ifstream &file) override{/* noop */};

  /**
   * @brief     Save Activation layer params. This is essentially noops for now.
   * @param[in] file output stream file
   */
  void save(std::ofstream &file) override{/* noop */};

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @brief setActivation by preset ActivationType
   *
   * @param[in] ActivationType
   */
  void setActivation(ActivationType acti_type) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ActivationLayer::type; };

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

  static const std::string type;

private:
  std::function<Tensor &(Tensor const &, Tensor &)> _act_fn;
  std::function<Tensor &(Tensor &, Tensor &, Tensor const &)> _act_prime_fn;

  Tensor backup_hidden;

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
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ACTIVATION_LAYER_H__ */
