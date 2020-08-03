/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
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

#include <fstream>
#include <iostream>
#include <layer.h>
#include <optimizer.h>
#include <tensor.h>
#include <vector>

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
  ActivationLayer();

  /**
   * @brief     Destructor of Activation Layer
   */
  ~ActivationLayer(){};

  /**
   * @brief     Initialize the layer
   *
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(bool last);

  /**
   * @brief     Read Activation layer params. This is essentially noops for now.
   * @param[in] file input stream file
   */
  void read(std::ifstream &file){/* noop */};

  /**
   * @brief     Save Activation layer params. This is essentially noops for now.
   * @param[in] file output stream file
   */
  void save(std::ofstream &file){/* noop */};

  /**
   * @copydoc Layer::forwarding(sharedTensor in)
   */
  sharedTensor forwarding(sharedTensor in);

  /**
   * @copydoc Layer::backwarding(sharedTensor in, int iteration)
   */
  sharedTensor backwarding(sharedTensor in, int iteration);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief setActivation by preset actiType
   *
   * @param[in] ActiType actiType actiType to be set
   */
  void setActivation(ActiType acti_type);

  /**
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "Activation"; };

  /**
   * @brief       Calculate softmax for Tensor Type
   * @param[in] t Tensor
   * @retval      Tensor
   */
  static Tensor softmax(Tensor const &x);

  /**
   * @brief     derivative softmax function for Tensor Type
   * @param[in] x Tensor
   * @retVal    Tensor
   */
  static Tensor softmaxPrime(Tensor const &x,
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

private:
  std::function<Tensor(Tensor const &)> _act_fn;
  std::function<Tensor(Tensor const &, Tensor const &)> _act_prime_fn;

  /**
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<Tensor(Tensor const &)> activation_fn activation
   *            function to be used
   * @param[in] std::function<Tensor(Tensor const &)> activation_prime_fn
   *            activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<Tensor(Tensor const &)> const &activation_fn,
    std::function<Tensor(Tensor const &)> const &activation_prime_fn);

  /**
   * @brief setActivation by custom activation function
   * @note  derivative not applied here as this activation_prime_fn applies
   * derivative itself
   * @param[in] std::function<Tensor(Tensor const &)> activation_fn activation
   *            function to be used
   * @param[in] std::function<Tensor(Tensor const &, Tensor const &)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(std::function<Tensor(Tensor const &)> const &activation_fn,
                    std::function<Tensor(Tensor const &, Tensor const &)> const
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
