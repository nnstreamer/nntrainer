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
   * @brief     forward propagation with input
   * @param[in] in Input Tensor from upper layer
   * @param[out] status Error Status of this function
   * @retval    Activation(f(x))
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     back propagation calculate activation prime.
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Numberof Epoch for ADAM
   * @retval    Tensor
   */
  Tensor backwarding(Tensor in, int iteration);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief setActivation by custom activation function
   *
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
   *
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
   * @brief setActivation by preset actiType
   *
   * @param[in] ActiType actiType actiType to be set
   */
  void setActivation(ActiType acti_type);

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

private:
  std::function<Tensor(Tensor const &)> _act_fn;
  std::function<Tensor(Tensor const &)> _act_prime_fn;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ACTIVATION_LAYER_H__ */
