/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	flatten_layer.h
 * @date	16 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Flatten Layer Class for Neural Network
 *
 */

#ifndef __FLATTEN_LAYER_H__
#define __FLATTEN_LAYER_H__
#ifdef __cplusplus

#include <fstream>
#include <iostream>
#include <layer.h>
#include <tensor.h>
#include <vector>

namespace nntrainer {

/**
 * @class   Flatten Layer
 * @brief   Flatten Layer
 */
class FlattenLayer : public Layer {
public:
  /**
   * @brief     Constructor of Flatten Layer
   */
  FlattenLayer() { setType(LAYER_FLATTEN); };

  /**
   * @brief     Destructor of Flatten Layer
   */
  ~FlattenLayer(){};

  /**
   * @brief     initialize layer
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(bool last);

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file){};

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file){};

  /**
   * @brief     forward propagation with input
   * @param[in] in Input Tensor from upper layer
   * @param[out] status Error Status of this function
   * @retval     return Flatten Result
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     foward propagation
   * @param[in] in input Tensor from lower layer.
   * @param[in] output dummy variable
   * @param[out] status status
   * @retval    return Flatten Result
   */
  Tensor forwarding(Tensor in, Tensor output, int &status);

  /**
   * @brief     back propagation
   *            Calculate Derivatives
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Number of Epoch
   * @retval    Splited derivatives
   */
  Tensor backwarding(Tensor in, int iteration);

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NOT_SUPPORTED Successful.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FLATTEN_LAYER_H__ */
