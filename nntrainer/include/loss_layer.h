/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	loss_layer.h
 * @date	12 June 2020
 * @brief	This is Loss Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __LOSS_LAYER_H__
#define __LOSS_LAYER_H__
#ifdef __cplusplus

#include <fstream>
#include <iostream>
#include <layer.h>
#include <optimizer.h>
#include <tensor.h>
#include <vector>

namespace nntrainer {

/**
 * @class   LossLayer
 * @brief   loss layer
 */
class LossLayer : public Layer {
public:
  /**
   * @brief     Constructor of Loss Layer
   */
  LossLayer() { setType(LAYER_LOSS); };

  /**
   * @brief     Destructor of Loss Layer
   */
  ~LossLayer(){};

  /**
   * @brief     Forward Propation of neural Network
   * @param[in] in Input Tensor taken by upper layer
   * @retval    Output Tensor
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] output input Tensor from lower layer.
   * @param[in] label label Tensor.
   * @retval    loss (cost)
   */
  Tensor forwarding(Tensor output, Tensor label, int &status);

  /**
   * @brief     back propagation
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    loss diff Tensor
   */
  Tensor backwarding(Tensor in, int iteration);

  /**
   * @brief     read layer Weight & Bias data from file
   * @param[in] file input file stream
   */
  void read(std::ifstream &file) {}

  /**
   * @brief     save layer Weight & Bias data from file
   * @param[in] file output file stream
   */
  void save(std::ofstream &file) {}

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     initialize layer
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(bool last);

private:
  /**
   * @brief     update loss
   * @param[in] l Tensor data to calculate
   */
  void updateLoss(const Tensor &l);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOSS_LAYER_H__ */
