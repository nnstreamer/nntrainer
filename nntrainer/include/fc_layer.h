/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	fc_layer.h
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_H__
#define __FC_LAYER_H__
#ifdef __cplusplus

#include <fstream>
#include <iostream>
#include <layer.h>
#include <optimizer.h>
#include <tensor.h>
#include <vector>

namespace nntrainer {

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayer : public Layer {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayer() : unit(0) { setType(LAYER_FC); };

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedLayer(){};

  /**
   *  @brief  Move constructor of Pooling 2D Layer.
   *  @param[in] FullyConnected &&
   */
  FullyConnectedLayer(FullyConnectedLayer &&rhs) = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedLayer to be moved.
   */
  FullyConnectedLayer &operator=(FullyConnectedLayer &&rhs) = default;

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file);

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file);

  /**
   * @copydoc Layer::forwarding(sharedConstTensor in)
   */
  sharedConstTensor forwarding(sharedConstTensor in);

  /**
   * @copydoc Layer::backwarding(sharedConstTensor in, int iteration)
   */
  sharedConstTensor backwarding(sharedConstTensor in, int iteration);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize();

  void setUnit(unsigned int u) { unit = u; };

  /**
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "FullyConnected"; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

private:
  unsigned int unit;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_H__ */
