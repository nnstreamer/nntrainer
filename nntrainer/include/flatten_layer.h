// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
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

#include <layer_internal.h>
#include <tensor.h>

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
  template <typename... Args>
  FlattenLayer(Args... args) : Layer(LayerType::LAYER_FLATTEN, args...) {}

  /**
   * @brief     Destructor of Flatten Layer
   */
  ~FlattenLayer(){};

  /**
   *  @brief  Move constructor of FlattenLayer.
   *  @param[in] FlattenLayer &&
   */
  FlattenLayer(FlattenLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FlattenLayer to be moved.
   */
  FlattenLayer &operator=(FlattenLayer &&rhs) = default;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize();

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
   * @copydoc Layer::forwarding(sharedConstTensors in)
   */
  sharedConstTensors forwarding(sharedConstTensors in);

  /**
   * @copydoc Layer::backwarding(sharedConstTensors in, int iteration)
   */
  sharedConstTensors backwarding(sharedConstTensors in, int iteration);

  /**
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "Flatten"; };
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FLATTEN_LAYER_H__ */
