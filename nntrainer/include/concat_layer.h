// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	concat_layer.h
 * @date	27 Oct 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Concat Layer Class for Neural Network
 *
 */

#ifndef __CONCAT_LAYER_H__
#define __CONCAT_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Concat Layer
 * @brief   Concat Layer
 */
class ConcatLayer : public Layer {
public:
  /**
   * @brief     Constructor of Concat Layer
   */
  template <typename... Args>
  ConcatLayer(unsigned int num_inputs_ = 0, Args... args) :
    Layer(LayerType::LAYER_CONCAT, args...) {
    num_inputs = num_inputs_;
  }

  /**
   * @brief     Destructor of Concat Layer
   */
  ~ConcatLayer(){};

  /**
   *  @brief  Move constructor of ConcatLayer.
   *  @param[in] ConcatLayer &&
   */
  ConcatLayer(ConcatLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ConcatLayer to be moved.
   */
  ConcatLayer &operator=(ConcatLayer &&rhs) = default;

  /**
   * @brief     initialize layer
   * @param[in] last last layer
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
  std::string getBaseName() { return "Concat"; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONCAT_LAYER_H__ */
