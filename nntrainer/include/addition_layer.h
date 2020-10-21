// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	addition_layer.h
 * @date	30 July 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Addition Layer Class for Neural Network
 *
 */

#ifndef __ADDITION_LAYER_H__
#define __ADDITION_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Addition Layer
 * @brief   Addition Layer
 */
class AdditionLayer : public Layer {
public:
  /**
   * @brief     Constructor of Addition Layer
   */
  template <typename... Args>
  AdditionLayer(unsigned int num_inputs_ = 0, Args... args) :
    Layer(LayerType::LAYER_ADDITION, args...) {
    num_inputs = num_inputs_;
  }

  /**
   * @brief     Destructor of Addition Layer
   */
  ~AdditionLayer(){};

  /**
   *  @brief  Move constructor of AdditionLayer.
   *  @param[in] AdditionLayer &&
   */
  AdditionLayer(AdditionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs AdditionLayer to be moved.
   */
  AdditionLayer &operator=(AdditionLayer &&rhs) = default;

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
  std::string getBaseName() { return "Addition"; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ADDITION_LAYER_H__ */
