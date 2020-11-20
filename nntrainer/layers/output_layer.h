// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        output_layer.h
 * @date        05 Nov 2020
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 * @brief       This is Multi Output Layer Class for Neural Network
 *
 */

#ifndef __OUTPUT_LAYER_H__
#define __OUTPUT_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Output Layer
 * @brief   Output Layer
 */
class OutputLayer : public Layer {
public:
  /**
   * @brief     Constructor of Output Layer
   */
  template <typename... Args>
  OutputLayer(unsigned int num_output_ = 1, Args... args) : Layer(args...) {
    num_outputs = num_output_;
  }

  /**
   * @brief     Destructor of Output Layer
   */
  ~OutputLayer(){};

  /**
   *  @brief  Move constructor of OutputLayer.
   *  @param[in] OutputLayer &&
   */
  OutputLayer(OutputLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs OutputLayer to be moved.
   */
  OutputLayer &operator=(OutputLayer &&rhs) = default;

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
  void forwarding(sharedConstTensors in);

  /**
   * @copydoc Layer::backwarding(sharedConstTensors in, int iteration)
   */
  void backwarding(int iteration, sharedConstTensors in);

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const { return OutputLayer::type; };

  static const std::string type;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __OUTPUT_LAYER_H__ */
