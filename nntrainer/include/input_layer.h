/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file	input_layer.h
 * @date	14 May 2020
 * @brief	This is Input Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __INPUT_LAYER_H__
#define __INPUT_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Input Layer
 * @brief   Just Handle the Input of Network
 */
class InputLayer : public Layer {
public:
  /**
   * @brief     Constructor of InputLayer
   */
  template <typename... Args>
  InputLayer(bool normalization = false, bool standardization = false,
             Args... args) :
    Layer(LayerType::LAYER_IN, args...),
    normalization(false),
    standardization(false) {}

  /**
   * @brief     Destructor of InputLayer
   */
  ~InputLayer() {}

  /**
   *  @brief  Move constructor of Pooling 2D Layer.
   *  @param[in] Input &&
   */
  InputLayer(InputLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs InputLayer to be moved.
   */
  InputLayer &operator=(InputLayer &&rhs) = default;

  /**
   * @brief     No Weight data for this Input Layer
   */
  void read(std::ifstream &file){};

  /**
   * @brief     No Weight data for this Input Layer
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
   * @brief     Initializer of Input Layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize();

  /**
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "Input"; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

private:
  bool normalization;
  bool standardization;

  /**
   * @brief     set normalization
   * @param[in] enable boolean
   */
  void setNormalization(bool enable) { this->normalization = enable; };

  /**
   * @brief     set standardization
   * @param[in] enable boolean
   */
  void setStandardization(bool enable) { this->standardization = enable; };
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __INPUT_LAYER_H__ */
