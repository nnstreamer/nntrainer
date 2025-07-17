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

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Input Layer
 * @note    input layers requires to be only single input, consider making the
 * class deal with multiple inputs
 * @brief   Just Handle the Input of Network
 */
class InputLayer : public Layer {
public:
  /**
   * @brief     Constructor of InputLayer
   */
  NNTR_API InputLayer();

  /**
   * @brief     Destructor of InputLayer
   */
  NNTR_API ~InputLayer() = default;

  /**
   *  @brief  Move constructor of Pooling 2D Layer.
   *  @param[in] Input &&
   */
  NNTR_API InputLayer(InputLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs InputLayer to be moved.
   */
  NNTR_API InputLayer &operator=(InputLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_API void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_API bool supportBackwarding() const override { return false; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  NNTR_API InPlaceType initializeInPlace() final {
    is_inplace = true;
    return InPlaceType::NON_RESTRICTING;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_API void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_API const std::string getType() const override {
    return InputLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) override;

  NNTR_API void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  static constexpr const char *type = "input";

private:
  std::tuple<props::Normalization, props::Standardization> input_props;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __INPUT_LAYER_H__ */
