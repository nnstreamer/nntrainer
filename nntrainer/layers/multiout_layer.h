// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        multiout_layer.h
 * @date        05 Nov 2020
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 * @brief       This is Multi Output Layer Class for Neural Network
 */

#ifndef __MULTIOUT_LAYER_H__
#define __MULTIOUT_LAYER_H__
#ifdef __cplusplus

#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Multiout Layer
 * @brief   Multiout Layer
 */
class MultiOutLayer : public Layer {
public:
  /**
   * @brief     Constructor of Multiout Layer
   */
  NNTR_API MultiOutLayer() : Layer() {}

  /**
   * @brief     Destructor of Multiout Layer
   */
  NNTR_API ~MultiOutLayer() = default;

  /**
   *  @brief  Move constructor of MultiOutLayer.
   *  @param[in] MultiOutLayer &&
   */
  NNTR_API MultiOutLayer(MultiOutLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs MultiOutLayer to be moved.
   */
  NNTR_API MultiOutLayer &operator=(MultiOutLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_API void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  NNTR_API void incremental_forwarding(RunLayerContext &context,
                                       unsigned int from, unsigned int to,
                                       bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_API bool supportBackwarding() const override { return true; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  NNTR_API InPlaceType initializeInPlace() final {
    is_inplace = true;
    return InPlaceType::RESTRICTING;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_API void
  exportTo(Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_API const std::string getType() const override {
    return MultiOutLayer::type;
  };

  NNTR_API void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  static constexpr const char *type = "multiout";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MULTIOUT_LAYER_H__ */
