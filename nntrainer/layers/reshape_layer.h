// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   reshape_layer.h
 * @date   16 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Reshape Layer Class for Neural Network
 *
 */

#ifndef __RESHAPE_LAYER_H__
#define __RESHAPE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Reshape Layer
 * @brief   Reshape Layer
 */
class ReshapeLayer : public Layer {
public:
  /**
   * @brief     Constructor of Reshape Layer
   */
  NNTR_EXPORT ReshapeLayer() : Layer() {}

  /**
   * @brief     Destructor of Reshape Layer
   */
  NNTR_EXPORT ~ReshapeLayer() = default;

  /**
   *  @brief  Move constructor of ReshapeLayer.
   *  @param[in] ReshapeLayer &&
   */
  NNTR_EXPORT ReshapeLayer(ReshapeLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ReshapeLayer to be moved.
   */
  NNTR_EXPORT ReshapeLayer &operator=(ReshapeLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_EXPORT void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_EXPORT void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_EXPORT void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  NNTR_EXPORT InPlaceType initializeInPlace() override {
    is_inplace = true;
    return InPlaceType::RESTRICTING;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_EXPORT void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_EXPORT const std::string getType() const override {
    return ReshapeLayer::type;
  };

  static constexpr const char *type = "reshape";

protected:
  std::tuple<props::TargetShape>
    reshape_props; /**< reshape properties : target_shape after reshape */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RESHAPE_LAYER_H__ */
