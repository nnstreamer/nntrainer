// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   unsqueeze_layer.h
 * @date   08 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @author Abhinav Dwivedi <abhinav.d@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Unsqueeze Layer Class for Neural Network
 *
 */

#ifndef __UNSQUEEZE_LAYER_H__
#define __UNSQUEEZE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Unsqueeze Layer
 * @brief   Unsqueeze Layer
 */
class UnsqueezeLayer : public Layer {
public:
  /**
   * @brief     Constructor of Unsqueeze Layer
   */
  UnsqueezeLayer() : Layer() {}

  /**
   * @brief     Destructor of Reshape Layer
   */
  ~UnsqueezeLayer() = default;

  /**
   *  @brief  Move constructor of UnsqueezeLayer.
   *  @param[in] ReshapeLayer &&
   */
  UnsqueezeLayer(UnsqueezeLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs UnsqueezeLayer to be moved.
   */
  UnsqueezeLayer &operator=(UnsqueezeLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() override {
    is_inplace = true;
    return InPlaceType::RESTRICTING;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return UnsqueezeLayer::type; };

  static constexpr const char *type = "unsqueeze";

protected:
  std::tuple<props::Print, props::Axis>
    unsqueeze_props; /**< unsqueeze properties : axis for unsqueeze */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __UNSQUEEZE_LAYER_H__ */
