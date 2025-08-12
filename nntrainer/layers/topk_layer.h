// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   topk_layer.h
 * @date   28 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Topk Layer Class for Neural Network
 *
 */

#ifndef __TOPK_LAYER_H__
#define __TOPK_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Topk Layer
 * @brief   Topk Layer
 */
class TopkLayer : public Layer {
public:
  /**
   * @brief     Constructor of topk Layer
   */
  TopkLayer() : Layer() {}

  /**
   * @brief     Destructor of topk Layer
   */
  ~TopkLayer() = default;

  /**
   *  @brief  Move constructor of TopkLayer.
   *  @param[in] Topk &&
   */
  TopkLayer(TopkLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs TopkLayer to be moved.
   */
  TopkLayer &operator=(TopkLayer &&rhs) = default;

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
  const std::string getType() const override { return TopkLayer::type; };

  static constexpr const char *type = "topk";

protected:
  std::tuple<props::Print, props::K>
    topk_props; /**< topk properties : k for topk */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TOPK_LAYER_H__ */
