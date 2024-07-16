// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   subtraction_layer.h
 * @date   15 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Subtraction Layer Class for Neural Network
 *
 */

#ifndef __SUBTRACTION_LAYER_H__
#define __SUBTRACTION_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class Subtraction Layer
 * @brief Subtraction Layer
 */
class SubtractionLayer : public Layer {
public:
  /**
   * @brief Constructor of Subtraction Layer
   */
  SubtractionLayer() : Layer(), sub_props(props::Print()) {}

  /**
   * @brief Destructor of Subtraction Layer
   */
  ~SubtractionLayer(){};

  /**
   *  @brief  Move constructor of Subtraction Layer.
   *  @param[in] SubtractionLayer &&
   */
  SubtractionLayer(SubtractionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs SubtractionLayer to be moved.
   */
  SubtractionLayer &operator=(SubtractionLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return SubtractionLayer::type; };

  std::tuple<props::Print> sub_props;

  inline static const std::string type = "subtraction";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SUBTRACTION_LAYER_H__ */
