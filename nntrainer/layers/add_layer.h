// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   add_layer.h
 * @date   2 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is add layer class
 *
 */

#ifndef __ADD_LAYER_H__
#define __ADD_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class Add Layer
 * @brief Add Layer
 */
class AddLayer : public Layer {
public:
  /**
   * @brief Constructor of Add Layer
   */
  AddLayer() : Layer(), add_props(props::Print()) {}

  /**
   * @brief Destructor of Add Layer
   */
  ~AddLayer(){};

  /**
   *  @brief  Move constructor of Add Layer.
   *  @param[in] AddLayer &&
   */
  AddLayer(AddLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs AddLayer to be moved.
   */
  AddLayer &operator=(AddLayer &&rhs) = default;

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

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
  const std::string getType() const override { return AddLayer::type; };

  std::tuple<props::Print> add_props;

  inline static const std::string type = "add";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ADD_LAYER_H__ */
