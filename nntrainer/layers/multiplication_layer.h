// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   multiplication_layer.h
 * @date   15 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Multiplication Layer Class for Neural Network
 *
 */

#ifndef __Multiplication_LAYER_H__
#define __Multiplication_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class Multiplication Layer
 * @brief Multiplication Layer
 */
class MultiplicationLayer : public Layer {
public:
  /**
   * @brief Constructor of Multiplication Layer
   */
  MultiplicationLayer() : Layer(), mul_props(props::Print()) {}

  /**
   * @brief Destructor of Multiplication Layer
   */
  ~MultiplicationLayer(){};

  /**
   *  @brief  Move constructor of Multiplication Layer.
   *  @param[in] MultiplicationLayer &&
   */
  MultiplicationLayer(MultiplicationLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs MultiplicationLayer to be moved.
   */
  MultiplicationLayer &operator=(MultiplicationLayer &&rhs) = default;

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
  const std::string getType() const override {
    return MultiplicationLayer::type;
  };

  std::tuple<props::Print> mul_props;

  inline static const std::string type = "multiplication";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Multiplication_LAYER_H__ */
