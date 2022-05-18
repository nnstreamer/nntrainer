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
  MultiOutLayer() : Layer() {}

  /**
   * @brief     Destructor of Multiout Layer
   */
  ~MultiOutLayer() = default;

  /**
   *  @brief  Move constructor of MultiOutLayer.
   *  @param[in] MultiOutLayer &&
   */
  MultiOutLayer(MultiOutLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs MultiOutLayer to be moved.
   */
  MultiOutLayer &operator=(MultiOutLayer &&rhs) = default;

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return true; }

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
  const std::string getType() const override { return MultiOutLayer::type; };

  inline static const std::string type = "multiout";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MULTIOUT_LAYER_H__ */
