// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   reorganization.h
 * @date   4 April 2023
 * @brief  This file contains the mean absolute error loss as a sample layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __REORGANIZATION_LAYER_H__
#define __REORGANIZATION_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

namespace custom {

/**
 * @brief A Re-orginazation layer for yolo v2.
 *
 */
class ReorgLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new Reorg Layer object
   *
   */
  ReorgLayer() : Layer() {}

  /**
   * @brief Destroy the Reorg Layer object
   *
   */
  ~ReorgLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ReorgLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override{};

  inline static const std::string type = "reorg_layer";
};

} // namespace custom

#endif /* __REORGANIZATION_LAYER_H__ */
