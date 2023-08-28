// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   transpose_layer.h
 * @date   21 August 2023
 * @brief  Implementation of transpose layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TRANSPOSE_LAYER_H__
#define __TRANSPOSE_LAYER_H__

#include <complex>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

namespace custom {

/**
 * @brief A tranpose layer.
 *
 */
class TransposeLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new transpose layer object
   *
   */
  TransposeLayer() : Layer() {}

  /**
   * @brief Destroy the transpose layer object
   *
   */
  ~TransposeLayer() {}

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
  const std::string getType() const override { return TransposeLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override{};

  inline static const std::string type = "transpose";
};
} // namespace custom

#endif /* __TRANSPOSE_LAYER_H__ */
