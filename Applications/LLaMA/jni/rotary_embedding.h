// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rotary_embedding.h
 * @date   31 July 2023
 * @brief  Implementation of Rotary Positional Embedding
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __ROTARY_POSITIONAL_EMBEDDING_LAYER_H__
#define __ROTARY_POSITIONAL_EMBEDDING_LAYER_H__

#include <complex>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

namespace custom {

/**
 * @brief A Rotary embedding layer for llama.
 *
 */
class RotaryEmbeddingLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new Rotary embedding layer object
   *
   */
  RotaryEmbeddingLayer() : Layer() {}

  /**
   * @brief Destroy the Rotary embedding layer object
   *
   */
  ~RotaryEmbeddingLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

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
  const std::string getType() const override {
    return RotaryEmbeddingLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override{};

  inline static const std::string type = "rotary_embedding";

private:
  std::vector<std::vector<std::complex<float>>> *freqs_cis;
};
} // namespace custom

#endif /* __ROTARY_POSITIONAL_EMBEDDING_LAYER_H__ */
