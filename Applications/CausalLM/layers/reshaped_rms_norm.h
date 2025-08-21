// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   reshaped_rms_norm.h
 * @date   15 July 2025
 * @brief  Implementation of RMS normalization function with reshaping.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This layer only supports inference mode.
 */

#ifndef __RMS_NORM_LAYER_H__
#define __RMS_NORM_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#include <base_properties.h>
#include <causallm_common_properties.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace causallm {

/**
 * @brief A custom Reshaped RMS normalization layer for llama.
 *
 */
WIN_EXPORT class ReshapedRMSNormLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new custom RMS normalization layer object
   *
   */
  WIN_EXPORT ReshapedRMSNormLayer() :
    Layer(),
    rms_props(props::RMS_NORM_GAMMA_INIT(), nntrainer::props::Epsilon(),
              props::FeatureSize()),
    feature_size(0) {
    wt_idx.fill(std::numeric_limits<unsigned int>::max());
  }

  /**
   * @brief Destroy the custom RMS normalization layer object
   *
   */
  WIN_EXPORT ~ReshapedRMSNormLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return ReshapedRMSNormLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override {
    auto remain_props = loadProperties(values, rms_props);
    NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
      << "[rms_norm] Unknown Layer Properties count " +
           std::to_string(values.size());
  };

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "reshaped_rms_norm";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::RMS_NORM_GAMMA_INIT, nntrainer::props::Epsilon,
             props::FeatureSize>
    rms_props;

  unsigned int feature_size;
};

} // namespace causallm

#endif /* __CAUSALLM_RMS_NORM_LAYER_H__ */
