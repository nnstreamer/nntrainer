// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.h
 * @date   19 July 2023
 * @brief  Implementation of RMS normalization function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __RMS_NORM_LAYER_H__
#define __RMS_NORM_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#include <base_properties.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace custom {

namespace props {

/**
 * @brief RMS_NORM_GAMMA_INIT Initialization Enumeration Information
 *
 */
class RMS_NORM_GAMMA_INIT final
  : public nntrainer::EnumProperty<nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a RMS_NORM_GAMMA_INIT object
   */
  RMS_NORM_GAMMA_INIT(nntrainer::Tensor::Initializer value =
                        nntrainer::Tensor::Initializer::ONES) {
    set(value);
  };

  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
};

}; // namespace props

enum RMSParams { gamma };

/**
 * @brief A RMS normalization layer for llama.
 *
 */
class RMSNormLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new RMS normalization layer object
   *
   */
  RMSNormLayer() : Layer() {}

  /**
   * @brief Destroy the RMS normalization layer object
   *
   */
  ~RMSNormLayer() {}

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
  const std::string getType() const override { return RMSNormLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override{};

  inline static const std::string type = "rms_norm";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::RMS_NORM_GAMMA_INIT> rms_props;
};

} // namespace custom

#endif /* __RMS_NORM_LAYER_H__ */
