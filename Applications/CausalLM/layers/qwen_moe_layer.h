// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   moe_layer.h
 * @date   09 June 2025
 * @brief  This is Mixture of Expert Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This file is part of the Mixture of Expert Layer implementation.
 *         It does not support shared experts.
 *         This layer is implemented based on the LLama-MoE.
 *         For more information, please refer to the following link:
 *         https://arxiv.org/pdf/2406.16554
 * @todo   This layer does not support backwarding yet.
 */

#ifndef __MOE_LAYER_H__
#define __MOE_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {
/**
 * @brief MoE activation type
 */
class MoEActivation final
  : public nntrainer::EnumProperty<nntrainer::props::ActivationTypeInfo> {
public:
  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "moe_activation";
};
/**
 * @brief NumExperts,  Number of experts property
 */
class NumExperts : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "num_experts"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;        /**< property type */
};

/**
 * @brief NumExpertsPerToken,  Number of experts per token property
 */
class NumExpertsPerToken : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "num_experts_per_token";                 /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};
} // namespace props
/**
 * @class   MoELayer
 * @brief   Mixture of Expert Layer
 */
class MoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Mixture of Expert Layer
   */
  MoELayer();

  /**
   * @brief     Destructor of Mixture of Expert Layer
   */
  ~MoELayer() = default;

  /**
   * @brief  Move constructor.
   *  @param[in] MoELayer &&
   */
  MoELayer(MoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs MoELayer to be moved.
   */
  MoELayer &operator=(MoELayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return MoELayer::type; };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "qwen_moe"; /**< type of the layer */

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token, i.e., topk */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation>
    moe_props;

  // weight indeices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int gate_idx;

  // Intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  inline nntrainer::Tensor
  compute_expert_forward(const nntrainer::Tensor &input,
                         const std::vector<float> &weights,
                         const nntrainer::Tensor &gate_proj_weight,
                         const nntrainer::Tensor &up_proj_weight,
                         const nntrainer::Tensor &down_proj_weight);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __MOE_LAYER_H__ */
