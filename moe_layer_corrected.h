// SPDX-License-Identifier: Apache-2.0
/**
 * @file   moe_layer_corrected.h
 * @date   15 January 2025
 * @brief  Corrected MoE Layer header matching original architecture
 * @see    https://github.com/EunjuYang/nntrainer/blob/6e2a028cd9bc237fa18fdc117f14b65a38c3e9dd/nntrainer/layers/moe_layer.cpp
 * @author NNTrainer Team
 * @bug    No known bugs except for NYI items
 */

#ifndef __MOE_LAYER_CORRECTED_H__
#define __MOE_LAYER_CORRECTED_H__
#ifdef __cplusplus

#include <vector>
#include <tuple>
#include <limits>

#include <acti_func.h>
#include <layer_impl.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   MoELayer
 * @brief   Corrected Mixture of Experts Layer matching original architecture
 */
class MoELayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of MoE Layer
   */
  MoELayer();

  /**
   * @brief     Destructor of MoE Layer
   */
  ~MoELayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @brief Incremental forwarding implementation (matches original exactly)
   * @param context RunLayerContext
   * @param from Starting position 
   * @param to Ending position
   * @param training Training mode flag
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
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter, const ml::train::ExportMethods& method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return MoELayer::type; }

  /**
   * @brief Get the layer type
   */
  inline static const std::string type = "moe";

private:
  // Core MoE parameters (matches original)
  unsigned int num_experts;      ///< Number of expert networks
  unsigned int topk;            ///< Number of experts to select per token (top-k)

  // MoE properties (matches original structure)
  std::tuple<props::NumExperts, props::NumExpertsPerToken, props::Unit, props::MoEActivation> moe_props;

  // Expert weight indices (matches original 3-layer structure)
  std::vector<unsigned int> expert_gate_proj_indices;  ///< Gate projection weights for each expert
  std::vector<unsigned int> expert_up_proj_indices;    ///< Up projection weights for each expert  
  std::vector<unsigned int> expert_down_proj_indices;  ///< Down projection weights for each expert

  // Router and intermediate tensor indices (matches original)
  unsigned int gate_idx;           ///< Router/gate network weight index
  unsigned int router_logits_idx;  ///< Router logits tensor index
  unsigned int expert_mask_idx;    ///< Expert mask tensor index

  // Activation function (matches original)
  ActiFunc acti_func;             ///< Activation function (SiLU/Swish)

  /**
   * @brief Compute expert forward pass with 3-layer structure (matches original exactly)
   * @param input Input tensor
   * @param token_indices Indices of tokens assigned to this expert
   * @param weights Routing weights for each token
   * @param gate_proj Gate projection weights
   * @param up_proj Up projection weights  
   * @param down_proj Down projection weights
   * @return Expert output tensor
   */
  Tensor compute_expert_forward_optimized(const Tensor &input, 
                                         const std::vector<unsigned> &token_indices,
                                         const std::vector<float> &weights,
                                         const Tensor &gate_proj, 
                                         const Tensor &up_proj, 
                                         const Tensor &down_proj);

  /**
   * @brief Optimized top-k computation (uses partial sort internally)
   * @param router_logits Router logits tensor
   * @param k Number of top experts to select
   * @return Tuple of (top_k_values, top_k_indices)
   */
  std::tuple<Tensor, Tensor> compute_optimized_topk(const Tensor &router_logits, 
                                                    unsigned int k);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOE_LAYER_CORRECTED_H__ */