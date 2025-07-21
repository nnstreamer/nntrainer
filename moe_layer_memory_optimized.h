// SPDX-License-Identifier: Apache-2.0
/**
 * @file   moe_layer_memory_optimized.h
 * @date   15 January 2025
 * @brief  Memory-optimized MoE Layer header avoiding getBatchSlice allocations
 * @see    https://github.com/EunjuYang/nntrainer/blob/6e2a028cd9bc237fa18fdc117f14b65a38c3e9dd/nntrainer/layers/moe_layer.cpp
 * @author NNTrainer Team
 * @bug    No known bugs except for NYI items
 */

#ifndef __MOE_LAYER_MEMORY_OPTIMIZED_H__
#define __MOE_LAYER_MEMORY_OPTIMIZED_H__
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
 * @brief   Memory-optimized Mixture of Experts Layer avoiding getBatchSlice allocations
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
   * @brief Memory-optimized incremental forwarding implementation
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

  // Memory optimization buffers (replaces getBatchSlice allocations)
  unsigned int expert_input_buffer_idx;   ///< Pre-allocated input buffer for experts
  unsigned int expert_output_buffer_idx;  ///< Pre-allocated output buffer for experts

  // Activation function (matches original)
  ActiFunc acti_func;             ///< Activation function (SiLU/Swish)

  /**
   * @brief Memory-optimized expert forward pass avoiding getBatchSlice allocations
   * @param input Input tensor
   * @param token_indices Indices of tokens assigned to this expert
   * @param weights Routing weights for each token
   * @param gate_proj Gate projection weights
   * @param up_proj Up projection weights  
   * @param down_proj Down projection weights
   * @param expert_input_buffer Pre-allocated input buffer
   * @param expert_output_buffer Pre-allocated output buffer
   */
  void compute_expert_forward_memory_optimized(
    const Tensor &input, const std::vector<unsigned> &token_indices,
    const std::vector<float> &weights, const Tensor &gate_proj, 
    const Tensor &up_proj, const Tensor &down_proj,
    Tensor &expert_input_buffer, Tensor &expert_output_buffer);

  /**
   * @brief Copy selected tokens to buffer efficiently (replaces getBatchSlice)
   * @param input Input tensor containing all tokens
   * @param token_indices Indices of tokens to copy
   * @param buffer Pre-allocated buffer to copy tokens to
   */
  void copy_selected_tokens_to_buffer(const Tensor &input, 
                                     const std::vector<unsigned> &token_indices, 
                                     Tensor &buffer);

  /**
   * @brief Apply routing weights efficiently using direct memory access
   * @param expert_output Expert output tensor
   * @param weights Routing weights for each token
   */
  void apply_routing_weights_optimized(Tensor &expert_output, 
                                      const std::vector<float> &weights);

  /**
   * @brief Copy expert output to main output efficiently (replaces getBatchSlice operations)
   * @param expert_output Expert output tensor
   * @param main_output Main output tensor
   * @param token_indices Token indices mapping
   * @param hidden_size Hidden dimension size
   * @param accumulate Whether to accumulate (add_i) or overwrite (copyData)
   */
  void copy_expert_output_to_main_output(const Tensor &expert_output, 
                                        Tensor &main_output,
                                        const std::vector<unsigned> &token_indices, 
                                        unsigned int hidden_size,
                                        bool accumulate);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOE_LAYER_MEMORY_OPTIMIZED_H__ */