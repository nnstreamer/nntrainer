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

namespace nntrainer {

/**
 * @brief Structure to represent routing information for efficient processing
 */
struct RoutingInfo {
  std::vector<std::vector<unsigned int>> expert_token_indices; /**< token indices per expert */
  std::vector<std::vector<float>> expert_token_weights;        /**< weights per expert */
  std::vector<unsigned int> token_expert_counts;               /**< number of experts per token */
  
  void clear() {
    expert_token_indices.clear();
    expert_token_weights.clear();
    token_expert_counts.clear();
  }
};

/**
 * @class   MoELayer
 * @brief   Mixture of Expert Layer
 */
class MoELayer : public LayerImpl {
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
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
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
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return MoELayer::type; };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "moe"; /**< type of the layer */

private:
  unsigned int num_experts; /**< number of experts */
  unsigned int topk;        /**< number of experts per token, i.e., topk */
  ActiFunc acti_func;       /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken, props::Unit,
             props::MoEActivation>
    moe_props;

  // weight indices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int gate_idx;

  // Intermediate tensor indices
  unsigned int router_logits_idx;
  
  // Thread-local pre-allocated tensors for efficient computation (one set per thread)
  static constexpr int MAX_THREADS = 64;  // Reasonable upper bound for OpenMP threads
  std::vector<unsigned int> temp_gate_out_indices;      // [MAX_THREADS] 
  std::vector<unsigned int> temp_up_out_indices;        // [MAX_THREADS]
  std::vector<unsigned int> temp_intermediate_indices;  // [MAX_THREADS]
  std::vector<unsigned int> temp_expert_input_indices;  // [MAX_THREADS]
  std::vector<unsigned int> temp_expert_output_indices; // [MAX_THREADS]

  // Routing information cache
  RoutingInfo routing_cache;

  /**
   * @brief Efficient expert forward computation using pre-allocated tensors
   * @param input_data Raw input data pointer
   * @param output_data Raw output data pointer
   * @param token_indices Token indices for this expert
   * @param token_weights Token weights for this expert
   * @param gate_proj Gate projection weight tensor
   * @param up_proj Up projection weight tensor
   * @param down_proj Down projection weight tensor
   * @param context Run context for accessing temporary tensors
   */
  void compute_expert_forward_optimized(
    const float* input_data,
    float* output_data,
    const std::vector<unsigned int>& token_indices,
    const std::vector<float>& token_weights,
    const Tensor& gate_proj,
    const Tensor& up_proj, 
    const Tensor& down_proj,
    RunLayerContext& context);

  /**
   * @brief Optimized routing computation that avoids expert mask tensor
   * @param router_logits Router logits tensor
   * @param routing_info Output routing information
   */
  void compute_routing_optimized(const Tensor& router_logits, RoutingInfo& routing_info);

  /**
   * @brief Batched GEMM operations for better cache utilization
   * @param input Input tensor
   * @param weight Weight tensor
   * @param output Output tensor
   * @param token_indices Token indices to process
   */
  void batched_gemm(const Tensor& input, const Tensor& weight, Tensor& output,
                   const std::vector<unsigned int>& token_indices);

  /**
   * @brief Debug utility to print MoE layer state for troubleshooting
   */
  void debug_print_state() const;

  inline Tensor compute_expert_forward(const Tensor &input,
                                       const Tensor &weights,
                                       const Tensor &gate_proj_weight,
                                       const Tensor &up_proj_weight,
                                       const Tensor &down_proj_weight);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOE_LAYER_H__ */
