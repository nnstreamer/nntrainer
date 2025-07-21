// SPDX-License-Identifier: Apache-2.0
/**
 * @file   moe_layer_zero_copy.h
 * @date   15 January 2025
 * @brief  Zero-copy MoE Layer header using only sharedTensor views (no memcpy)
 * @see    https://github.com/EunjuYang/nntrainer/blob/6e2a028cd9bc237fa18fdc117f14b65a38c3e9dd/nntrainer/layers/moe_layer.cpp
 * @author NNTrainer Team
 * @bug    No known bugs except for NYI items
 */

#ifndef __MOE_LAYER_ZERO_COPY_H__
#define __MOE_LAYER_ZERO_COPY_H__
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
 * @brief   Zero-copy Mixture of Experts Layer using only sharedTensor views
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
   * @brief Zero-copy incremental forwarding implementation
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

  // Zero-copy optimization buffer (minimal intermediate space)
  unsigned int expert_intermediate_buffer_idx;  ///< Buffer for gate_out and up_out only

  // Activation function (matches original)
  ActiFunc acti_func;             ///< Activation function (SiLU/Swish)

  /**
   * @brief Zero-copy expert forward pass using only sharedTensor views
   * @param input Input tensor (scattered access)
   * @param output Output tensor (scattered write)
   * @param token_indices Indices of tokens assigned to this expert
   * @param weights Routing weights for each token
   * @param gate_proj Gate projection weights
   * @param up_proj Up projection weights  
   * @param down_proj Down projection weights
   * @param intermediate_buffer Buffer for intermediate calculations only
   * @param accumulate Whether to accumulate (add_i) or overwrite (copyData)
   */
  void compute_expert_forward_zero_copy(
    const Tensor &input, Tensor &output,
    const std::vector<unsigned> &token_indices,
    const std::vector<float> &weights, const Tensor &gate_proj, 
    const Tensor &up_proj, const Tensor &down_proj,
    Tensor &intermediate_buffer, bool accumulate);

  /**
   * @brief Scattered projection without memcpy (input[indices] × weight → output)
   * @param input_data Input tensor data pointer
   * @param weight_data Weight matrix data pointer
   * @param output_data Output buffer data pointer
   * @param token_indices Indices to access from input
   * @param input_dim Input dimension size
   * @param output_dim Output dimension size
   */
  void compute_scattered_projection(
    const float *input_data, const float *weight_data, float *output_data,
    const std::vector<unsigned> &token_indices,
    unsigned int input_dim, unsigned int output_dim);

  /**
   * @brief Apply SiLU activation in-place with SIMD optimization
   * @param data Data pointer
   * @param size Number of elements
   */
  void apply_silu_inplace(float *data, unsigned int size);

  /**
   * @brief Element-wise multiplication in-place with SIMD optimization
   * @param a First operand (modified in-place)
   * @param b Second operand (read-only)
   * @param size Number of elements
   */
  void multiply_tensors_inplace(float *a, const float *b, unsigned int size);

  /**
   * @brief Final projection with routing weight and scattered output accumulation
   * @param input_data Intermediate result data pointer
   * @param weight_data Down projection weight data pointer
   * @param output_data Output tensor data pointer (scattered write)
   * @param token_indices Token indices for scattered output
   * @param weights Routing weights for each token
   * @param input_dim Intermediate dimension size
   * @param output_dim Output dimension size
   * @param accumulate Whether to accumulate or overwrite
   */
  void compute_final_projection_and_accumulate(
    const float *input_data, const float *weight_data, float *output_data,
    const std::vector<unsigned> &token_indices, const std::vector<float> &weights,
    unsigned int input_dim, unsigned int output_dim, bool accumulate);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOE_LAYER_ZERO_COPY_H__ */