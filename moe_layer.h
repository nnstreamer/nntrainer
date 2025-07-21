// SPDX-License-Identifier: Apache-2.0
/**
 * @file   moe_layer.h
 * @date   15 January 2025
 * @brief  Optimized Mixture of Experts Layer header
 * @see    https://github.com/EunjuYang/nntrainer/tree/feat/moe_layer_update
 * @author NNTrainer Team
 * @bug    No known bugs except for NYI items
 */

#ifndef __MOE_LAYER_H__
#define __MOE_LAYER_H__
#ifdef __cplusplus

#include <memory>
#include <vector>

#include <layer_impl.h>
#include <tensor.h>

namespace nntrainer {

// Forward declarations for optimization classes
class TensorPool;
struct ExpertCache;

/**
 * @class   MoELayer
 * @brief   Optimized Mixture of Experts Layer
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
  ~MoELayer();

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

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
  // Core MoE parameters
  unsigned int num_experts;      ///< Number of expert networks
  unsigned int top_k;           ///< Number of experts to select per token
  unsigned int expert_capacity; ///< Expert capacity for load balancing
  bool use_expert_cache;        ///< Enable expert caching for incremental forwarding

  // Gate network parameters
  TensorDim gate_dim;     ///< Gate network dimensions
  Tensor gate_weights;    ///< Gate network weights
  Tensor gate_bias;       ///< Gate network bias

  // Expert network parameters
  std::vector<Tensor> expert_weights; ///< Expert network weights
  std::vector<Tensor> expert_bias;    ///< Expert network biases

  // Optimization components
  std::unique_ptr<TensorPool> memory_pool;   ///< Memory pool for tensor reuse
  std::unique_ptr<ExpertCache> expert_cache; ///< Cache for incremental forwarding

  /**
   * @brief Full forward pass implementation
   * @param input Input tensor
   * @param output Output tensor
   * @param context Layer context
   * @param training Whether in training mode
   */
  void forwarding_full(const Tensor &input, Tensor &output, 
                      RunLayerContext &context, bool training);

  /**
   * @brief Optimized forward pass using cached expert outputs
   * @param input Input tensor
   * @param output Output tensor
   * @param context Layer context
   */
  void forwarding_with_cache(const Tensor &input, Tensor &output, 
                            RunLayerContext &context);

  /**
   * @brief Compute gate scores with optimized operations
   * @param input Input tensor
   * @param gate_scores Output gate scores tensor
   */
  void compute_gate_scores_optimized(const Tensor &input, Tensor &gate_scores);

  /**
   * @brief Select top-k experts efficiently
   * @param gate_scores Gate scores tensor
   * @param expert_assignments Output expert assignments per batch
   * @param expert_weights Output expert weights per batch
   */
  void select_top_k_experts_optimized(const Tensor &gate_scores,
                                     std::vector<std::vector<int>> &expert_assignments,
                                     std::vector<std::vector<float>> &expert_weights);

  /**
   * @brief Compute expert outputs only for active experts (sparse computation)
   * @param input Input tensor
   * @param expert_outputs Output expert tensors
   * @param active_experts List of active expert IDs
   */
  void compute_expert_outputs_sparse(const Tensor &input,
                                    std::vector<Tensor*> &expert_outputs,
                                    const std::vector<int> &active_experts);

  /**
   * @brief Compute output for a single expert
   * @param input Input tensor
   * @param output Output tensor
   * @param expert_id Expert ID
   */
  void compute_single_expert_output(const Tensor &input, Tensor &output, int expert_id);

  /**
   * @brief Aggregate expert outputs with optimized weighted combination
   * @param expert_outputs Expert output tensors
   * @param expert_assignments Expert assignments per batch
   * @param expert_weights Expert weights per batch
   * @param output Final output tensor
   */
  void aggregate_expert_outputs_optimized(const std::vector<Tensor*> &expert_outputs,
                                         const std::vector<std::vector<int>> &expert_assignments,
                                         const std::vector<std::vector<float>> &expert_weights,
                                         Tensor &output);

  /**
   * @brief Apply softmax activation in-place for memory efficiency
   * @param tensor Tensor to apply softmax to
   */
  void apply_softmax_inplace(Tensor &tensor);

  /**
   * @brief Check if routing pattern has changed significantly
   * @param current_gate_scores Current gate scores
   * @return True if routing changed significantly
   */
  bool routing_changed_significantly(const Tensor &current_gate_scores);

  /**
   * @brief Update expert cache with current outputs and weights
   * @param expert_outputs Current expert outputs
   * @param weights Current expert weights
   */
  void update_expert_cache(const std::vector<Tensor*> &expert_outputs,
                          const std::vector<float> &weights);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOE_LAYER_H__ */