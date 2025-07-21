// SPDX-License-Identifier: Apache-2.0
/**
 * @file   moe_layer_simplified.h
 * @date   15 January 2025
 * @brief  Simplified and Efficient MoE Layer header
 * @see    https://github.com/EunjuYang/nntrainer/tree/feat/moe_layer_update
 * @author NNTrainer Team
 * @bug    No known bugs except for NYI items
 */

#ifndef __MOE_LAYER_SIMPLIFIED_H__
#define __MOE_LAYER_SIMPLIFIED_H__
#ifdef __cplusplus

#include <vector>

#include <layer_impl.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   MoELayer
 * @brief   Simplified and Efficient Mixture of Experts Layer
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

  // Gate network parameters
  Tensor gate_weights;    ///< Gate network weights
  Tensor gate_bias;       ///< Gate network bias

  // Expert network parameters
  std::vector<Tensor> expert_weights; ///< Expert network weights
  std::vector<Tensor> expert_bias;    ///< Expert network biases

  /**
   * @brief Compute gate scores efficiently using direct tensor operations
   * @param input Input tensor
   * @param gate_scores Output gate scores tensor
   */
  void compute_gate_scores(const Tensor &input, Tensor &gate_scores);

  /**
   * @brief Apply softmax activation in-place for memory efficiency
   * @param gate_scores Gate scores tensor to apply softmax to
   */
  void apply_softmax_inplace(Tensor &gate_scores);

  /**
   * @brief Compute MoE output using top-k expert selection and direct computation
   * @param input Input tensor
   * @param gate_scores Gate scores after softmax
   * @param output Output tensor
   */
  void compute_moe_output(const Tensor &input, const Tensor &gate_scores, Tensor &output);

  /**
   * @brief Compute single expert's contribution to the output
   * @param input_data Pointer to input data for current sequence position
   * @param output_data Pointer to output data for current sequence position
   * @param expert_id Expert ID
   * @param weight Expert weight from gate
   * @param feature_dim Feature dimension size
   */
  void compute_expert_contribution(const float *input_data, 
                                 float *output_data,
                                 int expert_id, 
                                 float weight, 
                                 unsigned int feature_dim);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOE_LAYER_SIMPLIFIED_H__ */