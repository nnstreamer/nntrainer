// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   mol_attention_layer.h
 * @date   11 November 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MoL Attention Layer Class for Neural Network
 *
 */

#ifndef __MOL_ATTENTION_LAYER_H__
#define __MOL_ATTENTION_LAYER_H__
#ifdef __cplusplus

#include <attention_layer.h>

namespace nntrainer {

/**
 * @class   MoL Attention Layer
 * @brief   Mixture of Logistics Attention Layer
 */
class MoLAttentionLayer : public AttentionLayer {
public:
  /**
   * @brief     Constructor of MoL Attention Layer
   */
  MoLAttentionLayer();

  /**
   * @brief     Destructor of MoL Attention Layer
   */
  ~MoLAttentionLayer();

  /**
   *  @brief  Move constructor of MoLAttentionLayer.
   *  @param[in] MoLAttentionLayer &&
   */
  MoLAttentionLayer(MoLAttentionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs MoLAttentionLayer to be moved.
   */
  MoLAttentionLayer &operator=(MoLAttentionLayer &&rhs) = default;

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return MoLAttentionLayer::type;
  };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  inline static const std::string type = "mol_attention";

private:
  ActiFunc sm;                        /** softmax activation operation */
  std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOL_ATTENTION_LAYER_H__ */
