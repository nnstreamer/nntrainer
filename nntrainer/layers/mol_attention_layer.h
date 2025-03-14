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
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   MoL Attention Layer
 * @brief   Mixture of Logistics Attention Layer
 */
class MoLAttentionLayer : public LayerImpl {
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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

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

  static constexpr const char *type = "mol_attention";

private:
  std::tuple<props::Unit, props::MoL_K>
    mol_props; /**< mol attention layer properties : unit - number of output
                  neurons */

  bool helper_exec; /** check if the helper function has already ran */
  ActiFunc softmax; /** softmax activation operation */
  ActiFunc tanh;    /** softmax activation operation */
  ActiFunc sigmoid; /** softmax activation operation */
  std::array<unsigned int, 17>
    wt_idx; /**< indices of the weights and tensors */

  /**
   * @brief Helper function for calculation of the derivative
   *
   * @param context layer context
   * @param dstate to store the derivative of the state
   */
  void calcDerivativeHelper(RunLayerContext &context, Tensor &dstate);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MOL_ATTENTION_LAYER_H__ */
