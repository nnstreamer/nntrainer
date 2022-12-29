// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   attention_layer.h
 * @date   1 October 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Attention Layer Class for Neural Network
 *
 */

#ifndef __ATTENTION_LAYER_H__
#define __ATTENTION_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_devel.h>
#include <limits>

namespace nntrainer {

/**
 * @class   Attention Layer
 * @brief   Attention Layer
 */
class AttentionLayer : public virtual Layer {
public:
  /**
   * @brief     Constructor of Attention Layer
   */
  AttentionLayer();

  /**
   * @brief     Destructor of Attention Layer
   */
  ~AttentionLayer();

  /**
   *  @brief  Move constructor of AttentionLayer.
   *  @param[in] AttentionLayer &&
   */
  AttentionLayer(AttentionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs AttentionLayer to be moved.
   */
  AttentionLayer &operator=(AttentionLayer &&rhs) = default;

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
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return AttentionLayer::type; };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  inline static const std::string type = "attention";

protected:
  /**
   * @brief     Finalize the attention layer with the given context
   * @param[in] context InitLayerContext
   *
   * @note This function provides the basic finalize details which can be shared
   * with derived classes as well
   */
  void finalizeCommon(InitLayerContext &context);

  std::tuple<props::ScaledDotProduct> attention_props;

private:
  ActiFunc sm;                        /** softmax activation operation */
  std::array<unsigned int, 4> wt_idx; /**< indices of the weights and tensors */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ATTENTION_LAYER_H__ */
