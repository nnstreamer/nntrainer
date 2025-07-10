// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rnn.h
 * @date   17 March 2021
 * @brief  This is Recurrent Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __RNN_H__
#define __RNN_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   RNNLayer
 * @brief   RNNLayer
 */
class RNNLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of RNNLayer
   */
  NNTR_EXPORT RNNLayer();

  /**
   * @brief     Destructor of RNNLayer
   */
  NNTR_EXPORT ~RNNLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] RNNLayer &&
   */
  NNTR_EXPORT RNNLayer(RNNLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs RNNLayer to be moved.
   */
  NNTR_EXPORT RNNLayer &operator=(RNNLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_EXPORT void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_EXPORT void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_EXPORT void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  NNTR_EXPORT void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_EXPORT void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_EXPORT const std::string getType() const override {
    return RNNLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  NNTR_EXPORT bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  NNTR_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  NNTR_EXPORT void setBatch(RunLayerContext &context, unsigned int batch) override;

  static constexpr const char *type = "rnn";

private:
  /**
   * Unit: number of output neurons
   * HiddenStateActivation: activation type for hidden state. default is tanh
   * ReturnSequence: option for return sequence
   * DropOutRate: dropout rate
   * IntegrateBias: Integrate bias_ih, bias_hh to bias_h
   *
   * */
  std::tuple<props::Unit, props::HiddenStateActivation, props::ReturnSequences,
             props::DropOutRate, props::IntegrateBias>
    rnn_props;
  std::array<unsigned int, 7> wt_idx; /**< indices of the weights */

  /**
   * @brief     activation function for h_t : default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     to pretect overflow
   */
  float epsilon;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RNN_H__ */
