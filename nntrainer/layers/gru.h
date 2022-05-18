// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gru.h
 * @date   31 March 2021
 * @brief  This is Gated Recurrent Unit Layer Class of Neural Network
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __GRU_H__
#define __GRU_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   GRULayer
 * @brief   GRULayer
 */
class GRULayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of GRULayer
   */
  GRULayer();

  /**
   * @brief     Destructor of GRULayer
   */
  ~GRULayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] GRULayer &&
   */
  GRULayer(GRULayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs GRULayer to be moved.
   */
  GRULayer &operator=(GRULayer &&rhs) = default;

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
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return GRULayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  inline static const std::string type = "gru";

private:
  static constexpr unsigned int NUM_GATE = 3;

  /**
   * Unit: number of output neurons
   * HiddenStateActivation: activation type for hidden state. default is tanh
   * RecurrentActivation: activation type for recurrent. default is sigmoid
   * ReturnSequence: option for return sequence
   * DropOutRate: dropout rate
   * IntegrateBias: integrate bias_ih, bias_hh to bias_h
   * ResetAfter: Whether apply reset gate before/after the matrix
   *
   * */
  std::tuple<props::Unit, props::HiddenStateActivation,
             props::RecurrentActivation, props::ReturnSequences,
             props::DropOutRate, props::IntegrateBias, props::ResetAfter>
    gru_props;
  std::array<unsigned int, 9> wt_idx; /**< indices of the weights */

  /**
   * @brief     activation function for h_t : default is sigmoid
   */
  ActiFunc acti_func;

  /**
   * @brief     activation function for recurrent : default is tanh
   */
  ActiFunc recurrent_acti_func;

  /**
   * @brief     to protect overflow
   */
  float epsilon;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __GRU_H__ */
