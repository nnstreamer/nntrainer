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
  RNNLayer(
    ActivationType hidden_state_activation_type_ = ActivationType::ACT_NONE,
    bool ret_sequence = false, float dropout = 0.0) :
    LayerImpl(),
    props(props::Unit()),
    wt_idx({0}),
    hidden_state_activation_type(hidden_state_activation_type_),
    acti_func(hidden_state_activation_type, true),
    return_sequences(ret_sequence),
    dropout_rate(dropout){}

  /**
   * @brief     Destructor of RNNLayer
   */
  ~RNNLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] RNNLayer &&
   */
  RNNLayer(RNNLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs RNNLayer to be moved.
   */
  RNNLayer &operator=(RNNLayer &&rhs) = default;

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
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return RNNLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "rnn";

private:
  std::tuple<props::Unit>
    props; /**< rnn layer properties : unit - number of output neurons */
  std::array<unsigned int, 4> wt_idx; /**< indices of the weights */

  /**
   * @brief     activation type for recurrent : default is tanh
   */
  ActivationType hidden_state_activation_type;

  /**
   * @brief     activation function for h_t : default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     opiont for return sequence
   */
  bool return_sequences;

  /**
   * @brief     drop out rate
   */
  float dropout_rate;

  /**
   * @brief setProperty by type and value separated
   * @param[in] type property type to be passed
   * @param[in] value value to be passed
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  void setProperty(const std::string &type, const std::string &value);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RNN_H__ */
