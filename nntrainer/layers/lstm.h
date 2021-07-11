// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   lstm.h
 * @date   31 March 2021
 * @brief  This is Long Short-Term Memory Layer Class of Neural Network
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LSTM_H__
#define __LSTM_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   LSTMLayer
 * @brief   LSTMLayer
 */
class LSTMLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of LSTMLayer
   */
  LSTMLayer(
    ActivationType hidden_state_activation_type_ = ActivationType::ACT_NONE,
    ActivationType recurrent_activation_type_ = ActivationType::ACT_NONE,
    bool sequence = false) :
    LayerImpl(),
    props(props::Unit()),
    hidden_state_activation_type(hidden_state_activation_type_),
    recurrent_activation_type(recurrent_activation_type_),
    return_sequences(sequence){};

  /**
   * @brief     Destructor of LSTMLayer
   */
  ~LSTMLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] LSTMLayer &&
   */
  LSTMLayer(LSTMLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs LSTMLayer to be moved.
   */
  LSTMLayer &operator=(LSTMLayer &&rhs) = default;

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
  const std::string getType() const override { return LSTMLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "lstm";

private:
  std::tuple<props::Unit>
    props; /**< lstm layer properties : unit - number of output neurons */
  std::array<unsigned int, 6> wt_idx; /**< indices of the weights */

  /**
   * @brief     activation type for recurrent : default is tanh
   */
  ActivationType hidden_state_activation_type;

  /**
   * @brief     activation function for h_t : default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     activation type for recurrent : default is sigmoid
   */
  ActivationType recurrent_activation_type;

  /**
   * @brief     activation function for recurrent : default is sigmoid
   */
  ActiFunc recurrent_acti_func;

  /**
   * @brief     variable to set return sequences
   */
  bool return_sequences;

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
#endif /* __LSTM_H__ */
