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
#include <lstmcell_core.h>

namespace nntrainer {

/**
 * @class   LSTMLayer
 * @brief   LSTMLayer
 */
class LSTMLayer : public LSTMCore {
public:
  /**
   * @brief     Constructor of LSTMLayer
   */
  LSTMLayer();

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
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return LSTMLayer::type; };

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

  inline static const std::string type = "lstm";

private:
  static constexpr unsigned int NUM_GATE = 4;

  /** common properties like Unit, IntegrateBias, HiddenStateActivation and
   * RecurrentActivation are in lstmcore_props */

  /**
   * ReturnSequence: option for return sequence
   * Bidirectional: option for bidirectional
   * DropOutRate: dropout rate
   * MaxTimestep: maximum timestep for lstm
   *
   * */
  std::tuple<props::ReturnSequences, props::Bidirectional, props::DropOutRate,
             props::MaxTimestep>
    lstm_props;
  std::array<unsigned int, 17> wt_idx; /**< indices of the weights */

  /**
   * @brief run lstm fowarding for batch_first input
   *
   * @param NUM_GATE Number of gate which is 4 for lstm
   * @param batch_size batch size
   * @param feature_size feature size
   * @param disable_bias whether to disable bias or not
   * @param unit number of output neurons
   * @param integrate_bias integrate bias_ih, bias_hh to bias_h
   * @param acti_func activation function for memory cell, cell state
   * @param recurrent_acti_func activation function for input/output/forget
   * gate
   * @param enable_dropout whether to apply dropout
   * @param dropout_rate dropout rate
   * @param max_timestep maximum timestep for lstm
   * @param reverse indicate forward/backward direction for input in
   * bidirectional lstm
   * @param input_ input
   * @param weight_ih weight for input to hidden
   * @param weight_hh weight for hidden to hidden
   * @param bias_h bias for input and hidden.
   * @param bias_ih bias for input
   * @param bias_hh bias for hidden
   * @param hidden_state_ hidden state
   * @param cell_state_ cell state
   * @param ifgo_ input gate, forget gate, memory cell, output gate
   * @param mask_ dropout mask
   */
  void forwardingBatchFirstLSTM(
    unsigned int NUM_GATE, const unsigned int batch_size,
    const unsigned int feature_size, const bool disable_bias,
    const unsigned int unit, const bool integrate_bias, ActiFunc &acti_func,
    ActiFunc &recurrent_acti_func, const bool enable_dropout,
    const float dropout_rate, const unsigned int max_timestep,
    const bool reverse, const Tensor &input_, const Tensor &weight_ih,
    const Tensor &weight_hh, const Tensor &bias_h, const Tensor &bias_ih,
    const Tensor &bias_hh, Tensor &hidden_state_, Tensor &cell_state_,
    Tensor &ifgo_, const Tensor &mask_);

  /**
   * @brief calculate lstm gradient for batch_first input
   *
   * @param NUM_GATE Number of gate which is 4 for lstm
   * @param batch_size batch size
   * @param feature_size feature size
   * @param disable_bias whether to disable bias or not
   * @param unit number of output neurons
   * @param integrate_bias integrate bias_ih, bias_hh to bias_h
   * @param acti_func activation function for memory cell, cell state
   * @param recurrent_acti_func activation function for input/output/forget
   * gate
   * @param return_sequences return sequeces
   * @param bidirectional bidirectional lstm
   * @param enable_dropout whether to apply dropout
   * @param dropout_rate dropout rate
   * @param max_timestep maximum timestep for lstm
   * @param reverse indicate forward/backward direction for input in
   * bidirectional lstm
   * @param input_ input
   * @param incoming_derivative derivative for output which is incoming
   * derivative
   * @param d_weight_ih weight_ih(weight for input to hidden) gradient
   * @param weight_hh weight for hidden to hidden
   * @param d_weight_hh weight_hh(weight for hidden to hidden) gradient
   * @param d_bias_h bias_h(bias for input and hidden) gradient
   * @param d_bias_ih bias_ih(bias for input) gradient
   * @param d_bias_hh bias_hh(bias for hidden) gradient
   * @param hidden_state_ hidden state
   * @param d_hidden_state_ hidden state gradient
   * @param cell_state_ cell state
   * @param d_cell_state_ cell state gradient
   * @param ifgo_ input gate, forget gate, memory cell, output gate
   * @param d_ifgo_ gradient for input gate, forget gate, memory cell, output
   * gate
   * @param mask_ dropout mask
   */
  void calcGradientBatchFirstLSTM(
    unsigned int NUM_GATE, const unsigned int batch_size,
    const unsigned int feature_size, const bool disable_bias,
    const unsigned int unit, const bool integrate_bias, ActiFunc &acti_func,
    ActiFunc &recurrent_acti_func, const bool return_sequences,
    const bool bidirectional, const bool enable_dropout,
    const float dropout_rate, const unsigned int max_timestep,
    const bool reverse, const Tensor &input_, const Tensor &incoming_derivative,
    Tensor &d_weight_ih, const Tensor &weight_hh, Tensor &d_weight_hh,
    Tensor &d_bias_h, Tensor &d_bias_ih, Tensor &d_bias_hh,
    const Tensor &hidden_state_, Tensor &d_hidden_state_,
    const Tensor &cell_state_, Tensor &d_cell_state_, const Tensor &ifgo_,
    Tensor &d_ifgo_, const Tensor &mask_);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LSTM_H__ */
