// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   lstmcell_core.h
 * @date   25 November 2021
 * @brief  This is lstm core class.
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LSTMCELLCORE_H__
#define __LSTMCELLCORE_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common.h>
#include <layer_impl.h>
#include <node_exporter.h>

namespace nntrainer {

/**
 * @class   LSTMCore
 * @brief   LSTMCore
 */
class LSTMCore : public LayerImpl {
public:
  /**
   * @brief     Constructor of LSTMCore
   */
  LSTMCore();

  /**
   * @brief     Destructor of LSTMCore
   */
  ~LSTMCore() = default;

  /**
   * @brief lstm cell forwarding implementation
   *
   * @param batch_size batch size
   * @param unit number of output neurons
   * @param disable_bias whether to disable bias or not
   * @param integrate_bias integrate bias_ih, bias_hh to bias_h
   * @param acti_func activation function for memory cell, cell state
   * @param recurrent_acti_func activation function for input/output/forget
   * gate
   * @param input input
   * @param prev_hidden_state previous hidden state
   * @param prev_cell_state previous cell state
   * @param hidden_state hidden state
   * @param cell_state cell state
   * @param weight_ih weight for input to hidden
   * @param weight_hh weight for hidden to hidden
   * @param bias_h bias for input and hidden.
   * @param bias_ih bias for input
   * @param bias_hh bias for hidden
   * @param ifgo input gate, forget gate, memory cell, output gate
   */
  void forwardLSTM(const unsigned int batch_size, const unsigned int unit,
                   const bool disable_bias, const bool integrate_bias,
                   ActiFunc &acti_func, ActiFunc &recurrent_acti_func,
                   const Tensor &input, const Tensor &prev_hidden_state,
                   const Tensor &prev_cell_state, Tensor &hidden_state,
                   Tensor &cell_state, const Tensor &weight_ih,
                   const Tensor &weight_hh, const Tensor &bias_h,
                   const Tensor &bias_ih, const Tensor &bias_hh, Tensor &ifgo);

  /**
   * @brief lstm cell calculate derivative implementation
   *
   * @param outgoing_derivative derivative for input
   * @param weight_ih weight for input to hidden
   * @param d_ifgo gradient for input gate, forget gate, memory cell, output
   * gate
   * @param alpha value to be scale outgoing_derivative
   */
  void calcDerivativeLSTM(Tensor &outgoing_derivative, const Tensor &weight_ih,
                          const Tensor &d_ifgo, const float alpha = 0.0f);

  /**
   * @brief lstm cell calculate gradient implementation
   *
   * @param batch_size batch size
   * @param unit number of output neurons
   * @param disable_bias whether to disable bias or not
   * @param integrate_bias integrate bias_ih, bias_hh to bias_h
   * @param acti_func activation function for memory cell, cell state
   * @param recurrent_acti_func activation function for input/output/forget
   * gate
   * @param input input
   * @param prev_hidden_state previous hidden state
   * @param d_prev_hidden_state previous hidden state gradient
   * @param prev_cell_state previous cell state
   * @param d_prev_cell_state previous cell state gradient
   * @param d_hidden_state hidden state gradient
   * @param cell_state cell state
   * @param d_cell_state cell state gradient
   * @param d_weight_ih weight_ih(weight for input to hidden) gradient
   * @param weight_hh weight for hidden to hidden
   * @param d_weight_hh weight_hh(weight for hidden to hidden) gradient
   * @param d_bias_h bias_h(bias for input and hidden) gradient
   * @param d_bias_ih bias_ih(bias for input) gradient
   * @param d_bias_hh bias_hh(bias for hidden) gradient
   * @param ifgo input gate, forget gate, memory cell, output gate
   * @param d_ifgo gradient for input gate, forget gate, memory cell, output
   * gate
   */
  void calcGradientLSTM(const unsigned int batch_size, const unsigned int unit,
                        const bool disable_bias, const bool integrate_bias,
                        ActiFunc &acti_func, ActiFunc &recurrent_acti_func,
                        const Tensor &input, const Tensor &prev_hidden_state,
                        Tensor &d_prev_hidden_state,
                        const Tensor &prev_cell_state,
                        Tensor &d_prev_cell_state, const Tensor &d_hidden_state,
                        const Tensor &cell_state, const Tensor &d_cell_state,
                        Tensor &d_weight_ih, const Tensor &weight_hh,
                        Tensor &d_weight_hh, Tensor &d_bias_h,
                        Tensor &d_bias_ih, Tensor &d_bias_hh,
                        const Tensor &ifgo, Tensor &d_ifgo);

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

protected:
  /**
   * Unit: number of output neurons
   * IntegrateBias: integrate bias_ih, bias_hh to bias_h
   * HiddenStateActivation: activation type for hidden state. default is tanh
   * RecurrentActivation: activation type for recurrent. default is sigmoid
   *
   * */
  std::tuple<props::Unit, props::IntegrateBias, props::HiddenStateActivation,
             props::RecurrentActivation>
    lstmcore_props;

  /**
   * @brief     activation function: default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     activation function for recurrent: default is sigmoid
   */
  ActiFunc recurrent_acti_func;

  /**
   * @brief     to protect overflow
   */
  float epsilon;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LSTMCELLCORE_H__ */
