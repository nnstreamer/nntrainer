// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   lstmcell_core.h
 * @date   25 November 2021
 * @brief  This is LSTMCellCore Layer Class of Neural Network
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LSTMCELLCORE_H__
#define __LSTMCELLCORE_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

namespace init_lstm_context {
void fillLayerInitContext(InitLayerContext &context,
                          const InitLayerContext &core_context);
void fillWeights(std::vector<Weight> &weights, const RunLayerContext &context,
                 bool training, const std::vector<unsigned int> &wt_idx,
                 const unsigned int max_timestep, const unsigned int timestep,
                 bool test = false);
const std::vector<Weight *> getWeights(std::vector<Weight> &weights);
void fillInputs(std::vector<Var_Grad> &inputs, RunLayerContext &context,
                bool training, const std::vector<unsigned int> &wt_idx,
                const unsigned int max_timestep, const unsigned int timestep);
const std::vector<Var_Grad *> getInputs(std::vector<Var_Grad> &inputs);
void fillOutputs(std::vector<Var_Grad> &outputs, RunLayerContext &context,
                 bool training, const std::vector<unsigned int> &wt_idx,
                 const unsigned int max_timestep, const unsigned int timestep);
const std::vector<Var_Grad *> getOutputs(std::vector<Var_Grad> &outputs);
void fillTensors(std::vector<Var_Grad> &tensors, RunLayerContext &context,
                 bool training, const std::vector<unsigned int> &wt_idx,
                 const unsigned int max_timestep, const unsigned int timestep);
const std::vector<Var_Grad *> getTensors(std::vector<Var_Grad> &tensors);
} // namespace init_lstm_context

/**
 * @class   LSTMCellCoreLayer
 * @brief   LSTMCellCoreLayer
 */
class LSTMCellCoreLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of LSTMCellLayer
   */
  LSTMCellCoreLayer();

  /**
   * @brief     Destructor of LSTMCellLayer
   */
  ~LSTMCellCoreLayer() = default;

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
  const std::string getType() const override {
    return LSTMCellCoreLayer::type;
  };

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

  inline static const std::string type = "lstmcell_core";

private:
  static constexpr unsigned int NUM_GATE = 4;

  /**
   * Unit: number of output neurons
   * HiddenStateActivation: activation type for hidden state. default is tanh
   * RecurrentActivation: activation type for recurrent. default is sigmoid
   * IntegrateBias: integrate bias_ih, bias_hh to bias_h
   *
   * */
  std::tuple<props::Unit, props::HiddenStateActivation,
             props::RecurrentActivation, props::IntegrateBias>
    lstmcell_core_props;
  std::array<unsigned int, 6> wt_idx; /**< indices of the weights */

  /**
   * @brief     activation function for h_t : default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     activation function for recurrent : default is sigmoid
   */
  ActiFunc recurrent_acti_func;
};

void lstmcell_forwarding(const unsigned int unit, const unsigned int batch_size,
                         const bool disable_bias, const bool integrate_bias,
                         ActiFunc &acti_func, ActiFunc &recurrent_acti_func,
                         const Tensor &input, const Tensor &prev_hidden_state,
                         const Tensor &prev_cell_state, Tensor &hidden_state,
                         Tensor &cell_state, const Tensor &weight_ih,
                         const Tensor &weight_hh, const Tensor &bias_h,
                         const Tensor &bias_ih, const Tensor &bias_hh,
                         Tensor &ifgo);
void lstmcell_calcDerivative(const Tensor &d_ifgo, const Tensor &weight_ih,
                             Tensor &outgoing_derivative);
void lstmcell_calcGradient(
  const unsigned int unit, const unsigned int batch_size,
  const bool disable_bias, const bool integrate_bias, ActiFunc &acti_func,
  ActiFunc &recurrent_acti_func, const Tensor &input,
  const Tensor &prev_hidden_state, Tensor &d_prev_hidden_state,
  const Tensor &prev_cell_state, Tensor &d_prev_cell_state,
  Tensor &d_hidden_state, const Tensor &cell_state, Tensor &d_cell_state,
  Tensor &d_weight_ih, const Tensor &weight_hh, Tensor &d_weight_hh,
  Tensor &d_bias_h, Tensor &d_bias_ih, Tensor &d_bias_hh, const Tensor &ifgo,
  Tensor &d_ifgo);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LSTMCELLCORE_H__ */
