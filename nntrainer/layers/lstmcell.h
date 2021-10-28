// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lstmcell.h
 * @date   31 March 2021
 * @brief  This is LSTMCell Layer Class of Neural Network
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LSTMCELL_H__
#define __LSTMCELL_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   LSTMCellLayer
 * @brief   LSTMCellLayer
 */
class LSTMCellLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of LSTMCellLayer
   */
  LSTMCellLayer();

  /**
   * @brief     Destructor of LSTMCellLayer
   */
  ~LSTMCellLayer() = default;

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
  const std::string getType() const override { return LSTMCellLayer::type; };

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

  inline static const std::string type = "lstmcell";

private:
  static constexpr unsigned int NUM_GATE = 4;

  /**
   * Unit: number of output neurons
   * HiddenStateActivation: activation type for hidden state. default is tanh
   * RecurrentActivation: activation type for recurrent. default is sigmoid
   * DropOutRate: dropout rate
   * TimeStep: timestep for which lstm should operate
   *
   * */
  std::tuple<props::Unit, props::HiddenStateActivation,
             props::RecurrentActivation, props::DropOutRate, props::MaxTimestep,
             props::Timestep>
    lstm_props;
  std::array<unsigned int, 7> wt_idx; /**< indices of the weights */

  /**
   * @brief     activation function for h_t : default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     activation function for recurrent : default is sigmoid
   */
  ActiFunc recurrent_acti_func;

  /**
   * @brief     to protect overflow
   */
  float epsilon;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LSTMCELL_H__ */
