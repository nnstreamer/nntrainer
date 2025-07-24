// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   zoneout_lstmcell.h
 * @date   30 November 2021
 * @brief  This is ZoneoutLSTMCell Layer Class of Neural Network
 * @see	   https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/pdf/1606.01305.pdf
 *         https://github.com/teganmaharaj/zoneout
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __ZONEOUTLSTMCELL_H__
#define __ZONEOUTLSTMCELL_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <lstmcell_core.h>

namespace nntrainer {

/**
 * @class   ZoneoutLSTMCellLayer
 * @brief   ZoneoutLSTMCellLayer
 */
class ZoneoutLSTMCellLayer : public LSTMCore {
public:
  /**
   * @brief HiddenStateZoneOutRate property, this defines zone out rate for
   * hidden state
   *
   */
  class HiddenStateZoneOutRate : public nntrainer::Property<float> {

  public:
    /**
     * @brief Construct a new HiddenStateZoneOutRate object with a default value
     * 0.0
     *
     */
    HiddenStateZoneOutRate(float value = 0.0) :
      nntrainer::Property<float>(value) {}
    static constexpr const char *key =
      "hidden_state_zoneout_rate";   /**< unique key to access */
    using prop_tag = float_prop_tag; /**< property type */

    /**
     * @brief HiddenStateZoneOutRate validator
     *
     * @param v float to validate
     * @retval true if it is equal or greater than 0.0 and equal or smaller than
     * to 1.0
     * @retval false if it is samller than 0.0 or greater than 1.0
     */
    bool isValid(const float &value) const override;
  };

  /**
   * @brief CellStateZoneOutRate property, this defines zone out rate for cell
   * state
   *
   */
  class CellStateZoneOutRate : public nntrainer::Property<float> {

  public:
    /**
     * @brief Construct a new CellStateZoneOutRate object with a default value
     * 0.0
     *
     */
    CellStateZoneOutRate(float value = 0.0) :
      nntrainer::Property<float>(value) {}
    static constexpr const char *key =
      "cell_state_zoneout_rate";     /**< unique key to access */
    using prop_tag = float_prop_tag; /**< property type */

    /**
     * @brief CellStateZoneOutRate validator
     *
     * @param v float to validate
     * @retval true if it is equal or greater than 0.0 and equal or smaller than
     * to 1.0
     * @retval false if it is samller than 0.0 or greater than 1.0
     */
    bool isValid(const float &value) const override;
  };

  /**
   * @brief Test property, this property is set to true when test the zoneout
   * lstmcell in unittest
   *
   */
  class Test : public nntrainer::Property<bool> {

  public:
    /**
     * @brief Construct a new Test object with a default value false
     *
     */
    Test(bool value = false) : nntrainer::Property<bool>(value) {}
    static constexpr const char *key = "test"; /**< unique key to access */
    using prop_tag = bool_prop_tag;            /**< property type */
  };

  /**
   * @brief     Constructor of ZoneoutLSTMCellLayer
   */
  ZoneoutLSTMCellLayer();

  /**
   * @brief     Destructor of ZoneoutLSTMCellLayer
   */
  ~ZoneoutLSTMCellLayer() = default;

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
  const std::string getType() const override {
    return ZoneoutLSTMCellLayer::type;
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

  static constexpr const char *type = "zoneout_lstmcell";

private:
  static constexpr unsigned int NUM_GATE = 4;
  enum INOUT_INDEX {
    INPUT = 0,
    INPUT_HIDDEN_STATE = 1,
    INPUT_CELL_STATE = 2,
    OUTPUT_HIDDEN_STATE = 0,
    OUTPUT_CELL_STATE = 1
  };

  /** common properties like Unit, IntegrateBias, HiddenStateActivation and
   * RecurrentActivation are in lstmcore_props */

  /**
   * HiddenStateZoneOutRate: zoneout rate for hidden_state
   * CellStateZoneOutRate: zoneout rate for cell_state
   * Test: property for test mode
   * MaxTimestep: maximum timestep for zoneout lstmcell
   * TimeStep: timestep for which lstm should operate
   *
   * */
  std::tuple<HiddenStateZoneOutRate, CellStateZoneOutRate, Test,
             props::MaxTimestep, props::Timestep>
    zoneout_lstmcell_props;
  std::array<unsigned int, 9> wt_idx; /**< indices of the weights */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ZONEOUTLSTMCELL_H__ */
