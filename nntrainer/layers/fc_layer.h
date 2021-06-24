// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   fc_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_H__
#define __FC_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_internal.h>
#include <node_exporter.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  template <typename... Args>
  FullyConnectedLayer(unsigned int unit_ = 0, Args... args) :
    LayerV1(args...),
    fc_props(props::Unit(unit_)) {}

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedLayer(){};

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  FullyConnectedLayer(FullyConnectedLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedLayer to be moved.
   */
  FullyConnectedLayer &operator=(FullyConnectedLayer &&rhs) = default;

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @copydoc Layer::calcGradient()
   */
  void calcGradient() override;

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<LayerV1> l) override;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @copydoc Layer::export_to(Exporter &exporter, ExportMethods method)
   */
  void export_to(
    Exporter &exporter,
    ExportMethods method = ExportMethods::METHOD_STRINGVECTOR) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return FullyConnectedLayer::type;
  };

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  inline static const std::string type = "fully_connected";

private:
  std::tuple<props::Unit> fc_props;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_H__ */
