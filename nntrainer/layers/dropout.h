// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dropout.h
 * @date   05 July 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is DropOut Layer Class for Neural Network
 *
 */

#ifndef __DROPOUT_H__
#define __DROPOUT_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_internal.h>
#include <node_exporter.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   DropOut Layer
 * @brief   DropOut Layer
 */
class DropOutLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of DropOut Layer
   */
  template <typename... Args>
  DropOutLayer(float dropout = 0.0, Args... args) :
    LayerV1(args...),
    dropout_rate(props::DropOutSpec(dropout)) {
    setTrainable(false);
  }

  /**
   * @brief     Destructor of DropOut Layer
   */
  ~DropOutLayer() = default;

  /**
   *  @brief  Move constructor of DropOutLayer.
   *  @param[in] DropOutLayer &&
   */
  DropOutLayer(DropOutLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs DropOutLayer to be moved.
   */
  DropOutLayer &operator=(DropOutLayer &&rhs) = default;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file) override{};

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file) override{};

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return true; }

  /**
   * @copydoc Layer::setProperty(std::vector<std::string> values)
   */
  int setProperty(std::vector<std::string> values) override;

  /**
   * @copydoc Layer::export_to(Exporter &exporter, ExportMethods method)
   */
  void export_to(
    Exporter &exporter,
    ExportMethods method = ExportMethods::METHOD_STRINGVECTOR) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return DropOutLayer::type; };

  inline static const std::string type = "dropout";

private:
  std::tuple<props::DropOutSpec> dropout_rate;
  std::vector<std::shared_ptr<Tensor>> mask;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __DROPOUT_H__ */
