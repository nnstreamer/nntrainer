// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.h
 * @date   17 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <layer_devel.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Activation Layer
 * @brief   Activation Layer
 */
class ActivationLayer : public Layer {

public:
  /**
   * @brief     Constructor of Activation Layer
   */
  ActivationLayer() : Layer() {
    acti_func.setActiFunc(ActivationType::ACT_NONE);
  }

  /**
   * @brief     Destructor of Activation Layer
   */
  ~ActivationLayer() = default;

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ActivationLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return acti_func.supportInPlace(); }

  inline static const std::string type = "activation";

private:
  ActiFunc
    acti_func; /**< activation function designating the activation operation */

  /**
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)> activation_fn
   * activation function to be used
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
    std::function<Tensor &(Tensor &, Tensor &)> const &activation_prime_fn);

  /**
   * @brief setActivation by custom activation function
   * @note  derivative not applied here as this activation_prime_fn applies
   * derivative itself
   * @param[in] std::function<Tensor(Tensor const &, Tensor &)> activation_fn
   * activation function to be used
   * @param[in] std::function<Tensor(Tensor const &, Tensor &, Tensor const &)>
   * activation_prime_fn activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<Tensor &(Tensor const &, Tensor &)> const &activation_fn,
    std::function<Tensor &(Tensor &, Tensor &, Tensor const &)> const
      &activation_prime_fn);

  /**
   * @brief setActivation by custom activation function
   * @note  apply derivative as this activation_prime_fn does not utilize
   * derivative
   * @param[in] std::function<float(float const &)> activation_fn activation
   *            function to be used
   * @param[in] std::function<float(float const &)> activation_prime_fn
   *            activation_prime_function to be used
   * @retval #ML_ERROR_NONE when successful
   */
  int setActivation(
    std::function<float(float const)> const &activation_fn,
    std::function<float(float const)> const &activation_prime_fn);

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
#endif /* __ACTIVATION_LAYER_H__ */
