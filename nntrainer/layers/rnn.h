// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	rnn.h
 * @date	17 March 2021
 * @brief	This is Recurrent Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __RNN_H__
#define __RNN_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   RNNLayer
 * @brief   RNNLayer
 */
class RNNLayer : public Layer {
public:
  /**
   * @brief     Constructor of RNNLayer
   */
  template <typename... Args>
  RNNLayer(unsigned int unit_ = 0, Args... args) :
    Layer(args...),
    unit(unit_) {}

  /**
   * @brief     Destructor of RNNLayer
   */
  ~RNNLayer(){};

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
  void copy(std::shared_ptr<Layer> l) override;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return RNNLayer::type; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  void setActivation(ActivationType acti_type) override;

  static const std::string type;

private:
  unsigned int unit;
  ActiFunc acti_func;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RNN_H__ */
