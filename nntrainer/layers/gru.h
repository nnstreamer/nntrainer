// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gru.h
 * @date   31 March 2021
 * @brief  This is Gated Recurrent Unit Layer Class of Neural Network
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __GRU_H__
#define __GRU_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   GRULayer
 * @brief   GRULayer
 */
class GRULayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of GRULayer
   */
  template <typename... Args>
  GRULayer(
    unsigned int unit_ = 0,
    ActivationType hidden_state_activation_type_ = ActivationType::ACT_NONE,
    ActivationType recurrent_activation_type_ = ActivationType::ACT_NONE,
    bool sequence = false, float dropout = 0.0, Args... args) :
    LayerV1(args...),
    unit(unit_),
    hidden_state_activation_type(hidden_state_activation_type_),
    acti_func(hidden_state_activation_type, true),
    recurrent_activation_type(recurrent_activation_type_),
    recurrent_acti_func(recurrent_activation_type, true),
    return_sequences(sequence),
    dropout_rate(dropout){};

  /**
   * @brief     Destructor of GRULayer
   */
  ~GRULayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] GRULayer &&
   */
  GRULayer(GRULayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs GRULayer to be moved.
   */
  GRULayer &operator=(GRULayer &&rhs) = default;

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
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return GRULayer::type; };

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  static const std::string type;

private:
  /**
   * @brief     hidden state size
   */
  unsigned int unit;

  /**
   * @brief     activation type for hidden state : default is sigmoid
   */
  ActivationType hidden_state_activation_type;

  /**
   * @brief     activation function for h_t : default is sigmoid
   */
  ActiFunc acti_func;

  /**
   * @brief     activation type for recurrent : default is tanh
   */
  ActivationType recurrent_activation_type;

  /**
   * @brief     activation function for recurrent : default is tanh
   */
  ActiFunc recurrent_acti_func;

  /**
   * @brief     To save hidden state variable ( batch, 1, 1, unit )
   */
  Tensor h_prev;

  /**
   * @brief     To save intermediate gates
   */
  std::shared_ptr<Var_Grad> zrg;

  /**
   * @brief     hidden state
   */
  std::shared_ptr<Var_Grad> hidden;

  /**
   * @brief     variable to set return sequences
   */
  bool return_sequences;

  /**
   * @brief     drop out rate
   */
  float dropout_rate;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __GRU_H__ */
