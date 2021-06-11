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

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   LSTMLayer
 * @brief   LSTMLayer
 */
class LSTMLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of LSTMLayer
   */
  template <typename... Args>
  LSTMLayer(
    unsigned int unit_ = 0,
    ActivationType hidden_state_activation_type_ = ActivationType::ACT_NONE,
    ActivationType recurrent_activation_type_ = ActivationType::ACT_NONE,
    bool sequence = false, Args... args) :
    LayerV1(args...),
    unit(unit_),
    hidden_state_activation_type(hidden_state_activation_type_),
    recurrent_activation_type(recurrent_activation_type_),
    return_sequences(sequence){};

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
   * @brief     Activation Type Getter
   * @retval    Activation Type.
   */
  ActivationType getRecurrentActivationType() {
    return this->recurrent_activation_type;
  }

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
  const std::string getType() const override { return LSTMLayer::type; };

  /**
   * @brief     Activation Setter
   * @param[in] activation activation type
   * @throw std::invalid_argument when ActivationType is unknown
   */
  void setRecurrentActivation(ActivationType activation);

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  inline static const std::string type = "lstm";

private:
  /**
   * @brief     hidden state size
   */
  unsigned int unit;

  /**
   * @brief     activation type for recurrent : default is tanh
   */
  ActivationType hidden_state_activation_type;

  /**
   * @brief     activation function for h_t : default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     activation type for recurrent : default is sigmoid
   */
  ActivationType recurrent_activation_type;

  /**
   * @brief     activation function for recurrent : default is sigmoid
   */
  ActiFunc recurrent_acti_func;

  /**
   * @brief     To save hidden state variable ( batch, 1, 1, unit )
   */
  Tensor h_prev;

  /**
   * @brief     To save memory cell variable ( batch, 1, 1, unit )
   */
  Tensor c_prev;

  /**
   * @brief     To save cell data
   */
  std::shared_ptr<Var_Grad> mem_cell;

  /**
   * @brief     To save intermediate gates
   */
  std::shared_ptr<Var_Grad> fgio;

  /**
   * @brief     hidden state
   */
  std::shared_ptr<Var_Grad> hidden;

  /**
   * @brief     variable to set return sequences
   */
  bool return_sequences;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LSTM_H__ */
