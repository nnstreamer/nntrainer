// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rnn.h
 * @date   17 March 2021
 * @brief  This is Recurrent Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
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
class RNNLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of RNNLayer
   */
  template <typename... Args>
  RNNLayer(
    unsigned int unit_ = 0,
    ActivationType hidden_state_activation_type_ = ActivationType::ACT_NONE,
    bool sequence = false, Args... args) :
    LayerV1(args...),
    unit(unit_),
    hidden_state_activation_type(hidden_state_activation_type_),
    return_sequences(sequence){};

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
  const std::string getType() const override { return RNNLayer::type; };

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
   * @brief     activation type for recurrent : default is tanh
   */
  ActivationType hidden_state_activation_type;

  /**
   * @brief     activation function for h_t : default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     To save hidden state variable ( batch, 1, 1, unit )
   */
  Tensor h_prev;

  /**
   * @brief     opiont for return sequence
   */
  bool return_sequences;

  /**
   * @brief     hidden variable for rnn
   */
  std::shared_ptr<Var_Grad> hidden;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RNN_H__ */
