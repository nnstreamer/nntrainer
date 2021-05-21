// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   time_dist.h
 * @date   01 April 2021
 * @brief  This is Time Distributed Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TIME_DIST_H__
#define __TIME_DIST_H__
#ifdef __cplusplus

#include <tensor.h>

namespace nntrainer {

/**
 * @class   TimeDistLayer
 * @brief   Time Distribution Layer
 */
class TimeDistLayer : public Layer {
public:
  /**
   * @brief     Constructor of Time Distribution Layer
   */
  template <typename... Args> TimeDistLayer(Args... args) : Layer(args...) {
    for (unsigned int i = 0; i < 4; ++i) {
      positions[i] = nullptr;
    }
  }

  /**
   * @brief     Destructor of Time Distributed Layer
   */
  ~TimeDistLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] TimeDistLayer &&
   */
  TimeDistLayer(TimeDistLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs TimeDistLayer to be moved.
   */
  TimeDistLayer &operator=(TimeDistLayer &&rhs) = default;

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
   * @brief     set distribute layer
   * @param[in] l layer to distribute along time
   */
  void setDistLayer(std::shared_ptr<Layer> l);

  /**
   * @brief     get distribute layer type
   * @retval layer type
   */
  std::string getDistLayerType() { return dist_layer->getType(); }

  /**
   * @brief     get distribute layer
   * @retval dist_layer std::shared_ptr<Layer>
   */
  std::shared_ptr<Layer> &getDistLayer() { return dist_layer; };

  /**
   * @brief     get transposed Tensor according to time iteration axis
   *            [b, 1, h, w] to [h, 1, b, w]
   * @param[in] m Tensor
   * @retval Tensor transposed Tensor
   */
  Tensor transposeTensor(Tensor &m);

  /**
   * @brief  calculate the pointer of each input and output tensors
   */
  void setPosition();

  /**
   * @brief  Transpose Input and Output Tensors to avoid duplicatation becuase
   * of memory optimization
   * It transpose the net_input.getVariableRef, net_input.getGradientRef,
   * net_hidden.getVariableRef and net_hidden.getGradientRef.
   */
  void transposeInOut();

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override {
    /**
     * @note assumption: name of the dist_layer is set via setName() and not
     * with setProperty()
     */
    dist_layer->setProperty(type, value);
  }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return TimeDistLayer::type; };

  static const std::string type;

private:
  /**
   * @brief Layer to be distributed through time
   */
  std::shared_ptr<Layer> dist_layer;

  /**
   * @brief pointer value of each input/output tensors to compare position
   */
  float *positions[4];
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TIME_DIST_H__ */
