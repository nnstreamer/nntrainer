// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	loss_layer.h
 * @date	12 June 2020
 * @brief	This is Loss Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __LOSS_LAYER_H__
#define __LOSS_LAYER_H__
#ifdef __cplusplus

#include <layer.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Enumeration of loss function type
 */
enum class LossType {
  LOSS_MSE,             /** Mean Squared Error */
  LOSS_ENTROPY,         /** Cross Entropy */
  LOSS_ENTROPY_SIGMOID, /** Cross Entropy amalgamated with sigmoid for stability
                         */
  LOSS_ENTROPY_SOFTMAX, /** Cross Entropy amalgamated with softmax for stability
                         */
  LOSS_UNKNOWN          /** Unknown */
};

/**
 * @class   LossLayer
 * @brief   loss layer
 */
class LossLayer : public Layer {
public:
  /**
   * @brief     Constructor of Loss Layer
   */
  LossLayer() : loss_type(LossType::LOSS_UNKNOWN) { setType(LAYER_LOSS); };

  /**
   * @brief     Destructor of Loss Layer
   */
  ~LossLayer(){};

  /**
   * @copydoc Layer::forwarding(sharedConstTensor in)
   */
  sharedConstTensor forwarding(sharedConstTensor in);

  /**
   * @brief     Forward Propagation of a layer
   * @param[in] in List of Input Tensors taken by this layer
   * @param[in] label List of Label Tensors for the model
   * @retval    List of Input Tensors as it is.
   */
  sharedConstTensor forwarding(sharedConstTensor in, sharedConstTensor label);

  /**
   * @copydoc Layer::backwarding(sharedConstTensor in, int iteration)
   */
  sharedConstTensor backwarding(sharedConstTensor in, int iteration);

  /**
   * @brief     read layer Weight & Bias data from file
   * @param[in] file input file stream
   */
  void read(std::ifstream &file) {}

  /**
   * @brief     save layer Weight & Bias data from file
   * @param[in] file output file stream
   */
  void save(std::ofstream &file) {}

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize();

  /**
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "Loss"; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

  /**
   * @brief     set loss function
   * @param[in] l loss type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setLoss(LossType l);

private:
  LossType loss_type; /**< loss type of loss layer */

  /**
   * @brief     update loss
   * @param[in] l Tensor data to calculate
   */
  void updateLoss(const Tensor &l);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOSS_LAYER_H__ */
