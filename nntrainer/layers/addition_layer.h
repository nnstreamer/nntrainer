// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.h
 * @date   30 July 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#ifndef __ADDITION_LAYER_H__
#define __ADDITION_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Addition Layer
 * @brief   Addition Layer
 */
class AdditionLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of Addition Layer
   */
  template <typename... Args>
  AdditionLayer(unsigned int num_inputs_ = 1, Args... args) : LayerV1(args...) {
    setNumInputs(num_inputs_);
  }

  /**
   * @brief     Destructor of Addition Layer
   */
  ~AdditionLayer(){};

  /**
   *  @brief  Move constructor of AdditionLayer.
   *  @param[in] AdditionLayer &&
   */
  AdditionLayer(AdditionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs AdditionLayer to be moved.
   */
  AdditionLayer &operator=(AdditionLayer &&rhs) = default;

  /**
   * @brief     initialize layer
   * @param[in] last last layer
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
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return AdditionLayer::type; };

  inline static const std::string type = "addition";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ADDITION_LAYER_H__ */
