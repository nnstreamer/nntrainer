// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   mae_loss.h
 * @date   10 June 2021
 * @brief  This file contains the mean absoulte error loss as a sample layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __MAE_LOSS_LAYER_H__
#define __MAE_LOSS_LAYER_H__
#include <string>

/// @todo migrate these to API and ensure those headers are exposed to devel
#include <layer_internal.h>
#include <manager.h>
#include <tensor.h>

namespace custom {

/**
 * @brief A sample loss layer which calculates mean absolute error from output
 * @todo update this to LayerV2
 *
 */
class MaeLossLayer final : public nntrainer::LayerV1 {
public:
  /**
   * @brief Construct a new Pow Layer object that does elementwise power
   *
   */
  MaeLossLayer() : LayerV1() {}

  /**
   * @brief Destroy the Pow Layer object
   *
   */
  ~MaeLossLayer() {}

  using nntrainer::LayerV1::setProperty;

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values) override;

  /**
   * @brief initializing nntrainer
   *
   * @return int ML_ERROR_NONE if success
   */
  int initialize(nntrainer::Manager &manager) override;

  /**
   * @brief nntrainer forwarding function
   * @param[in] training true if forwarding is on training
   */
  void forwarding(bool training = true) override;

  /**
   * @brief require label of a function
   *
   * @return bool true if requires label
   */
  bool requireLabel() const override;

  /**
   * @brief     calc the derivative to be passed to the previous layer
   */
  void calcDerivative() override;

  /**
   * @brief Get the type, it must return MaeLossLayer::type
   *
   * @return const std::string get type
   */
  const std::string getType() const override { return MaeLossLayer::type; }

  inline static const std::string type = "mae_loss";
};

} // namespace custom

#endif /* __MAE_LOSS_LAYER_H__ */
