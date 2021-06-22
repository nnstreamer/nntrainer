// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   centering.h
 * @date   08 Jan 2021
 * @brief  This file contains the simple centering layer which has hardcoded
 * mean feature vectors from given combinations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CENTERING_H__
#define __CENTERING_H__
#include <string>

/// @todo migrate these to API
#include <layer_internal.h>
#include <manager.h>

#include <tensor.h>

namespace simpleshot {
namespace layers {

/**
 * @brief Center Layer Class that does elementwise
 * subtraction from mean feature vector
 *
 */
class CenteringLayer : public nntrainer::LayerV1 {
public:
  /**
   * @brief Construct a new Centering Layer object
   */
  CenteringLayer() : LayerV1() {}

  /**
   * @brief Construct a new Centering Layer object
   *
   * @param feature_path feature path to read the variable
   */
  CenteringLayer(const std::string &feature_path);

  /**
   *  @brief  Move constructor.
   *  @param[in] CenteringLayer &&
   */
  CenteringLayer(CenteringLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CenteringLayer to be moved.
   */
  CenteringLayer &operator=(CenteringLayer &&rhs) = default;

  /**
   * @brief Destroy the Centering Layer object
   *
   */
  ~CenteringLayer() {}

  using nntrainer::LayerV1::setProperty;

  /**
   * @brief     set Property of layer,
   * feature_path: feature *.bin that contains mean feature vector that will be
   * used for the model.
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
   */
  void forwarding(bool training = true) override;

  /**
   * @brief     calc the derivative to be passed to the previous layer
   */
  void calcDerivative() override;

  /**
   * @brief     read layer Weight & Bias data from file
   * @param[in] file input file stream
   */
  void read(std::ifstream &file) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @brief Get the Type object
   *
   * @return const std::string
   */
  const std::string getType() const override { return CenteringLayer::type; }

  inline static const std::string type = "centering";

private:
  std::string feature_path;
  nntrainer::Tensor mean_feature_vector;
};
} // namespace layers
} // namespace simpleshot

#endif /* __CENTERING_H__ */
