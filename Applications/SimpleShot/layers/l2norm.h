// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   l2norm.h
 * @date   09 Jan 2021
 * @brief  This file contains the simple l2norm layer which normalizes
 * the given feature
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __L2NORM__H_
#define __L2NORM__H_
#include <string>

/// @todo migrate these to API
#include <layer_internal.h>
#include <manager.h>

#include <tensor.h>

namespace simpleshot {
namespace layers {

/**
 * @brief Layer class that l2normalizes a feature vector
 *
 */
class L2NormLayer : public nntrainer::LayerV1 {
public:
  /**
   * @brief Construct a new L2norm Layer object
   * that normlizes given feature with l2norm
   */
  L2NormLayer() : LayerV1() {}

  /**
   *  @brief  Move constructor.
   *  @param[in] L2NormLayer &&
   */
  L2NormLayer(L2NormLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs L2NormLayer to be moved.
   */
  L2NormLayer &operator=(L2NormLayer &&rhs) = default;

  /**
   * @brief Destroy the Centering Layer object
   *
   */
  ~L2NormLayer() {}

  using nntrainer::LayerV1::setProperty;

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
   * @brief Get the Type object
   *
   * @return const std::string
   */
  const std::string getType() const override { return L2NormLayer::type; }

  /**
   * @brief get boolean if the function is trainable
   *
   * @retval true trainable
   * @retval false not trainable
   */
  bool getTrainable() noexcept override { return false; }

  inline static const std::string type = "l2norm";
};
} // namespace layers
} // namespace simpleshot

#endif /* __L2NORM__H_ */
