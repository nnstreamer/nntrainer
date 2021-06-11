// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   centroid_knn.h
 * @date   09 Jan 2021
 * @details  This file contains the simple nearest neighbor layer, this layer
 * takes centroid and calculate l2 distance
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NEAREST_NEIGHBORS_H__
#define __NEAREST_NEIGHBORS_H__
#include <string>

/// @todo migrate these to API
#include <layer_internal.h>
#include <manager.h>

#include <tensor.h>

namespace simpleshot {
namespace layers {

/**
 * @brief Centroid KNN layer which takes centroid and do k-nearest neighbor
 * classification
 */
class CentroidKNN : public nntrainer::LayerV1 {
public:
  /**
   * @brief Construct a new NearestNeighbors Layer object that does elementwise
   * subtraction from mean feature vector
   */
  CentroidKNN() : LayerV1(), num_class(0) {}

  /**
   *  @brief  Move constructor.
   *  @param[in] CentroidKNN &&
   */
  CentroidKNN(CentroidKNN &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CentroidKNN to be moved.
   */
  CentroidKNN &operator=(CentroidKNN &&rhs) = default;

  /**
   * @brief Destroy the NearestNeighbors Layer object
   *
   */
  ~CentroidKNN() {}

  using nntrainer::LayerV1::setProperty;

  /**
   * @brief     set Property of layer,
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
   * @brief nntrainer forwarding function,
   * returns distance vector of shape (num_class, )
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return true; }

  /**
   * @brief     calc the derivative to be passed to the previous layer
   * essentially noop and pass whatever it got.
   */
  void calcDerivative() override;

  /**
   * @brief Get the Type object
   *
   * @return const std::string
   */
  const std::string getType() const override { return CentroidKNN::type; }

  /**
   * @brief get boolean if the function is trainable
   *
   * @retval true trainable
   * @retval false not trainable
   */
  bool getTrainable() noexcept override { return false; }

  inline static const std::string type = "centroid_knn";

private:
  unsigned int num_class;
};
} // namespace layers
} // namespace simpleshot

#endif /** __NEAREST_NEIGHBORS_H__ */
