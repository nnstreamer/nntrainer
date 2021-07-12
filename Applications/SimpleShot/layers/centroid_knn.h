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

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace simpleshot {
namespace layers {

/**
 * @brief Centroid KNN layer which takes centroid and do k-nearest neighbor
 * classification
 */
class CentroidKNN : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new NearestNeighbors Layer object that does elementwise
   * subtraction from mean feature vector
   */
  CentroidKNN() : Layer(), num_class(0) {}

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
  ~CentroidKNN() = default;

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return true; }

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const nntrainer::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return CentroidKNN::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "centroid_knn";

private:
  unsigned int num_class;
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};
} // namespace layers
} // namespace simpleshot

#endif /** __NEAREST_NEIGHBORS_H__ */
