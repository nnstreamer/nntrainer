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
#ifndef __CENTROID_KNN_H__
#define __CENTROID_KNN_H__
#include <string>

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @brief Centroid KNN layer which takes centroid and do k-nearest neighbor
 * classification
 */
class CentroidKNN : public Layer {
public:
  /**
   * @brief Construct a new NearestNeighbors Layer object that does elementwise
   * subtraction from mean feature vector
   */
  CentroidKNN();

  /**
   *  @brief  Move constructor.
   *  @param[in] CentroidKNN &&
   */
  CentroidKNN(CentroidKNN &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CentroidKNN to be moved.
   */
  CentroidKNN &operator=(CentroidKNN &&rhs) noexcept = default;

  /**
   * @brief Destroy the NearestNeighbors Layer object
   *
   */
  ~CentroidKNN();

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
                const ml::train::ExportMethods &method) const override;

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
  std::tuple<props::NumClass> centroid_knn_props;
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};
} // namespace nntrainer

#endif /** __CENTROID_KNN_H__ */
