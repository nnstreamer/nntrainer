// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   yolo_v3_loss.h
 * @date   16 June 2023
 * @brief  This file contains the yolo v3 loss layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __YOLO_V3_LOSS_LAYER_H__
#define __YOLO_V3_LOSS_LAYER_H__

#include <string>

#include <acti_func.h>
#include <base_properties.h>
#include <iostream>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace custom {

namespace props {

/**
 * @brief maximum object number in 1 image for given dataset
 *
 */
class MaxObjectNumber final : public nntrainer::PositiveIntegerProperty {
public:
  MaxObjectNumber(const unsigned &value = 1);
  static constexpr const char *key = "max_object_number";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief class number for given dataset
 *
 */
class ClassNumber final : public nntrainer::PositiveIntegerProperty {
public:
  ClassNumber(const unsigned &value = 1);
  static constexpr const char *key = "class_number";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief grid height number
 *
 */
class GridHeightNumber final : public nntrainer::PositiveIntegerProperty {
public:
  GridHeightNumber(const unsigned &value = 1);
  static constexpr const char *key = "grid_height_number";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief grid width number
 *
 */
class GridWidthNumber final : public nntrainer::PositiveIntegerProperty {
public:
  GridWidthNumber(const unsigned &value = 1);
  static constexpr const char *key = "grid_width_number";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief scale of feature pyramid (1: large, 2: medium, 3: small)
 *
 */
class Scale final : public nntrainer::PositiveIntegerProperty {
public:
  Scale(const unsigned &value = 1);
  static constexpr const char *key = "scale";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief Yolo V3 loss layer
 *
 */
class YoloV3LossLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new YoloV3Loss Layer object
   *
   */
  YoloV3LossLayer();

  /**
   * @brief Destroy the YoloV3Loss Layer object
   *
   */
  ~YoloV3LossLayer() {}

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
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const { return true; }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return YoloV3LossLayer::type; };

  inline static const std::string type = "yolo_v3_loss";

private:
  static constexpr unsigned int NUM_ANCHOR = 3;
  const float img_size = 416;
  // empty buffers for anchors
  float anchors_w_buf[NUM_ANCHOR] = {0, 0, 0};
  float anchors_h_buf[NUM_ANCHOR] = {0, 0, 0};
  // anchors for large scale
  float anchors_w_buf_1[NUM_ANCHOR] = {116 / img_size, 156 / img_size,
                                       373 / img_size};
  float anchors_h_buf_1[NUM_ANCHOR] = {90 / img_size, 198 / img_size,
                                       326 / img_size};
  // anchors for medium scale
  float anchors_w_buf_2[NUM_ANCHOR] = {30 / img_size, 62 / img_size,
                                       59 / img_size};
  float anchors_h_buf_2[NUM_ANCHOR] = {61 / img_size, 45 / img_size,
                                       119 / img_size};
  // anchors for small scale
  float anchors_w_buf_3[NUM_ANCHOR] = {10 / img_size, 16 / img_size,
                                       33 / img_size};
  float anchors_h_buf_3[NUM_ANCHOR] = {13 / img_size, 30 / img_size,
                                       23 / img_size};

  nntrainer::Tensor *anchors_w;
  nntrainer::Tensor *anchors_h;
  nntrainer::Tensor anchors_ratio;

  nntrainer::ActiFunc sigmoid; /** sigmoid activation operation */

  std::tuple<props::MaxObjectNumber, props::ClassNumber,
             props::GridHeightNumber, props::GridWidthNumber, props::Scale>
    yolo_v3_loss_props;
  std::array<unsigned int, 22> wt_idx; /**< indices of the weights */

  /**
   * @brief find responsible anchors per object
   */
  unsigned int find_responsible_anchors(float bbox_ratio);

  /**
   * @brief generate ground truth, mask from labels
   */
  void generate_ground_truth(nntrainer::RunLayerContext &context);
};

} // namespace custom

#endif /* __YOLO_V3_LOSS_LAYER_H__ */
