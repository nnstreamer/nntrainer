// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   yolo_v2_loss.cpp
 * @date   07 March 2023
 * @brief  This file contains the yolo v2 loss layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "yolo_v2_loss.h"
#include <nntrainer_log.h>

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum YoloV2LossParams {
  bbox_x_pred,
  bbox_y_pred,
  bbox_w_pred,
  bbox_h_pred,
  confidence_pred,
  class_pred,
  bbox_w_pred_anchor,
  bbox_h_pred_anchor,
  bbox_x_gt,
  bbox_y_gt,
  bbox_w_gt,
  bbox_h_gt,
  confidence_gt,
  class_gt,
  bbox_class_mask,
  iou_mask,
  bbox1_width,
  bbox1_height,
  is_xy_min_max,
  intersection_width,
  intersection_height,
  unions,
};

namespace props {
MaxObjectNumber::MaxObjectNumber(const unsigned &value) { set(value); }
ClassNumber::ClassNumber(const unsigned &value) { set(value); }
GridHeightNumber::GridHeightNumber(const unsigned &value) { set(value); }
GridWidthNumber::GridWidthNumber(const unsigned &value) { set(value); }
} // namespace props

/**
 * @brief mse
 *
 * @param pred prediction
 * @param ground_truth ground truth
 * @return float loss
 * @todo make loss behaves like acti_func
 */
float mse(nntrainer::Tensor &pred, nntrainer::Tensor &ground_truth) {
  nntrainer::Tensor residual;
  pred.subtract(ground_truth, residual);

  float l2norm = residual.l2norm();
  l2norm *= l2norm / residual.size();

  return l2norm;
}

/**
 * @brief backwarding of mse
 *
 * @param pred prediction
 * @param ground_truth ground truth
 * @param outgoing_derivative outgoing derivative
 */
void msePrime(nntrainer::Tensor &pred, nntrainer::Tensor &ground_truth,
              nntrainer::Tensor &outgoing_derivative) {
  pred.subtract(ground_truth, outgoing_derivative);
  float divider = ((float)pred.size()) / 2;
  if (outgoing_derivative.divide_i(divider) != ML_ERROR_NONE) {
    throw std::runtime_error(
      "[YoloV2LossLayer::calcDerivative] Error when calculating loss");
  }
}

/**
 * @brief calculate iou
 *
 * @param bbox1_x1 bbox1_x1
 * @param bbox1_y1 bbox1_y1
 * @param bbox1_w bbox1_w
 * @param bbox1_h bbox1_h
 * @param bbox2_x1 bbox2_x1
 * @param bbox2_y1 bbox2_y1
 * @param bbox2_w bbox2_w
 * @param bbox2_h bbox2_h
 * @param[out] bbox1_width bbox1 width
 * @param[out] bbox1_height bbox1 height
 * @param[out] is_xy_min_max For x1, y1 this value is 1 if x1 > x2, y1 > y2 and
 * for x2, y2 this is value is 1 if x2 < x1, y2 < y1. else 0.
 * @param[out] intersection_width intersection width
 * @param[out] intersection_height intersection height
 * @param[out] unions unions
 * @return nntrainer::Tensor iou
 */
nntrainer::Tensor
calc_iou(nntrainer::Tensor &bbox1_x1, nntrainer::Tensor &bbox1_y1,
         nntrainer::Tensor &bbox1_w, nntrainer::Tensor &bbox1_h,
         nntrainer::Tensor &bbox2_x1, nntrainer::Tensor &bbox2_y1,
         nntrainer::Tensor &bbox2_w, nntrainer::Tensor &bbox2_h,
         nntrainer::Tensor &bbox1_width, nntrainer::Tensor &bbox1_height,
         nntrainer::Tensor &is_xy_min_max,
         nntrainer::Tensor &intersection_width,
         nntrainer::Tensor &intersection_height, nntrainer::Tensor &unions) {
  nntrainer::Tensor bbox1_x2 = bbox1_x1.add(bbox1_w);
  nntrainer::Tensor bbox1_y2 = bbox1_y1.add(bbox1_h);
  nntrainer::Tensor bbox2_x2 = bbox2_x1.add(bbox2_w);
  nntrainer::Tensor bbox2_y2 = bbox2_y1.add(bbox2_h);

  bbox1_x2.subtract(bbox1_x1, bbox1_width);
  bbox1_y2.subtract(bbox1_y1, bbox1_height);
  nntrainer::Tensor bbox1 = bbox1_width.multiply(bbox1_height);

  nntrainer::Tensor bbox2_width = bbox2_x2.subtract(bbox2_x1);
  nntrainer::Tensor bbox2_height = bbox2_y2.subtract(bbox2_y1);
  nntrainer::Tensor bbox2 = bbox2_width.multiply(bbox2_height);

  auto min_func = [&](nntrainer::Tensor &bbox1_xy, nntrainer::Tensor &bbox2_xy,
                      nntrainer::Tensor &intersection_xy) {
    std::transform(bbox1_xy.getData(), bbox1_xy.getData() + bbox1_xy.size(),
                   bbox2_xy.getData(), intersection_xy.getData(),
                   [](float x1, float x2) { return std::min(x1, x2); });
  };
  auto max_func = [&](nntrainer::Tensor &bbox1_xy, nntrainer::Tensor &bbox2_xy,
                      nntrainer::Tensor &intersection_xy) {
    std::transform(bbox1_xy.getData(), bbox1_xy.getData() + bbox1_xy.size(),
                   bbox2_xy.getData(), intersection_xy.getData(),
                   [](float x1, float x2) { return std::max(x1, x2); });
  };

  nntrainer::Tensor intersection_x1(bbox1_x1.getDim());
  nntrainer::Tensor intersection_x2(bbox1_x1.getDim());
  nntrainer::Tensor intersection_y1(bbox1_y1.getDim());
  nntrainer::Tensor intersection_y2(bbox1_y1.getDim());
  max_func(bbox1_x1, bbox2_x1, intersection_x1);
  min_func(bbox1_x2, bbox2_x2, intersection_x2);
  max_func(bbox1_y1, bbox2_y1, intersection_y1);
  min_func(bbox1_y2, bbox2_y2, intersection_y2);

  auto is_min_max_func = [&](nntrainer::Tensor &xy,
                             nntrainer::Tensor &intersection,
                             nntrainer::Tensor &is_min_max) {
    std::transform(xy.getData(), xy.getData() + xy.size(),
                   intersection.getData(), is_min_max.getData(),
                   [](float x, float m) {
                     return nntrainer::absFloat(x - m) < 1e-4 ? 1.0 : 0.0;
                   });
  };

  nntrainer::Tensor is_bbox1_x1_max(bbox1_x1.getDim());
  nntrainer::Tensor is_bbox1_y1_max(bbox1_x1.getDim());
  nntrainer::Tensor is_bbox1_x2_min(bbox1_x1.getDim());
  nntrainer::Tensor is_bbox1_y2_min(bbox1_x1.getDim());
  is_min_max_func(bbox1_x1, intersection_x1, is_bbox1_x1_max);
  is_min_max_func(bbox1_y1, intersection_y1, is_bbox1_y1_max);
  is_min_max_func(bbox1_x2, intersection_x2, is_bbox1_x2_min);
  is_min_max_func(bbox1_y2, intersection_y2, is_bbox1_y2_min);

  nntrainer::Tensor is_bbox_min_max = nntrainer::Tensor::cat(
    {is_bbox1_x1_max, is_bbox1_y1_max, is_bbox1_x2_min, is_bbox1_y2_min}, 3);
  is_xy_min_max.copyData(is_bbox_min_max);

  intersection_x2.subtract(intersection_x1, intersection_width);

  auto type_intersection_width = intersection_width.getDataType();
  if (type_intersection_width == ml::train::TensorDim::DataType::FP32) {
    intersection_width.apply_i<float>(nntrainer::ActiFunc::relu<float>);
  } else if (type_intersection_width == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    intersection_width.apply_i<_FP16>(nntrainer::ActiFunc::relu<_FP16>);
#else
    throw std::runtime_error("Not supported data type");
#endif
  }

  intersection_y2.subtract(intersection_y1, intersection_height);

  auto type_intersection_height = intersection_height.getDataType();
  if (type_intersection_height == ml::train::TensorDim::DataType::FP32) {
    intersection_height.apply_i<float>(nntrainer::ActiFunc::relu<float>);
  } else if (type_intersection_height == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    intersection_height.apply_i<_FP16>(nntrainer::ActiFunc::relu<_FP16>);
#else
    throw std::runtime_error("Not supported data type");
#endif
  }

  nntrainer::Tensor intersection =
    intersection_width.multiply(intersection_height);
  bbox1.add(bbox2, unions);
  unions.subtract_i(intersection);

  return intersection.divide(unions);
}

/**
 * @brief calculate iou graident
 * @details Let say bbox_pred as x, intersection as f(x), union as g(x) and iou
 * as y. Then y = f(x)/g(x). Also g(x) = bbox1 + bbox2 - f(x). Partial
 * derivative of y with respect to x will be (f'(x)g(x) - f(x)g'(x))/(g(x)^2).
 * Partial derivative of g(x) with respect to x will be bbox1'(x) - f'(x).
 * @param confidence_gt_grad incoming derivative for iou
 * @param bbox1_width bbox1_width
 * @param bbox1_height bbox1_height
 * @param is_xy_min_max For x1, y1 this value is 1 if x1 > x2, y1 > y2 and for
 * x2, y2 this is value is 1 if x2 < x1, y2 < y1. else 0.
 * @param intersection_width intersection width
 * @param intersection_height intersection height
 * @param unions unions
 * @return std::vector<nntrainer::Tensor> iou_grad
 */
std::vector<nntrainer::Tensor> calc_iou_grad(
  nntrainer::Tensor &confidence_gt_grad, nntrainer::Tensor &bbox1_width,
  nntrainer::Tensor &bbox1_height, nntrainer::Tensor &is_xy_min_max,
  nntrainer::Tensor &intersection_width, nntrainer::Tensor &intersection_height,
  nntrainer::Tensor &unions) {
  nntrainer::Tensor intersection =
    intersection_width.multiply(intersection_height);

  // 1. calculate intersection local gradient [f'(x)]
  nntrainer::Tensor intersection_width_relu_prime;
  nntrainer::Tensor intersection_height_relu_prime;
  auto type_intersection_width = intersection_width.getDataType();
  if (type_intersection_width == ml::train::TensorDim::DataType::FP32) {
    intersection_width_relu_prime =
      intersection_width.apply<float>(nntrainer::ActiFunc::reluPrime<float>);
  } else if (type_intersection_width == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    intersection_height_relu_prime =
      intersection_height.apply<_FP16>(nntrainer::ActiFunc::reluPrime<_FP16>);
#else
    throw std::runtime_error("Not supported data type");
#endif
  }

  nntrainer::Tensor intersection_x2_local_grad =
    intersection_width_relu_prime.multiply(intersection_height);
  nntrainer::Tensor intersection_y2_local_grad =
    intersection_height_relu_prime.multiply(intersection_width);
  nntrainer::Tensor intersection_x1_local_grad =
    intersection_x2_local_grad.multiply(-1.0);
  nntrainer::Tensor intersection_y1_local_grad =
    intersection_y2_local_grad.multiply(-1.0);

  nntrainer::Tensor intersection_local_grad = nntrainer::Tensor::cat(
    {intersection_x1_local_grad, intersection_y1_local_grad,
     intersection_x2_local_grad, intersection_y2_local_grad},
    3);
  intersection_local_grad.multiply_i(is_xy_min_max);

  // 2. calculate union local gradient [g'(x)]
  nntrainer::Tensor bbox1_x1_grad = bbox1_height.multiply(-1.0);
  nntrainer::Tensor bbox1_y1_grad = bbox1_width.multiply(-1.0);
  nntrainer::Tensor bbox1_x2_grad = bbox1_height;
  nntrainer::Tensor bbox1_y2_grad = bbox1_width;
  nntrainer::Tensor bbox1_grad = nntrainer::Tensor::cat(
    {bbox1_x1_grad, bbox1_y1_grad, bbox1_x2_grad, bbox1_y2_grad}, 3);

  nntrainer::Tensor unions_local_grad =
    bbox1_grad.subtract(intersection_local_grad);

  // 3. calculate iou local gradient [(f'(x)g(x) - f(x)g'(x))/(g(x)^2)]
  nntrainer::Tensor lhs = intersection_local_grad.multiply(unions);
  nntrainer::Tensor rhs = unions_local_grad.multiply(intersection);
  nntrainer::Tensor iou_grad = lhs.subtract(rhs);
  iou_grad.divide_i(unions);
  iou_grad.divide_i(unions);

  // 3. multiply with incoming derivative
  iou_grad.multiply_i(confidence_gt_grad);

  auto splitted_iou_grad = iou_grad.split({1, 1, 1, 1}, 3);
  std::vector<nntrainer::Tensor> ret = {
    splitted_iou_grad[0].add(splitted_iou_grad[2]),
    splitted_iou_grad[1].add(splitted_iou_grad[3]), splitted_iou_grad[2],
    splitted_iou_grad[3]};
  return ret;
}

YoloV2LossLayer::YoloV2LossLayer() :
  anchors_w({1, 1, NUM_ANCHOR, 1}, anchors_w_buf),
  anchors_h({1, 1, NUM_ANCHOR, 1}, anchors_h_buf),
  sigmoid(nntrainer::ActivationType::ACT_SIGMOID, true),
  softmax(nntrainer::ActivationType::ACT_SOFTMAX, true),
  yolo_v2_loss_props(props::MaxObjectNumber(), props::ClassNumber(),
                     props::GridHeightNumber(), props::GridWidthNumber()) {
  anchors_ratio = anchors_w.divide(anchors_h);
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void YoloV2LossLayer::finalize(nntrainer::InitLayerContext &context) {
  nntrainer::TensorDim input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  const unsigned int batch_size = input_dim.batch();
  const unsigned int class_number =
    std::get<props::ClassNumber>(yolo_v2_loss_props).get();
  const unsigned int grid_height_number =
    std::get<props::GridHeightNumber>(yolo_v2_loss_props).get();
  const unsigned int grid_width_number =
    std::get<props::GridWidthNumber>(yolo_v2_loss_props).get();
  const unsigned int max_object_number =
    std::get<props::MaxObjectNumber>(yolo_v2_loss_props).get();
  nntrainer::TensorDim label_dim(batch_size, 1, max_object_number, 5);
  context.setOutputDimensions({label_dim});

  nntrainer::TensorDim bbox_x_pred_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_x_pred] = context.requestTensor(
    bbox_x_pred_dim, "bbox_x_pred", nntrainer::Tensor::Initializer::NONE, true,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_y_pred_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_y_pred] = context.requestTensor(
    bbox_y_pred_dim, "bbox_y_pred", nntrainer::Tensor::Initializer::NONE, true,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_w_pred_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_w_pred] = context.requestTensor(
    bbox_w_pred_dim, "bbox_w_pred", nntrainer::Tensor::Initializer::NONE, true,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_h_pred_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_h_pred] = context.requestTensor(
    bbox_h_pred_dim, "bbox_h_pred", nntrainer::Tensor::Initializer::NONE, true,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim confidence_pred_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::confidence_pred] =
    context.requestTensor(confidence_pred_dim, "confidence_pred",
                          nntrainer::Tensor::Initializer::NONE, true,
                          nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim class_pred_dim(batch_size,
                                      grid_height_number * grid_width_number,
                                      NUM_ANCHOR, class_number);
  wt_idx[YoloV2LossParams::class_pred] = context.requestTensor(
    class_pred_dim, "class_pred", nntrainer::Tensor::Initializer::NONE, true,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_w_pred_anchor_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_w_pred_anchor] =
    context.requestTensor(bbox_w_pred_anchor_dim, "bbox_w_pred_anchor",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_h_pred_anchor_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_h_pred_anchor] =
    context.requestTensor(bbox_h_pred_anchor_dim, "bbox_h_pred_anchor",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_x_gt_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_x_gt] = context.requestTensor(
    bbox_x_gt_dim, "bbox_x_gt", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_y_gt_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_y_gt] = context.requestTensor(
    bbox_y_gt_dim, "bbox_y_gt", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_w_gt_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_w_gt] = context.requestTensor(
    bbox_w_gt_dim, "bbox_w_gt", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_h_gt_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_h_gt] = context.requestTensor(
    bbox_h_gt_dim, "bbox_h_gt", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim confidence_gt_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::confidence_gt] = context.requestTensor(
    confidence_gt_dim, "confidence_gt", nntrainer::Tensor::Initializer::NONE,
    false, nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim class_gt_dim(batch_size,
                                    grid_height_number * grid_width_number,
                                    NUM_ANCHOR, class_number);
  wt_idx[YoloV2LossParams::class_gt] = context.requestTensor(
    class_gt_dim, "class_gt", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox_class_mask_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox_class_mask] =
    context.requestTensor(bbox_class_mask_dim, "bbox_class_mask",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim iou_mask_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::iou_mask] = context.requestTensor(
    iou_mask_dim, "iou_mask", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox1_width_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox1_width] = context.requestTensor(
    bbox1_width_dim, "bbox1_width", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim bbox1_height_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::bbox1_height] = context.requestTensor(
    bbox1_height_dim, "bbox1_height", nntrainer::Tensor::Initializer::NONE,
    false, nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim is_xy_min_max_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 4);
  wt_idx[YoloV2LossParams::is_xy_min_max] = context.requestTensor(
    is_xy_min_max_dim, "is_xy_min_max", nntrainer::Tensor::Initializer::NONE,
    false, nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim intersection_width_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::intersection_width] =
    context.requestTensor(intersection_width_dim, "intersection_width",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim intersection_height_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::intersection_height] =
    context.requestTensor(intersection_height_dim, "intersection_height",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

  nntrainer::TensorDim unions_dim(
    batch_size, grid_height_number * grid_width_number, NUM_ANCHOR, 1);
  wt_idx[YoloV2LossParams::unions] = context.requestTensor(
    unions_dim, "unions", nntrainer::Tensor::Initializer::NONE, false,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);
}

void YoloV2LossLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {
  const unsigned int max_object_number =
    std::get<props::MaxObjectNumber>(yolo_v2_loss_props).get();

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);

  std::vector<nntrainer::Tensor> splited_input =
    input.split({1, 1, 1, 1, 1, max_object_number}, 3);
  nntrainer::Tensor bbox_x_pred_ = splited_input[0];
  nntrainer::Tensor bbox_y_pred_ = splited_input[1];
  nntrainer::Tensor bbox_w_pred_ = splited_input[2];
  nntrainer::Tensor bbox_h_pred_ = splited_input[3];
  nntrainer::Tensor confidence_pred_ = splited_input[4];
  nntrainer::Tensor class_pred_ = splited_input[5];

  nntrainer::Tensor &bbox_x_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_x_pred]);
  nntrainer::Tensor &bbox_y_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_y_pred]);
  nntrainer::Tensor &bbox_w_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_w_pred]);
  nntrainer::Tensor &bbox_h_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_h_pred]);

  nntrainer::Tensor &confidence_pred =
    context.getTensor(wt_idx[YoloV2LossParams::confidence_pred]);
  nntrainer::Tensor &class_pred =
    context.getTensor(wt_idx[YoloV2LossParams::class_pred]);

  nntrainer::Tensor &bbox_w_pred_anchor =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_w_pred_anchor]);
  nntrainer::Tensor &bbox_h_pred_anchor =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_h_pred_anchor]);

  bbox_x_pred.copyData(bbox_x_pred_);
  bbox_y_pred.copyData(bbox_y_pred_);
  bbox_w_pred.copyData(bbox_w_pred_);
  bbox_h_pred.copyData(bbox_h_pred_);

  confidence_pred.copyData(confidence_pred_);
  class_pred.copyData(class_pred_);

  nntrainer::Tensor &bbox_x_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_x_gt]);
  nntrainer::Tensor &bbox_y_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_y_gt]);
  nntrainer::Tensor &bbox_w_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_w_gt]);
  nntrainer::Tensor &bbox_h_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_h_gt]);

  nntrainer::Tensor &confidence_gt =
    context.getTensor(wt_idx[YoloV2LossParams::confidence_gt]);
  nntrainer::Tensor &class_gt =
    context.getTensor(wt_idx[YoloV2LossParams::class_gt]);

  nntrainer::Tensor &bbox_class_mask =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_class_mask]);
  nntrainer::Tensor &iou_mask =
    context.getTensor(wt_idx[YoloV2LossParams::iou_mask]);

  bbox_x_gt.setValue(0);
  bbox_y_gt.setValue(0);
  bbox_w_gt.setValue(0);
  bbox_h_gt.setValue(0);

  confidence_gt.setValue(0);
  class_gt.setValue(0);

  // init mask
  bbox_class_mask.setValue(0);
  iou_mask.setValue(0.5);

  // activate pred
  sigmoid.run_fn(bbox_x_pred, bbox_x_pred);
  sigmoid.run_fn(bbox_y_pred, bbox_y_pred);

  auto type_bbox_w_pred = bbox_w_pred.getDataType();
  if (type_bbox_w_pred == ml::train::TensorDim::DataType::FP32) {
    bbox_w_pred.apply_i<float>(nntrainer::exp_util<float>);
  } else if (type_bbox_w_pred == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    bbox_w_pred.apply_i<_FP16>(nntrainer::exp_util<_FP16>);
#else
    throw std::runtime_error("Not supported data type");
#endif
  }

  auto type_bbox_h_pred = bbox_h_pred.getDataType();
  if (type_bbox_h_pred == ml::train::TensorDim::DataType::FP32) {
    bbox_h_pred.apply_i<float>(nntrainer::exp_util<float>);
  } else if (type_bbox_h_pred == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    bbox_h_pred.apply_i<_FP16>(nntrainer::exp_util<_FP16>);
#else
    throw std::runtime_error("Not supported data type");
#endif
  }

  sigmoid.run_fn(confidence_pred, confidence_pred);
  softmax.run_fn(class_pred, class_pred);

  bbox_w_pred_anchor.copyData(bbox_w_pred);
  bbox_h_pred_anchor.copyData(bbox_h_pred);

  // apply anchors to bounding box
  bbox_w_pred_anchor.multiply_i(anchors_w);
  auto type_bbox_w_pred_anchor = bbox_w_pred_anchor.getDataType();
  if (type_bbox_w_pred_anchor == ml::train::TensorDim::DataType::FP32) {
    bbox_w_pred_anchor.apply_i<float>(nntrainer::sqrtFloat<float>);
  } else if (type_bbox_w_pred_anchor == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    bbox_w_pred_anchor.apply_i<_FP16>(nntrainer::sqrtFloat<_FP16>);
#else
    throw std::runtime_error("Not supported data type");
#endif
  }

  bbox_h_pred_anchor.multiply_i(anchors_h);
  auto type_bbox_h_pred_anchor = bbox_h_pred_anchor.getDataType();
  if (type_bbox_h_pred_anchor == ml::train::TensorDim::DataType::FP32) {
    bbox_h_pred_anchor.apply_i<float>(nntrainer::sqrtFloat<float>);
  } else if (type_bbox_h_pred_anchor == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    bbox_h_pred_anchor.apply_i<_FP16>(nntrainer::sqrtFloat<_FP16>);
#else
    throw std::runtime_error("Not supported data type");
#endif
  }

  generate_ground_truth(context);

  nntrainer::Tensor bbox_pred = nntrainer::Tensor::cat(
    {bbox_x_pred, bbox_y_pred, bbox_w_pred_anchor, bbox_h_pred_anchor}, 3);
  nntrainer::Tensor masked_bbox_pred = bbox_pred.multiply(bbox_class_mask);
  nntrainer::Tensor masked_confidence_pred = confidence_pred.multiply(iou_mask);
  nntrainer::Tensor masked_class_pred = class_pred.multiply(bbox_class_mask);

  nntrainer::Tensor bbox_gt =
    nntrainer::Tensor::cat({bbox_x_gt, bbox_y_gt, bbox_w_gt, bbox_h_gt}, 3);
  nntrainer::Tensor masked_bbox_gt = bbox_gt.multiply(bbox_class_mask);
  nntrainer::Tensor masked_confidence_gt = confidence_gt.multiply(iou_mask);
  nntrainer::Tensor masked_class_gt = class_gt.multiply(bbox_class_mask);

  float bbox_loss = mse(masked_bbox_pred, masked_bbox_gt);
  float confidence_loss = mse(masked_confidence_pred, masked_confidence_gt);
  float class_loss = mse(masked_class_pred, masked_class_gt);

  float loss = 5 * bbox_loss + confidence_loss + class_loss;
  ml_logd("Current iteration loss: %f", loss);
}

void YoloV2LossLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &bbox_x_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_x_pred]);
  nntrainer::Tensor &bbox_x_pred_grad =
    context.getTensorGrad(wt_idx[YoloV2LossParams::bbox_x_pred]);
  nntrainer::Tensor &bbox_y_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_y_pred]);
  nntrainer::Tensor &bbox_y_pred_grad =
    context.getTensorGrad(wt_idx[YoloV2LossParams::bbox_y_pred]);
  nntrainer::Tensor &bbox_w_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_w_pred]);
  nntrainer::Tensor &bbox_w_pred_grad =
    context.getTensorGrad(wt_idx[YoloV2LossParams::bbox_w_pred]);
  nntrainer::Tensor &bbox_h_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_h_pred]);
  nntrainer::Tensor &bbox_h_pred_grad =
    context.getTensorGrad(wt_idx[YoloV2LossParams::bbox_h_pred]);

  nntrainer::Tensor &confidence_pred =
    context.getTensor(wt_idx[YoloV2LossParams::confidence_pred]);
  nntrainer::Tensor &confidence_pred_grad =
    context.getTensorGrad(wt_idx[YoloV2LossParams::confidence_pred]);
  nntrainer::Tensor &class_pred =
    context.getTensor(wt_idx[YoloV2LossParams::class_pred]);
  nntrainer::Tensor &class_pred_grad =
    context.getTensorGrad(wt_idx[YoloV2LossParams::class_pred]);

  nntrainer::Tensor &bbox_w_pred_anchor =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_w_pred_anchor]);
  nntrainer::Tensor &bbox_h_pred_anchor =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_h_pred_anchor]);

  nntrainer::Tensor &bbox_x_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_x_gt]);
  nntrainer::Tensor &bbox_y_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_y_gt]);
  nntrainer::Tensor &bbox_w_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_w_gt]);
  nntrainer::Tensor &bbox_h_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_h_gt]);

  nntrainer::Tensor &confidence_gt =
    context.getTensor(wt_idx[YoloV2LossParams::confidence_gt]);
  nntrainer::Tensor &class_gt =
    context.getTensor(wt_idx[YoloV2LossParams::class_gt]);

  nntrainer::Tensor &bbox_class_mask =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_class_mask]);
  nntrainer::Tensor &iou_mask =
    context.getTensor(wt_idx[YoloV2LossParams::iou_mask]);

  nntrainer::Tensor &bbox1_width =
    context.getTensor(wt_idx[YoloV2LossParams::bbox1_width]);
  nntrainer::Tensor &bbox1_height =
    context.getTensor(wt_idx[YoloV2LossParams::bbox1_height]);
  nntrainer::Tensor &is_xy_min_max =
    context.getTensor(wt_idx[YoloV2LossParams::is_xy_min_max]);
  nntrainer::Tensor &intersection_width =
    context.getTensor(wt_idx[YoloV2LossParams::intersection_width]);
  nntrainer::Tensor &intersection_height =
    context.getTensor(wt_idx[YoloV2LossParams::intersection_height]);
  nntrainer::Tensor &unions =
    context.getTensor(wt_idx[YoloV2LossParams::unions]);

  nntrainer::Tensor bbox_pred = nntrainer::Tensor::cat(
    {bbox_x_pred, bbox_y_pred, bbox_w_pred_anchor, bbox_h_pred_anchor}, 3);
  nntrainer::Tensor masked_bbox_pred = bbox_pred.multiply(bbox_class_mask);
  nntrainer::Tensor masked_confidence_pred = confidence_pred.multiply(iou_mask);
  nntrainer::Tensor masked_class_pred = class_pred.multiply(bbox_class_mask);

  nntrainer::Tensor bbox_gt =
    nntrainer::Tensor::cat({bbox_x_gt, bbox_y_gt, bbox_w_gt, bbox_h_gt}, 3);
  nntrainer::Tensor masked_bbox_gt = bbox_gt.multiply(bbox_class_mask);
  nntrainer::Tensor masked_confidence_gt = confidence_gt.multiply(iou_mask);
  nntrainer::Tensor masked_class_gt = class_gt.multiply(bbox_class_mask);

  nntrainer::Tensor masked_bbox_pred_grad;
  nntrainer::Tensor masked_confidence_pred_grad;
  nntrainer::Tensor masked_confidence_gt_grad;
  nntrainer::Tensor masked_class_pred_grad;

  nntrainer::Tensor confidence_gt_grad;

  msePrime(masked_bbox_pred, masked_bbox_gt, masked_bbox_pred_grad);
  msePrime(masked_confidence_pred, masked_confidence_gt,
           masked_confidence_pred_grad);
  msePrime(masked_confidence_gt, masked_confidence_pred,
           masked_confidence_gt_grad);
  msePrime(masked_class_pred, masked_class_gt, masked_class_pred_grad);

  masked_bbox_pred_grad.multiply_i(5);

  nntrainer::Tensor bbox_pred_grad;

  masked_bbox_pred_grad.multiply(bbox_class_mask, bbox_pred_grad);
  masked_confidence_pred_grad.multiply(iou_mask, confidence_pred_grad);
  masked_confidence_gt_grad.multiply(iou_mask, confidence_gt_grad);
  masked_class_pred_grad.multiply(bbox_class_mask, class_pred_grad);

  std::vector<nntrainer::Tensor> splitted_bbox_pred_grad =
    bbox_pred_grad.split({1, 1, 1, 1}, 3);
  bbox_x_pred_grad.copyData(splitted_bbox_pred_grad[0]);
  bbox_y_pred_grad.copyData(splitted_bbox_pred_grad[1]);
  bbox_w_pred_grad.copyData(splitted_bbox_pred_grad[2]);
  bbox_h_pred_grad.copyData(splitted_bbox_pred_grad[3]);

  // std::vector<nntrainer::Tensor> bbox_pred_iou_grad =
  //   calc_iou_grad(confidence_gt_grad, bbox1_width, bbox1_height,
  //   is_xy_min_max,
  //                 intersection_width, intersection_height, unions);
  // bbox_x_pred_grad.add_i(bbox_pred_iou_grad[0]);
  // bbox_y_pred_grad.add_i(bbox_pred_iou_grad[1]);
  // bbox_w_pred_grad.add_i(bbox_pred_iou_grad[2]);
  // bbox_h_pred_grad.add_i(bbox_pred_iou_grad[3]);

  /**
   * @brief calculate gradient for applying anchors to bounding box
   * @details Let say bbox_pred as x, anchor as c indicated that anchor is
   * constant for bbox_pred and bbox_pred_anchor as y. Then we can denote y =
   * sqrt(cx). Partial derivative of y with respect to x will be
   * sqrt(c)/(2*sqrt(x)) which is equivalent to sqrt(cx)/(2x) and we can replace
   * sqrt(cx) with y.
   * @note divide by bbox_pred(x) will not be executed because bbox_pred_grad
   * will be multiply by bbox_pred(x) soon after.
   */
  bbox_w_pred_grad.multiply_i(bbox_w_pred_anchor);
  bbox_h_pred_grad.multiply_i(bbox_h_pred_anchor);
  /** intended comment */
  // bbox_w_pred_grad.divide_i(bbox_w_pred);
  // bbox_h_pred_grad.divide_i(bbox_h_pred);
  bbox_w_pred_grad.divide_i(2);
  bbox_h_pred_grad.divide_i(2);

  sigmoid.run_prime_fn(bbox_x_pred, bbox_x_pred, bbox_x_pred_grad,
                       bbox_x_pred_grad);
  sigmoid.run_prime_fn(bbox_y_pred, bbox_y_pred, bbox_y_pred_grad,
                       bbox_y_pred_grad);
  /** intended comment */
  // bbox_w_pred_grad.multiply_i(bbox_w_pred);
  // bbox_h_pred_grad.multiply_i(bbox_h_pred);
  sigmoid.run_prime_fn(confidence_pred, confidence_pred, confidence_pred_grad,
                       confidence_pred_grad);
  softmax.run_prime_fn(class_pred, class_pred, class_pred_grad,
                       class_pred_grad);

  nntrainer::Tensor outgoing_derivative_ = nntrainer::Tensor::cat(
    {bbox_x_pred_grad, bbox_y_pred_grad, bbox_w_pred_grad, bbox_h_pred_grad,
     confidence_pred_grad, class_pred_grad},
    3);
  nntrainer::Tensor &outgoing_derivative =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  outgoing_derivative.copyData(outgoing_derivative_);
}

void YoloV2LossLayer::exportTo(nntrainer::Exporter &exporter,
                               const ml::train::ExportMethods &method) const {
  exporter.saveResult(yolo_v2_loss_props, method, this);
}

void YoloV2LossLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, yolo_v2_loss_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[YoloV2LossLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void YoloV2LossLayer::setBatch(nntrainer::RunLayerContext &context,
                               unsigned int batch) {
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_x_pred], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_y_pred], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_w_pred], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_h_pred], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::confidence_pred], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::class_pred], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_w_pred_anchor], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_h_pred_anchor], batch);

  context.updateTensor(wt_idx[YoloV2LossParams::bbox_x_gt], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_y_gt], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_w_gt], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_h_gt], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::confidence_gt], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::class_gt], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox_class_mask], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::iou_mask], batch);

  context.updateTensor(wt_idx[YoloV2LossParams::bbox1_width], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::bbox1_height], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::is_xy_min_max], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::intersection_width], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::intersection_height], batch);
  context.updateTensor(wt_idx[YoloV2LossParams::unions], batch);
}

unsigned int YoloV2LossLayer::find_responsible_anchors(float bbox_ratio) {
  nntrainer::Tensor similarity = anchors_ratio.subtract(bbox_ratio);
  auto data_type = similarity.getDataType();
  if (data_type == ml::train::TensorDim::DataType::FP32) {
    similarity.apply_i<float>(nntrainer::absFloat<float>);
  } else if (data_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    similarity.apply_i<_FP16>(nntrainer::absFloat<_FP16>);
#else
    throw std::runtime_error("Not supported data type");
#endif
  }
  auto data = similarity.getData();

  auto min_iter = std::min_element(data, data + NUM_ANCHOR);
  return std::distance(data, min_iter);
}

void YoloV2LossLayer::generate_ground_truth(
  nntrainer::RunLayerContext &context) {
  const unsigned int max_object_number =
    std::get<props::MaxObjectNumber>(yolo_v2_loss_props).get();
  const unsigned int grid_height_number =
    std::get<props::GridHeightNumber>(yolo_v2_loss_props).get();
  const unsigned int grid_width_number =
    std::get<props::GridWidthNumber>(yolo_v2_loss_props).get();

  nntrainer::Tensor &label = context.getLabel(SINGLE_INOUT_IDX);

  nntrainer::Tensor &bbox_x_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_x_pred]);
  nntrainer::Tensor &bbox_y_pred =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_y_pred]);
  nntrainer::Tensor &bbox_w_pred_anchor =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_w_pred_anchor]);
  nntrainer::Tensor &bbox_h_pred_anchor =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_h_pred_anchor]);

  nntrainer::Tensor &bbox_x_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_x_gt]);
  nntrainer::Tensor &bbox_y_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_y_gt]);
  nntrainer::Tensor &bbox_w_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_w_gt]);
  nntrainer::Tensor &bbox_h_gt =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_h_gt]);

  nntrainer::Tensor &confidence_gt =
    context.getTensor(wt_idx[YoloV2LossParams::confidence_gt]);
  nntrainer::Tensor &class_gt =
    context.getTensor(wt_idx[YoloV2LossParams::class_gt]);

  nntrainer::Tensor &bbox_class_mask =
    context.getTensor(wt_idx[YoloV2LossParams::bbox_class_mask]);
  nntrainer::Tensor &iou_mask =
    context.getTensor(wt_idx[YoloV2LossParams::iou_mask]);

  nntrainer::Tensor &bbox1_width =
    context.getTensor(wt_idx[YoloV2LossParams::bbox1_width]);
  nntrainer::Tensor &bbox1_height =
    context.getTensor(wt_idx[YoloV2LossParams::bbox1_height]);
  nntrainer::Tensor &is_xy_min_max =
    context.getTensor(wt_idx[YoloV2LossParams::is_xy_min_max]);
  nntrainer::Tensor &intersection_width =
    context.getTensor(wt_idx[YoloV2LossParams::intersection_width]);
  nntrainer::Tensor &intersection_height =
    context.getTensor(wt_idx[YoloV2LossParams::intersection_height]);
  nntrainer::Tensor &unions =
    context.getTensor(wt_idx[YoloV2LossParams::unions]);

  const unsigned int batch_size = bbox_x_pred.getDim().batch();

  std::vector<nntrainer::Tensor> splited_label =
    label.split({1, 1, 1, 1, 1}, 3);
  nntrainer::Tensor bbox_x_label = splited_label[0];
  nntrainer::Tensor bbox_y_label = splited_label[1];
  nntrainer::Tensor bbox_w_label = splited_label[2];
  nntrainer::Tensor bbox_h_label = splited_label[3];
  nntrainer::Tensor class_label = splited_label[4];

  bbox_x_label.multiply_i(grid_width_number);
  bbox_y_label.multiply_i(grid_height_number);

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    for (unsigned int object = 0; object < max_object_number; ++object) {
      if (!bbox_w_label.getValue(batch, 0, object, 0) &&
          !bbox_h_label.getValue(batch, 0, object, 0)) {
        break;
      }
      unsigned int grid_x_index = bbox_x_label.getValue(batch, 0, object, 0);
      unsigned int grid_y_index = bbox_y_label.getValue(batch, 0, object, 0);
      unsigned int grid_index = grid_y_index * grid_width_number + grid_x_index;
      unsigned int responsible_anchor =
        find_responsible_anchors(bbox_w_label.getValue(batch, 0, object, 0) /
                                 bbox_h_label.getValue(batch, 0, object, 0));

      bbox_x_gt.setValue(batch, grid_index, responsible_anchor, 0,
                         bbox_x_label.getValue(batch, 0, object, 0) -
                           grid_x_index);
      bbox_y_gt.setValue(batch, grid_index, responsible_anchor, 0,
                         bbox_y_label.getValue(batch, 0, object, 0) -
                           grid_y_index);
      bbox_w_gt.setValue(
        batch, grid_index, responsible_anchor, 0,
        nntrainer::sqrtFloat(bbox_w_label.getValue(batch, 0, object, 0)));
      bbox_h_gt.setValue(
        batch, grid_index, responsible_anchor, 0,
        nntrainer::sqrtFloat(bbox_h_label.getValue(batch, 0, object, 0)));

      class_gt.setValue(batch, grid_index, responsible_anchor,
                        class_label.getValue(batch, 0, object, 0), 1);
      bbox_class_mask.setValue(batch, grid_index, responsible_anchor, 0, 1);
      iou_mask.setValue(batch, grid_index, responsible_anchor, 0, 1);
    }
  }

  nntrainer::Tensor iou = calc_iou(
    bbox_x_pred, bbox_y_pred, bbox_w_pred_anchor, bbox_h_pred_anchor, bbox_x_gt,
    bbox_y_gt, bbox_w_gt, bbox_h_gt, bbox1_width, bbox1_height, is_xy_min_max,
    intersection_width, intersection_height, unions);
  confidence_gt.copyData(iou);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_yolo_v2_loss_layer() {
  auto layer = new YoloV2LossLayer();
  return layer;
}

void destory_yolo_v2_loss_layer(nntrainer::Layer *layer) { delete layer; }

/**
 * @note ml_train_layer_pluggable defines the entry point for nntrainer to
 * register a plugin layer
 */
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_yolo_v2_loss_layer,
                                                   destory_yolo_v2_loss_layer};
}

#endif
} // namespace custom
