// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   refinedet_loss.cpp
 * @date   16 November 2020
 * @brief  This file contains refinedet loss layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <regex>
#include <nntrainer_error.h>
#include <math.h>
#include <acti_func.h>
#include <vector>

#include "concat_layer.h"
#include "layer_context.h"
#include "refinedet_loss.h"
#include "tensor.h"
#include "tensor_wrap_specs.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

const unsigned int feature_map_size1 = 4;
const unsigned int feature_map_size2 = 2;
const unsigned int feature_map_size3 = 1;
const unsigned int feature_map_size4 = 1;
const unsigned int num_ratios = 3;
const unsigned int num_anchors = num_ratios * (
  feature_map_size1 * feature_map_size1 + 
  feature_map_size2 * feature_map_size2 + 
  feature_map_size3 * feature_map_size3 + 
  feature_map_size4 * feature_map_size4);
const unsigned int num_classes = 20;
const float positive_anchor_threshold = 0.5;

using nntrainer::Tensor;
using nntrainer::TensorDim;
using nntrainer::TensorLifespan;


void RefineDetLoss::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[RefineDetLoss] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw std::invalid_argument(msg);
  }
}

// enum RefineDetLossParams {
//   arm_yx_pred,
//   arm_hw_pred,
//   arm_pconf_pred,
//   arm_class_pred,
//   odm_yx_pred,
//   odm_hw_pred,
//   odm_pconf_pred,
//   odm_class_pred,
//   bbox_w_pred_anchor,
//   bbox_h_pred_anchor,
//   bbox_x_gt,
//   bbox_y_gt,
//   bbox_w_gt,
//   bbox_h_gt,
//   confidence_gt,
//   class_gt,
//   bbox_class_mask,
//   iou_mask,
//   bbox1_width,
//   bbox1_height,
//   is_xy_min_max,
//   intersection_width,
//   intersection_height,
//   unions,
// };

void RefineDetLoss::finalize(nntrainer::InitLayerContext &context) {
  const TensorDim &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  TensorDim out_dim = in_dim;
  out_dim.width(4 + num_classes);
  context.setOutputDimensions({out_dim});
  std::cout << context.getName() << " out dim: " << out_dim.batch() << ":" << out_dim.channel() << ":" << out_dim.height() << ":" << out_dim.width()
     << " in dim: "<< in_dim.batch() << ":" << in_dim.channel() << ":" << in_dim.height() << ":" << in_dim.width() << std::endl;

//   TensorDim input_dim =
//     context.getInputDimensions()[SINGLE_INOUT_IDX];
//   const unsigned int batch_size = input_dim.batch();
//   const unsigned int class_number = 20;
//   const unsigned int grid_height_number = 4;
//   const unsigned int grid_width_number = 4;
//   const unsigned int max_object_number = 10;
//   TensorDim label_dim(batch_size, 1, max_object_number, 5);
//   context.setOutputDimensions({label_dim});

//   TensorDim bbox_x_pred_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::arm_yx_pred] = context.requestTensor(
//     bbox_x_pred_dim, "arm_yx_pred", Tensor::Initializer::NONE, true,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_y_pred_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::arm_hw_pred] = context.requestTensor(
//     bbox_y_pred_dim, "arm_hw_pred", Tensor::Initializer::NONE, true,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_w_pred_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::arm_pconf_pred] = context.requestTensor(
//     bbox_w_pred_dim, "arm_pconf_pred", Tensor::Initializer::NONE, true,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_h_pred_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::arm_class_pred] = context.requestTensor(
//     bbox_h_pred_dim, "arm_class_pred", Tensor::Initializer::NONE, true,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim confidence_pred_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::arm_pconf_pred] =
//     context.requestTensor(confidence_pred_dim, "confidence_pred",
//                           Tensor::Initializer::NONE, true,
//                           TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim class_pred_dim({
//     batch_size, num_anchors, num_classes});
//   wt_idx[RefineDetLossParams::arm_class_pred] = context.requestTensor(
//     class_pred_dim, "class_pred", Tensor::Initializer::NONE, true,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_w_pred_anchor_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox_w_pred_anchor] =
//     context.requestTensor(bbox_w_pred_anchor_dim, "bbox_w_pred_anchor",
//                           Tensor::Initializer::NONE, false,
//                           TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_h_pred_anchor_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox_h_pred_anchor] =
//     context.requestTensor(bbox_h_pred_anchor_dim, "bbox_h_pred_anchor",
//                           Tensor::Initializer::NONE, false,
//                           TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_x_gt_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox_x_gt] = context.requestTensor(
//     bbox_x_gt_dim, "bbox_x_gt", Tensor::Initializer::NONE, false,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_y_gt_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox_y_gt] = context.requestTensor(
//     bbox_y_gt_dim, "bbox_y_gt", Tensor::Initializer::NONE, false,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_w_gt_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox_w_gt] = context.requestTensor(
//     bbox_w_gt_dim, "bbox_w_gt", Tensor::Initializer::NONE, false,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_h_gt_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox_h_gt] = context.requestTensor(
//     bbox_h_gt_dim, "bbox_h_gt", Tensor::Initializer::NONE, false,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim confidence_gt_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::confidence_gt] = context.requestTensor(
//     confidence_gt_dim, "confidence_gt", Tensor::Initializer::NONE,
//     false, TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim class_gt_dim({
//     batch_size, num_anchors, num_classes});
//   wt_idx[RefineDetLossParams::class_gt] = context.requestTensor(
//     class_gt_dim, "class_gt", Tensor::Initializer::NONE, false,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox_class_mask_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox_class_mask] =
//     context.requestTensor(bbox_class_mask_dim, "bbox_class_mask",
//                           Tensor::Initializer::NONE, false,
//                           TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim iou_mask_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::iou_mask] = context.requestTensor(
//     iou_mask_dim, "iou_mask", Tensor::Initializer::NONE, false,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox1_width_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox1_width] = context.requestTensor(
//     bbox1_width_dim, "bbox1_width", Tensor::Initializer::NONE, false,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim bbox1_height_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::bbox1_height] = context.requestTensor(
//     bbox1_height_dim, "bbox1_height", Tensor::Initializer::NONE,
//     false, TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim is_xy_min_max_dim({
//     batch_size, num_anchors, 4});
//   wt_idx[RefineDetLossParams::is_xy_min_max] = context.requestTensor(
//     is_xy_min_max_dim, "is_xy_min_max", Tensor::Initializer::NONE,
//     false, TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim intersection_width_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::intersection_width] =
//     context.requestTensor(intersection_width_dim, "intersection_width",
//                           Tensor::Initializer::NONE, false,
//                           TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim intersection_height_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::intersection_height] =
//     context.requestTensor(intersection_height_dim, "intersection_height",
//                           Tensor::Initializer::NONE, false,
//                           TensorLifespan::FORWARD_DERIV_LIFESPAN);

//   TensorDim unions_dim({
//     batch_size, num_anchors, 1});
//   wt_idx[RefineDetLossParams::unions] = context.requestTensor(
//     unions_dim, "unions", Tensor::Initializer::NONE, false,
//     TensorLifespan::FORWARD_DERIV_LIFESPAN);
}

std::vector<Tensor> create_anchors_(
  const unsigned int& anchor_size,
  const unsigned int& stride,
  const unsigned int& feature_map_size
  ) {
  const std::vector<float> anchor_ratios = {0.5, 1, 2};
  
  Tensor topleft_yx = Tensor(feature_map_size, feature_map_size, anchor_ratios.size(), 2);
  for (unsigned int b = 0; b < topleft_yx.batch(); b++ ) {
    for (unsigned int c = 0; c < topleft_yx.channel(); c++) {
        for (unsigned int h = 0; h < topleft_yx.height(); h++) {
            for(unsigned int w = 0; w < topleft_yx.width(); w++) {
                topleft_yx.setValue(b, c, h, w, 
                ((b + 0.5) * (1 - w) + (c * 0.5) * w) * stride);
            }
        }
    }
  }

  std::vector<std::vector<float>> priors_;
  for (float ratio: anchor_ratios) {
    priors_.push_back({anchor_size * ((float)pow(ratio, 0.5)), anchor_size / ((float)pow(ratio, 0.5))});
  }
  Tensor priors = Tensor(priors_);
  priors.reshape(TensorDim({1, 1, anchor_ratios.size(), 2}));

  Tensor anchor_y1x1 = topleft_yx.subtract(priors).divide(2);
  anchor_y1x1.reshape({anchor_y1x1.size() / 2, 2});
  Tensor anchor_y2x2 = topleft_yx.subtract(priors).divide(2);
  anchor_y2x2.reshape({anchor_y2x2.size() / 2, 2});
  Tensor anchor_center = anchor_y1x1.add(anchor_y2x2).divide(2);
  Tensor anchor_hw = anchor_y2x2.subtract(anchor_y1x1);
  return {anchor_y1x1, anchor_y2x2, anchor_center, anchor_hw};
}

std::vector<Tensor> create_anchors() {
  std::vector<Tensor> anchors1 = create_anchors_(8*4, 8, 4);
  std::vector<Tensor> anchors2 = create_anchors_(16*4, 16, 2);
  std::vector<Tensor> anchors3 = create_anchors_(32*4, 32, 1);
  std::vector<Tensor> anchors4 = create_anchors_(64*4,  64,1);
  Tensor anchor_y1x1 = Tensor::cat({anchors1[0], anchors2[0], anchors3[0], anchors4[0]}, 2);
  Tensor anchor_y2x2 = Tensor::cat({anchors1[1], anchors2[1], anchors3[1], anchors4[1]}, 2);
  Tensor anchor_center = Tensor::cat({anchors1[2], anchors2[2], anchors3[2], anchors4[2]}, 2);
  Tensor anchor_hw = Tensor::cat({anchors1[3], anchors2[3], anchors3[3], anchors4[3]}, 2);
  return {anchor_y1x1, anchor_y2x2, anchor_center, anchor_hw}; 
}

// Applies softmax instead of sigmoid
// float binary_cross_entropy(Tensor& x, Tensor& y) {
//   float loss = 0;
//   Tensor output = Tensor(x.getDim());
//   nntrainer::ActiFunc::softmax(x, output);
//   std::vector<unsigned int> label_idx = y.argmax();
//   for (unsigned int a = 0; a < output.batch(); a++) {
//     loss += log(output.getValue(a * num_classes + (label_idx[a] > 0)) + 1e-10);
//   }
//   return -loss / output.batch();
// }

// float categorical_cross_entropy(Tensor& x, Tensor& y) {
//   float loss = 0;
//   Tensor output = Tensor(x.getDim());
//   nntrainer::ActiFunc::softmax(x, output);
//   std::vector<unsigned int> label_idx = y.argmax();
//   for (unsigned int a = 0; a < output.batch(); a++) {
//     loss += log(output.getValue(a * num_classes + label_idx[a]) + 1e-10);
//   }
//   return -loss / output.batch();
// }

float cross_entropy(Tensor& x, std::vector<unsigned int> l) {
  float loss = 0;
  Tensor output = Tensor(x.getDim());
  nntrainer::ActiFunc::softmax(x, output);
  for (unsigned int a = 0; a < output.batch(); a++) {
    loss += log(output.getValue(a * num_classes + l[a]) + 1e-10);
  }
  return -loss / output.batch();
}

float cross_entropy_with_mask(Tensor& x, std::vector<unsigned int> mask, std::vector<unsigned int> l) {
  float loss = 0;
  Tensor output = Tensor(x.getDim());
  nntrainer::ActiFunc::softmax(x, output);
  for (unsigned int a = 0; a < output.batch(); a++) {
    if (!mask[a]) {continue;}
    loss += log(output.getValue(a * num_classes + l[a]) + 1e-10);
  }
  return -loss / output.batch();
}

std::vector<std::pair<unsigned int, float>> cross_entropy_per_anchor(Tensor& x, std::vector<unsigned int> l) {
  std::vector<std::pair<unsigned int, float>> idx_loss(l.size());
  Tensor output = Tensor(x.getDim());
  nntrainer::ActiFunc::softmax(x, output);
  for (unsigned int a = 0; a < output.batch(); a++) {
    idx_loss.push_back(std::pair<unsigned int, float>(a, log(output.getValue(a * num_classes + l[a]) + 1e-10)));
  }
  return idx_loss;
}

float smooth_l1(Tensor& x, Tensor& y, const std::vector<unsigned int> l) {
  x.subtract_i(y);
  x.apply_i([](float val) {
      if (val < 0) {
          val = -val;
      }
      if (val < 1) {
          return 0.5 * val * val;
      } else {
          return val - 0.5;
      }
  });
  x.sum(1);
  for (unsigned int a = 0; a < x.batch(); a++) {
    if (!l[a]) {
      x.setValueInt(a, 0);
    }
  }
  return x.sum(0).getValue(0);
}

// box2 dim=1
std::vector<float> calc_iou(Tensor box1_yx, Tensor box1_hw, Tensor box2_yx, Tensor box2_hw) {
  Tensor box1_yx2 = box1_yx.add(box1_hw);

  std::vector<Tensor> box1_yx_split = box1_yx.split(2, 3);
  Tensor box1_y1 = box1_yx_split[0];
  Tensor box1_x1 = box1_yx_split[1];
  std::vector<Tensor> box1_yx2_split = box1_yx2.split(2, 3);
  Tensor box1_y2 = box1_yx2_split[0];
  Tensor box1_x2 = box1_yx2_split[1];

  Tensor box2_yx2 = box2_yx.add(box2_hw);

  std::vector<Tensor> box2_yx_split = box2_yx.split(2, 3);
  Tensor box2_y1 = box2_yx_split[0];
  Tensor box2_x1 = box2_yx_split[1];
  std::vector<Tensor> box2_yx2_split = box2_yx2.split(2, 3);
  Tensor box2_y2 = box2_yx2_split[0];
  Tensor box2_x2 = box2_yx2_split[1];

  auto min_func = [&](Tensor &bbox1_xy, Tensor &bbox2_xy, Tensor &intersection_xy) {
    std::transform(bbox1_xy.getData(), bbox1_xy.getData() + bbox1_xy.size(), intersection_xy.getData(),
                   [&bbox2_xy](float x1) { return std::min(x1, bbox2_xy.getValue(0)); });
  };
  auto max_func = [&](Tensor &bbox1_xy, Tensor &bbox2_xy, Tensor &intersection_xy) {
    std::transform(bbox1_xy.getData(), bbox1_xy.getData() + bbox1_xy.size(), intersection_xy.getData(),
                   [&bbox2_xy](float x1) { return std::max(x1, bbox2_xy.getValue(0)); });
  };

  unsigned int num_anc = box1_x1.getDim()[2];
  Tensor inter_x1(num_anc);
  Tensor inter_y1(num_anc);
  Tensor inter_x2(num_anc);
  Tensor inter_y2(num_anc);
  max_func(box1_x1, box2_x1, inter_x1);
  min_func(box1_x2, box2_x2, inter_x2);
  max_func(box1_y1, box2_y1, inter_y1);
  min_func(box1_y2, box2_y2, inter_y2);


  std::vector<Tensor> box1_hw_split = box1_hw.split(2, 3);
  Tensor box1_h = box1_hw_split[0];
  Tensor box1_w = box1_hw_split[1];
  std::vector<Tensor> box2_hw_split = box2_hw.split(2, 3);
  Tensor box2_h = box2_hw_split[0];
  Tensor box2_w = box2_hw_split[1];

  Tensor inter_area = inter_x2.subtract(inter_x1).apply(nntrainer::ActiFunc::relu).multiply(
    inter_y2.subtract(inter_y1).apply(nntrainer::ActiFunc::relu));
  float* iou_vec = inter_area.divide(box1_h.multiply(box1_w).add(box2_h.multiply(box2_w)).subtract(inter_area)).getData();
  std::cout<<"hi3"<<std::endl;
  return std::vector<float>(iou_vec, iou_vec + num_anc);
}

void RefineDetLoss::forwarding(nntrainer::RunLayerContext &context, bool training) { 
  std::cout << "refinedet forwarding 1" << std::endl;
  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gt = context.getLabel(SINGLE_INOUT_IDX);
  std::cout << "refinedet forwarding 1.1" << std::endl;
  std::vector<Tensor> input_split = input.split({2, 2, 2, 2, 2, num_classes}, 3);
  Tensor& arm_yx = input_split[0];
  Tensor& arm_hw = input_split[1];
  Tensor& arm_conf = input_split[2];
  Tensor& odm_yx = input_split[3];
  Tensor& odm_hw = input_split[4];
  Tensor& odm_conf = input_split[5];
  std::cout << "refinedet forwarding 1.2" << std::endl;
  std::vector<Tensor> gt_split = gt.split({2, 2, num_classes}, 3);
  Tensor& gt_yx = gt_split[0];
  Tensor& gt_hw = gt_split[1];
  Tensor& gt_class = gt_split[2];
  std::cout << "refinedet forwarding 1.3" << std::endl;
  std::vector<Tensor> anchors = create_anchors();
  unsigned int anchors_num = anchors[0].height();
  std::cout << "refinedet forwarding 2" << std::endl;
  for (unsigned int b = 0; b < arm_conf.batch(); b++) {
    Tensor arm_conf_ = arm_conf.getBatchSlice(b, 1);
    Tensor arm_yx_ = arm_yx.getBatchSlice(b, 1);
    Tensor arm_hw_ = arm_hw.getBatchSlice(b, 1);
    Tensor odm_conf_ = odm_conf.getBatchSlice(b, 1);
    Tensor odm_yx_ = odm_yx.getBatchSlice(b, 1);
    Tensor odm_hw_ = odm_hw.getBatchSlice(b, 1);
    Tensor gt_class_ = gt_class.getBatchSlice(b, 1);
    Tensor gt_yx_ = gt_yx.getBatchSlice(b, 1);
    Tensor gt_hw_ = gt_hw.getBatchSlice(b, 1);
    // arm_conf_.reshape({arm_conf_.size() / 2, 2});
    // arm_yx_.reshape({arm_yx_.size() / 2, 2});
    // arm_hw.reshape({arm_hw_.size() / 2, 2});
    // gt_class_.reshape({gt_class_.size() / num_classes, num_classes});
    // gt_yx_.reshape({gt_yx_.size() / 2, 2});
    // gt_hw_.reshape({gt_hw_.size() / 2, 2});
    std::cout << "refinedet forwarding 3" << std::endl;
    // Search anchor with best iou or higher iou than 0.5
    std::set<unsigned int> positive_idx_set = {};
    std::vector<int> anchor_gt_label_idx(anchors_num, -1);
    std::vector<float> anchor_gt_label_iou(anchors_num);

    // std::cout << gt_yx_.batch() << " " << gt_yx_.channel() << " " << gt_yx_.height() << " " << gt_yx_.width() << " " << std::endl;
    std::vector<Tensor> gt_yx_boxes = gt_yx_.split(gt_class_.height(), 2);
    std::vector<Tensor> gt_hw_boxes = gt_hw_.split(gt_class_.height(), 2);
    for (unsigned int gt = 0; gt < gt_class_.height(); gt++) {
      std::cout << "refinedet forwarding 4" << std::endl;
      std::vector<float> anc_gt_iou = calc_iou(anchors[0], anchors[3], gt_yx_boxes[gt], gt_hw_boxes[gt]);
      unsigned int max_idx = *std::max_element(anc_gt_iou.begin(), anc_gt_iou.end());
      if (anchor_gt_label_iou[max_idx] < anc_gt_iou[max_idx]) {
        anchor_gt_label_idx[max_idx] = gt;
        anchor_gt_label_iou[max_idx] = anc_gt_iou[max_idx];
      }
      positive_idx_set.insert(max_idx);
      for (unsigned int i = 0; i < anchors_num; i++) {
        std::cout << "refinedet forwarding 5" << std::endl;
        if (anc_gt_iou[i] > 0.5) {
          positive_idx_set.insert(i);
        }
      }
    }
    unsigned int num_positive_anchors = positive_idx_set.size();
    std::vector<unsigned int> positive_mask(anchors_num);
    for (auto& i : positive_idx_set) {
      positive_mask[i] = 1;
    }
    std::cout << "refinedet forwarding 6" << std::endl;
    // ARM loss
    output.add_i(cross_entropy(arm_conf_, positive_mask) / num_positive_anchors);

    auto log_ = [&](float val) {return (float)log(val);};
    Tensor gt_yx_ratio = gt_yx_.subtract(anchors[0]).divide(anchors[1]);
    Tensor gt_hw_log = gt_hw_.divide(anchors[1]).apply(log_);
    Tensor gt_yxhw = Tensor::cat({gt_yx_ratio, gt_hw_log}, 1);
    Tensor arm_yxhw = Tensor::cat({arm_yx_, arm_hw_}, 1);
    output.add_i(smooth_l1(arm_yxhw, gt_yxhw, positive_mask) / num_positive_anchors);
    std::cout << "refinedet forwarding 7" << std::endl;
    // ODM loss
    // get initial positive boxes from arm/gt iou??
    // Negative anchor filtering
    unsigned int num_negative_anchors = anchors_num - num_positive_anchors;
    std::vector<unsigned int> negative_mask(anchors_num);
    std::vector<unsigned int> ones(anchors_num);
    ones.assign(anchors_num, 1);
    std::transform(ones.begin(), ones.end(), positive_mask.begin(), negative_mask.begin(), std::minus<unsigned int>());
    for (unsigned int i = 0; i < anchors_num; i++) {
      if (arm_conf_.getValue(2 * i + 1) > 0.99 && negative_mask[i]) {
        negative_mask[i] = 0;
        num_negative_anchors--;
      }
    }
    std::cout << "refinedet forwarding 8" << std::endl;
    // Hard negative mining
    std::vector<std::pair<unsigned int, float>> arm_loss_per_anchor = cross_entropy_per_anchor(arm_conf_, positive_mask);
    sort(arm_loss_per_anchor.begin(), arm_loss_per_anchor.end(), 
      [&](std::pair<unsigned int, float> v1, std::pair<unsigned int, float>v2) {return v1.second < v2.second;});
    if (num_negative_anchors > 3 * num_positive_anchors) {
      num_negative_anchors = 3 * num_positive_anchors;
      arm_loss_per_anchor.resize(num_negative_anchors);
    }
    std::cout << "refinedet forwarding 9" << std::endl;
    // Should NMS be done here??
    for (unsigned int i = 0; i < num_negative_anchors; i++) {
      unsigned int ith = arm_loss_per_anchor[i].first;
      Tensor ith_anchor_yx = anchors[0].getBatchSlice(ith, 1);
      Tensor ith_anchor_hw = anchors[3].getBatchSlice(ith, 1);
      ith_anchor_yx.reshape({2});
      ith_anchor_hw.reshape({2});
      std::vector<float> ith_anchor_iou = calc_iou(anchors[0], anchors[3], ith_anchor_yx, ith_anchor_hw);
      for(unsigned int j = 0; j < ith_anchor_iou.size(); j++) {
        if (i == j) {continue;}
        if (ith_anchor_iou[j] > 0.7 && negative_mask[j]) {
          negative_mask[j] = 0;
          num_negative_anchors--;
        }
      }
    }
    std::cout << "refinedet forwarding 10" << std::endl;
    Tensor odm_yxhw = Tensor::cat({odm_yx_, odm_hw_}, 1);
    std::vector<unsigned int> pos_neg_mask(anchors_num);
    std::vector<unsigned int> gt_class_labels(anchors_num);
    for (unsigned int i = 0; i < anchors_num; i++) {
      pos_neg_mask[i] = (positive_mask[i] + negative_mask[i]) * odm_conf_.argmax()[i];
      if (anchor_gt_label_idx[i] == -1) {
        gt_class_labels[i] = 0;
      }
      else {
        gt_class_labels[i] = gt_class_.argmax()[anchor_gt_label_idx[i]];
      }
    }
    std::cout << "refinedet forwarding 1`" << std::endl;
    output.add_i(cross_entropy_with_mask(odm_conf_, pos_neg_mask, gt_class_labels) / num_positive_anchors);
    output.add_i(smooth_l1(odm_yxhw, gt_yxhw, pos_neg_mask) / num_positive_anchors);
  }
    std::cout << "refinedet forwarding 12" << std::endl;
    LossLayer::updateLoss(context, output);
    std::cout << "refinedet forwarding 13" << std::endl;
}

void binary_cross_entropy_derivative(const Tensor& derivative, Tensor& dp, Tensor& x, Tensor& y) {
  std::vector<unsigned int> label_idx = y.argmax();
  for (unsigned int a = 0; a < x.batch(); a++) {
    if (!label_idx[a]) {
      x.setValueInt(a * 2 + 1, x.getValue(a * 2 + 1) - 1);
    }
    else {
      x.setValueInt(a * 2, x.getValue(a * 2) - 1);
    }
  }
  dp = x.multiply(derivative);
}

void categorical_cross_entropy_derivative(const Tensor& derivative, Tensor& dc, Tensor& x, Tensor& y) {
  dc = x.subtract(y);
  dc.multiply_i(derivative);
}

void smooth_l1_derivative(const Tensor& derivative, Tensor& dx, Tensor& x, Tensor& y) {
  x.subtract_i(y);
  x.apply_i([](float val) {
      if (val < 0) {
          val = -val;
      }
      if (val < 1) {
          return val;
      } else {
          return 1.0f;
      }
  });
  dx = x.multiply(derivative);
}

void RefineDetLoss::calcDerivative(nntrainer::RunLayerContext &context) {
/// intended here to demonstrate that PowLayer::backwarding is being called
#ifdef DEBUG
  std::cout << "pow layer backward is called\n";
#endif
  // Get the incoming derivative
  const Tensor &incoming_derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &p = context.getInput(0);   // predicted binary confidence
  Tensor &x = context.getInput(1);   // predicted coordinates by ARM
  Tensor &c = context.getInput(2);   // predicted class confidence
  Tensor &t = context.getInput(3);   // predicted coordinates by ODM
  Tensor l, g;
  if (context.isLabelAvailable(0)) {
    Tensor &l = context.getLabel(0);   // ground truth label
  }
  if (context.isLabelAvailable(1)) {
    Tensor &g = context.getLabel(1);   // ground truth coordinates
  }

  // Get the outgoing derivatives
  Tensor &dp = context.getOutgoingDerivative(0);
  Tensor &dx = context.getOutgoingDerivative(1);
  Tensor &dc = context.getOutgoingDerivative(2);
  Tensor &dt = context.getOutgoingDerivative(3);

  for (unsigned int b = 0; b < dp.batch(); b++) {
    
    Tensor p_ = p.getBatchSlice(b, 1);  // (anchor, 2)
    Tensor x_ = x.getBatchSlice(b, 1);  // (anchor, 4)
    Tensor c_ = c.getBatchSlice(b, 1);  // (anchor, class)
    Tensor t_ = t.getBatchSlice(b, 1);  // (anchor, 4)
    Tensor l_ = l.getBatchSlice(b, 1);  // (anchor, class)
    Tensor g_ = g.getBatchSlice(b, 1);  // (anchor, 4)

    Tensor dp_ = dp.getBatchSlice(b, 1);  // (anchor, 2)
    Tensor dx_ = dx.getBatchSlice(b, 1);  // (anchor, 4)
    Tensor dc_ = dc.getBatchSlice(b, 1);  // (anchor, class)
    Tensor dt_ = dt.getBatchSlice(b, 1);  // (anchor, 4)
    binary_cross_entropy_derivative(incoming_derivative, dp, p_, l_);
    categorical_cross_entropy_derivative(incoming_derivative, dc, c_, l_);
    smooth_l1_derivative(incoming_derivative, dx, x_, g_);
    smooth_l1_derivative(incoming_derivative, dt, t_, g_);
  }

#ifdef DEBUG
  std::cout << "input: " << context.getOutput(SINGLE_INOUT_IDX);
  std::cout << "output: " << context.getInput(SINGLE_INOUT_IDX);
  /// PowUtil::pause();
#endif
}

#ifdef PLUGGABLE

nntrainer::Layer *create_pow_layer() {
  auto layer = new PowLayer();
  std::cout << "power created\n";
  return layer;
}

void destory_pow_layer(nntrainer::Layer *layer) {
  std::cout << "power deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_pow_layer,
                                                   destory_pow_layer};
}

#endif

} // namespace custom
