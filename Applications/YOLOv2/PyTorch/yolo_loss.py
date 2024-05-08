# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file yolo_loss.py
# @date 8 March 2023
# @brief Define loss class for yolo
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import torch
from torch import nn
import numpy as np


##
# @brief calculate iou between two boxes list
def calculate_iou(bbox1, bbox2):
    """
    @param bbox1 shape(numb_of_bbox, 4), it contains x, y, w, h
    @param bbox2 shape(numb_of_bbox, 4), it contains x, y, w, h
    @return result shape(numb_of_bbox, 1)
    """
    # bbox coordinates
    b1x1, b1y1 = (bbox1[:, :2]).split(1, 1)
    b1x2, b1y2 = (bbox1[:, :2] + (bbox1[:, 2:4])).split(1, 1)
    b2x1, b2y1 = (bbox2[:, :2]).split(1, 1)
    b2x2, b2y2 = (bbox2[:, :2] + (bbox2[:, 2:4])).split(1, 1)

    # box areas
    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)

    # intersections
    min_x_of_max_x, max_x_of_min_x = torch.min(b1x2, b2x2), torch.max(b1x1, b2x1)
    min_y_of_max_y, max_y_of_min_y = torch.min(b1y2, b2y2), torch.max(b1y1, b2y1)
    intersection_width = (min_x_of_max_x - max_x_of_min_x).clamp(min=0)
    intersection_height = (min_y_of_max_y - max_y_of_min_y).clamp(min=0)
    intersections = intersection_width * intersection_height

    # unions
    unions = (areas1 + areas2) - intersections

    result = intersections / unions
    return result


##
# @brief find best iou and its index
def find_best_ratio(anchors, bbox):
    """
    @param anchors shape(numb_of_anchors, 2), it contains w, h
    @param bbox shape(numb_of_bbox, 2), it contains w, h
    @return best_match index of best match, shape(numb_of_bbox, 1)
    """
    b1 = np.divide(anchors[:, 0], anchors[:, 1])
    b2 = np.divide(bbox[:, 0], bbox[:, 1])
    similarities = np.abs(b1.reshape(-1, 1) - b2)
    best_match = np.argmin(similarities, axis=0)
    return best_match


##
# @brief loss class for yolo
class YoloV2_LOSS(nn.Module):
    """Yolo v2 loss"""

    def __init__(self, num_classes, img_shape=(416, 416), outsize=(13, 13)):
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.outsize = outsize
        self.hook = {}

        self.anchors = torch.FloatTensor(
            [
                (1.3221, 1.73145),
                (3.19275, 4.00944),
                (5.05587, 8.09892),
                (9.47112, 4.84053),
                (11.2364, 10.0071),
            ]
        )

        self.mse = nn.MSELoss()
        self.bbox_loss, self.iou_loss, self.cls_loss = None, None, None

    ##
    # @brief function to track gradients of non-leaf varibles.
    def hook_variable(self, name, var):
        """Do not use this function when training. It is for debugging."""
        self.hook[name] = var
        self.hook[name].requires_grad_().retain_grad()

    ##
    # @brief function to print gradients of non-leaf varibles.
    def print_hook_variables(self):
        """Do not use this function when training. It is for debugging."""
        for k, var in self.hook.items():
            print(f"gradients of variable {k}:")
            batch, channel, height, width = var.grad.shape
            for b in range(batch):
                for c in range(channel):
                    for h in range(height):
                        for w in range(width):
                            if torch.abs(var.grad[b, c, h, w]).item() >= 1e-3:
                                print(
                                    f"(b: {b}, c: {c}, h: {h}, w: {w}) =\
                                          {var.grad[b, c, h, w]}"
                                )
            print("=" * 20)

    def forward(self, bbox_pred, iou_pred, prob_pred, bbox_gt, cls_gt):
        """
        @param bbox_pred shape(batch_size, cell_h x cell_w, num_anchors, 4)
        @param iou_pred shape(batch_size, cell_h x cell_w, 1)
        @param prob_pred shape(batch_size, cell_h x cell_w, num_anchors, num_classes)
        @param bbox_gt shape(batch_size, num_bbox, 4), data range(0~1)
        @param cls_gt shape(batch_size, num_bbox, 1)
        @return loss shape(1,)
        """
        self.hook_variable("bbox_pred", bbox_pred)
        bbox_pred = self.apply_anchors_to_bbox(bbox_pred)

        bbox_built, iou_built, cls_built, bbox_mask, iou_mask, cls_mask = (
            self._build_target(bbox_pred, bbox_gt, cls_gt)
        )

        self.bbox_loss = self.mse(bbox_pred * bbox_mask, bbox_built * bbox_mask)
        self.iou_loss = self.mse(iou_pred * iou_mask, iou_built * iou_mask)
        self.cls_loss = self.mse(prob_pred * cls_mask, cls_built * cls_mask)

        return self.bbox_loss * 5 + self.iou_loss + self.cls_loss

    def apply_anchors_to_bbox(self, bbox_pred):
        """
        @param bbox_pred shape(batch_size, cell_h x cell_w, num_anchors, 4)
        @return bbox_pred shape(batch_size, cell_h x cell_w, num_anchors, 4)
        """
        anchor_w = self.anchors[:, 0].contiguous().view(-1, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(-1, 1)
        bbox_pred_tmp = bbox_pred.clone()
        bbox_pred_tmp[:, :, :, 2:3] = torch.sqrt(bbox_pred[:, :, :, 2:3] * anchor_w)
        bbox_pred_tmp[:, :, :, 3:4] = torch.sqrt(bbox_pred[:, :, :, 3:4] * anchor_h)
        return bbox_pred_tmp

    def _build_target(self, bbox_pred, bbox_gt, cls_gt):
        """
        @param bbox_pred shape(batch_size, cell_h x cell_w, num_anchors, 4)
        @param bbox_gt shape(batch_size, num_bbox, 4)
        @param cls_gt shape(batch_size, num_bbox, 1)
        @return tuple of (bbox_built, iou_built, cls_built, bbox_mask, iou_mask, cls_mask)
        """
        bbox_built, bbox_mask = [], []
        iou_built, iou_mask = [], []
        cls_built, cls_mask = [], []

        batch_size = bbox_pred.shape[0]

        for i in range(batch_size):
            _bbox_built, _iou_built, _cls_built, _bbox_mask, _iou_mask, _cls_mask = (
                self._make_target_per_sample(
                    torch.FloatTensor(bbox_pred[i]),
                    torch.FloatTensor(np.array(bbox_gt[i])),
                    torch.LongTensor(cls_gt[i]),
                )
            )

            bbox_built.append(_bbox_built)
            bbox_mask.append(_bbox_mask)
            iou_built.append(_iou_built)
            iou_mask.append(_iou_mask)
            cls_built.append(_cls_built)
            cls_mask.append(_cls_mask)

        bbox_built = torch.stack(bbox_built)
        bbox_mask = torch.stack(bbox_mask)
        iou_built = torch.stack(iou_built)
        iou_mask = torch.stack(iou_mask)
        cls_built = torch.stack(cls_built)
        cls_mask = torch.stack(cls_mask)

        return bbox_built, iou_built, cls_built, bbox_mask, iou_mask, cls_mask

    def _make_target_per_sample(self, _bbox_pred, _bbox_gt, _cls_gt):
        """
        @param _bbox_pred shape(cell_h x cell_w, num_anchors, 4)
        @param _bbox_gt shape(num_bbox, 4)
        @param _cls_gt shape(num_bbox,)
        @return tuple of (_bbox_built, _iou_built, _cls_built, _bbox_mask, _iou_mask, _cls_mask)
        """
        hw, num_anchors, _ = _bbox_pred.shape

        # set result template
        _bbox_built = torch.zeros((hw, num_anchors, 4))
        _bbox_mask = torch.zeros((hw, num_anchors, 1))

        _iou_built = torch.zeros((hw, num_anchors, 1))
        _iou_mask = torch.ones((hw, num_anchors, 1)) * 0.5

        _cls_built = torch.zeros((hw, num_anchors, self.num_classes))
        _cls_mask = torch.zeros((hw, num_anchors, 1))

        # find best anchors
        _bbox_gt_wh = _bbox_gt.clone()[:, 2:]
        best_anchors = find_best_ratio(self.anchors, _bbox_gt_wh)

        # normalize x, y pos based on cell coornindates
        cx = _bbox_gt[:, 0] * self.outsize[0]
        cy = _bbox_gt[:, 1] * self.outsize[1]
        # calculate cell pos and normalize x, y
        cell_idx = np.floor(cy) * self.outsize[0] + np.floor(cx)
        cell_idx = np.array(cell_idx, dtype=np.int16)
        cx -= np.floor(cx)
        cy -= np.floor(cy)

        # set bbox of gt
        _bbox_built[cell_idx, best_anchors, 0] = cx
        _bbox_built[cell_idx, best_anchors, 1] = cy
        _bbox_built[cell_idx, best_anchors, 2] = torch.sqrt(_bbox_gt[:, 2])
        _bbox_built[cell_idx, best_anchors, 3] = torch.sqrt(_bbox_gt[:, 3])
        _bbox_mask[cell_idx, best_anchors, :] = 1

        # set cls of gt
        _cls_built[cell_idx, best_anchors, _cls_gt] = 1
        _cls_mask[cell_idx, best_anchors, :] = 1

        # set confidence score of gt
        _iou_built = calculate_iou(
            _bbox_pred.reshape(-1, 4), _bbox_built.view(-1, 4)
        ).detach()
        _iou_built = _iou_built.view(hw, num_anchors, 1)
        _iou_mask[cell_idx, best_anchors, :] = 1

        return _bbox_built, _iou_built, _cls_built, _bbox_mask, _iou_mask, _cls_mask
