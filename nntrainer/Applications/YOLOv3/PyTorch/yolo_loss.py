# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file yolo_loss.py
# @date 12 June 2023
# @brief Define loss class for yolo v3
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import torch
import torch.nn as nn
import torch.functional as F
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
    
    # result
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
# @brief loss class for yolo v3
class YoloV3_LOSS(nn.Module):
    """Yolo v3 loss"""
    def __init__(self, num_classes, anchors, img_shape = (416, 416), outsize = (13, 26, 52)):
        
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.outsize = outsize        
        self.anchors = anchors
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()        

    def forward(self, hypothesis, bbox_gt, cls_gt, scale, debug=False):        
        """
        @param hypothesis shape(batch_size, out_size**2, num_anchors, 5 + num_classes)
        @param bbox_gt shape(batch_size, num_bbox, 4), data range(0~1)
        @param cls_gt shape(batch_size, num_bbox, 1)
        @param scale 0: large, 1: medium, 2: small
        @return loss shape(1,)
        """
        bbox_loss, iou_loss, cls_loss = 0, 0, 0

        # split each prediction(bbox, iou, class prob)
        anchor_w = self.anchors[scale][:, 0].contiguous().view(-1, 1)
        anchor_h = self.anchors[scale][:, 1].contiguous().view(-1, 1)

        bbox_pred_xy = torch.sigmoid(hypothesis[..., :2])
        bbox_pred_wh = torch.exp(hypothesis[..., 2:4])
        bbox_pred = torch.cat((bbox_pred_xy, bbox_pred_wh), 3)        
        bbox_pred[:, :, :, 2:3] *= anchor_w
        bbox_pred[:, :, :, 3:4] *= anchor_h 

        iou_pred = torch.sigmoid(hypothesis[..., 4:5])
        prob_pred = torch.sigmoid(hypothesis[..., 5:])                
        
        # build target
        bbox_built, iou_built, cls_built, bbox_mask, iou_mask, cls_mask =\
            self._build_target(bbox_pred, bbox_gt, cls_gt, scale)        
        
        # for debugging
        if debug:
            return bbox_built
        
        # calculate loss
        bbox_loss = self.mse(bbox_pred * bbox_mask,
                                        bbox_built * bbox_mask)   
        iou_loss = self.mse(iou_pred * iou_mask,
                                    iou_built * iou_mask)
        cls_loss = self.mse(prob_pred * cls_mask,
                                    cls_built * cls_mask)
        # print('loss of iter: {:.4f}, {:.4f}, {:.4f}'.format(bbox_loss.item()*5, iou_loss.item(), cls_loss.item()*0.5))
        return bbox_loss * 5 + iou_loss + cls_loss * 0.5

    def _build_target(self, bbox_pred, bbox_gt, cls_gt, scale):
        """
        @param bbox_pred list of tensor (batch_size, cell_h x cell_w, num_anchors, 4)
        @param bbox_gt shape(batch_size, num_bbox, 4)
        @param cls_gt shape(batch_size, num_bbox, 1)
        @param scale 0: large, 1: medium, 2: small
        @return tuple of (bbox_built, iou_built, cls_built, bbox_mask, iou_mask, cls_mask)
        """    
        bbox_built, bbox_mask = [], []
        iou_built, iou_mask = [], []
        cls_built, cls_mask = [], []
        
        batch_size = bbox_pred.shape[0]
                
        for i in range(batch_size):
            _bbox_built, _iou_built, _cls_built,\
                _bbox_mask, _iou_mask, _cls_mask =\
                    self._make_target_per_sample(
                        torch.FloatTensor(bbox_pred[i]),
                        torch.FloatTensor(np.array(bbox_gt[i])),
                        torch.LongTensor(cls_gt[i]),
                        scale
                    )
            
            bbox_built.append(_bbox_built.numpy())
            bbox_mask.append(_bbox_mask.numpy())
            iou_built.append(_iou_built.numpy())
            iou_mask.append(_iou_mask.numpy())
            cls_built.append(_cls_built.numpy())
            cls_mask.append(_cls_mask.numpy())

        bbox_built, bbox_mask, iou_built, iou_mask, cls_built, cls_mask =\
            torch.FloatTensor(np.array(bbox_built)),\
            torch.FloatTensor(np.array(bbox_mask)),\
            torch.FloatTensor(np.array(iou_built)),\
            torch.FloatTensor(np.array(iou_mask)),\
            torch.FloatTensor(np.array(cls_built)),\
            torch.FloatTensor(np.array(cls_mask))
                    
        return bbox_built, iou_built, cls_built, bbox_mask, iou_mask, cls_mask
    def _make_target_per_sample(self, _bbox_pred, _bbox_gt, _cls_gt, scale):
        """
        @param _bbox_pred shape(cell_h x cell_w, num_anchors, 4)
        @param _bbox_gt shape(num_bbox, 4)
        @param _cls_gt shape(num_bbox,)
        @param scale 0: large, 1: medium, 2: small
        @return tuple of (_bbox_built, _iou_built, _cls_built, _bbox_mask, _iou_mask, _cls_mask)
        """
        hw, num_anchors, _  = _bbox_pred.shape
        
        # set result template
        _bbox_built = torch.zeros((hw, num_anchors, 4))
        _bbox_mask = torch.zeros((hw, num_anchors, 1))
        
        _iou_built = torch.zeros((hw, num_anchors, 1))
        _iou_mask = torch.ones((hw, num_anchors, 1)) * 0.5
        
        _cls_built = torch.zeros((hw, num_anchors, self.num_classes))
        _cls_mask = torch.zeros((hw, num_anchors, 1))
        
        if len(_bbox_gt) == 0:
            return _bbox_built, _iou_built, _cls_built, _bbox_mask, _iou_mask, _cls_mask
        
        # find best anchors        
        _bbox_gt_wh = _bbox_gt.clone()[:, 2:]        
        best_anchors = find_best_ratio(self.anchors[scale], _bbox_gt_wh)              

        # normalize x, y pos based on cell coornindates
        cx = _bbox_gt[:, 0] * self.outsize[scale]
        cy = _bbox_gt[:, 1] * self.outsize[scale]

        # calculate cell pos and normalize x, y
        cell_idx = np.floor(cy) * self.outsize[scale] + np.floor(cx)
        cell_idx = np.array(cell_idx, dtype=np.int16)
        cx -= np.floor(cx)
        cy -= np.floor(cy)
                
        # set bbox of gt        
        _bbox_built[cell_idx, best_anchors, 0] = cx 
        _bbox_built[cell_idx, best_anchors, 1] = cy
        _bbox_built[cell_idx, best_anchors, 2] = _bbox_gt[:, 2]
        _bbox_built[cell_idx, best_anchors, 3] = _bbox_gt[:, 3]        
        _bbox_mask[cell_idx, best_anchors, :] = 1

        # set cls of gt       
        _cls_built[cell_idx, best_anchors, _cls_gt] = 1
        _cls_mask[cell_idx, best_anchors, :] = 1
        
        # set confidence score of gt
        _iou_built = calculate_iou(_bbox_pred.reshape(-1, 4), _bbox_built.view(-1, 4)).detach()    
        _iou_built = _iou_built.view(hw, num_anchors, 1)        
        _iou_mask[cell_idx, best_anchors, :] = 1
        
        return _bbox_built, _iou_built, _cls_built,\
                _bbox_mask, _iou_mask, _cls_mask  
