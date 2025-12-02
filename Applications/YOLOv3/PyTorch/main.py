#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file main.py
# @date 13 June 2023
# @brief Implement training for yolo v3
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import sys
import os

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from yolo import YoloV3
from yolo_loss import YoloV3_LOSS
from dataset import YOLODataset, collate_db

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# get pyutils path using relative path
def get_util_path():
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.abspath(os.path.dirname(current_path))
    target_path = os.path.abspath(os.path.dirname(parent_path))
    return os.path.dirname(target_path) + '/tools/pyutils/'

# add pyutils path to sys.path
sys.path.append(get_util_path())
from torchconverter import save_bin

# set config
out_size = (13, 26, 52)
num_classes = 4
num_anchors = 3
anchors = torch.FloatTensor([
            [[116, 90], [156, 198], [373, 326]], # large
            [[30, 61], [62, 45], [59,119]], # medium
            [[10, 13], [16, 30], [33,23]] # small
        ]) / 416

epochs = 1000
batch_size = 8

train_img_dir = '/home/user/train_dir/images/*'
train_ann_dir = '/home/user/train_dir/annotations/*'
valid_img_dir = '/home/user/valid_dir/images/*'
valid_ann_dir = '/home/user/valid_dir/annotations/*'

# load data
train_dataset = YOLODataset(train_img_dir, train_ann_dir, normalize=416)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_db, shuffle=True, drop_last=True)
valid_dataset = YOLODataset(valid_img_dir, valid_ann_dir, normalize=416)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_db, shuffle=False, drop_last=False)

# set model, loss and optimizer
model = YoloV3(num_classes, 'darknet53.conv.74').to(device)
# model.load_state_dict(torch.load('./best_model.pt'))

criterion = YoloV3_LOSS(num_classes=num_classes, anchors=anchors)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# train model
best_loss = 1e+10
for epoch in range(epochs):
    epoch_train_loss = 0
    epoch_valid_loss = 0
    for idx, (img, y_bbox, y_cls) in enumerate(train_loader):        
        loss = 0
        model.train()
        optimizer.zero_grad()

        # model prediction
        pred_list = model(img.to(device))
        for i, hypothesis in enumerate(pred_list):
            hypothesis = hypothesis.permute((0, 2, 3, 1)).to('cpu')
            hypothesis = hypothesis.reshape((batch_size, out_size[i]**2, num_anchors, 5+num_classes))
            loss += criterion(hypothesis, y_bbox, y_cls, scale=i)

        # back prop
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_train_loss += loss.item()
        print("epoch: {}, iter: {}/{}, loss: {:.4f}".format(epoch, idx, len(train_loader), loss.item()))

    for idx, (img, y_bbox, y_cls) in enumerate(valid_loader):
        loss = 0
        model.eval()
        with torch.no_grad():
            # model prediction
            pred_list = model(img.to(device))
            for i, hypothesis in enumerate(pred_list):
                hypothesis = hypothesis.permute((0, 2, 3, 1)).to('cpu')
                hypothesis = hypothesis.reshape((hypothesis.shape[0], out_size[i]**2, num_anchors, 5+num_classes))
                loss += criterion(hypothesis, y_bbox, y_cls, scale=i)
        epoch_valid_loss += loss.item()
        
    if epoch_valid_loss < best_loss:
        best_loss = epoch_valid_loss
        torch.save(model.state_dict(), './best_model.pt')
        save_bin(model, 'best_model')
        
    print("{}epoch, train loss: {:.4f}, valid loss: {:.4f}".format(
        epoch, epoch_train_loss / len(train_loader), epoch_valid_loss / len(valid_loader)))

##
# @brief bbox post process function for inference
def post_process_for_bbox(hypothesis, scale):
    """
    @param hypothesis shape(batch_size, cell_h x cell_w, num_anchors, 5+num_classes)
    @param scale 0: large, 1: medium, 2: small
    @return bbox_pred shape(batch_size, cell_h x cell_w, num_anchors, 4)    
    """
    # make bbox_post
    anchor_w = anchors[scale][:, 0].contiguous().view(-1, 1)
    anchor_h = anchors[scale][:, 1].contiguous().view(-1, 1)
    bbox_pred_xy = torch.sigmoid(hypothesis[..., :2])
    bbox_pred_wh = torch.exp(hypothesis[..., 2:4])
    bbox_pred = torch.cat((bbox_pred_xy, bbox_pred_wh), 3)        
    bbox_pred[:, :, :, 2:3] *= anchor_w
    bbox_pred[:, :, :, 3:4] *= anchor_h 
        
    # restore cell pos to x, y
    width = height = out_size[scale]
    for w in range(width):
        for h in range(height):
            bbox_pred[:, height*h + w, :, 0] += w
            bbox_pred[:, height*h + w, :, 1] += h
    bbox_pred[:, :, :, :2] /= out_size[scale]    

    return bbox_pred

##
# @brief visualize function for inference
def visualize(img, bbox_preds):
    # set img
    img_array = (img * 255).permute((1, 2, 0)).numpy().astype(np.uint8)    
    img = Image.fromarray(img_array)
    
    for bbox_pred in bbox_preds:
        if torch.sum(bbox_pred) == 0: continue
        print(bbox_pred)
        # set bbox
        bbox_pred = [int(x * 416) for x in bbox_pred]
        
        x_lefttop = bbox_pred[0]
        y_lefttop = bbox_pred[1]
        width = bbox_pred[2]
        height = bbox_pred[3]

        # draw bb
        draw = ImageDraw.Draw(img)
        draw.rectangle([(x_lefttop, y_lefttop), (x_lefttop+width, y_lefttop+height)])
    
    # show img
    plt.imshow(img)
    plt.show()    

# visualize trained result
test_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_db, shuffle=False, drop_last=False)
for img, _, _ in test_loader:            
    model.eval()
    pred_list = model(img.to(device))

    bbox_stack = None
    for scale, hypothesis in enumerate(pred_list):
        hypothesis = hypothesis.permute((0, 2, 3, 1)).to('cpu')
        hypothesis = hypothesis.reshape((1, out_size[scale]**2, num_anchors, 5+num_classes))        
        bbox_pred = post_process_for_bbox(hypothesis, scale)
        iou_pred = torch.sigmoid(hypothesis[..., 4:5])
        prob_pred = torch.sigmoid(hypothesis[..., 5:].contiguous())
        
        # stack bbox for visualization
        iou_mask = (iou_pred > 0.5)    
        bbox_pred = bbox_pred * iou_mask
        if scale == 0:
            bbox_stack = bbox_pred.reshape(-1, 4)
        else:
            bbox_stack = torch.cat((bbox_stack, bbox_pred.reshape(-1, 4)), dim=0)
    
    # visualize
    if torch.sum(bbox_stack) > 1e-2:
        visualize(img[0], bbox_stack)
