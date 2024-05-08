# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file main.py
# @date 8 March 2023
# @brief Implement training for yolo
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import sys
import os

from torchconverter import save_bin
import torch
from torch import optim
from torch.utils.data import DataLoader

from yolo import YoloV2
from yolo_loss import YoloV2_LOSS
from dataset import YOLODataset, collate_db


# get pyutils path using relative path
def get_util_path():
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.abspath(os.path.dirname(current_path))
    target_path = os.path.abspath(os.path.dirname(parent_path))
    return os.path.dirname(target_path) + "/tools/pyutils/"


# add pyutils path to sys.path
sys.path.append(get_util_path())

# set config
out_size = 13
num_classes = 4
num_anchors = 5

epochs = 3
batch_size = 4

train_img_dir = "/home/user/TRAIN_DIR/images/*"
train_ann_dir = "/home/user/TRAIN_DIR/annotations/*"
valid_img_dir = "/home/user/VALID_DIR/images/*"
valid_ann_dir = "/home/user/VALID_DIR/annotations/*"

# load data
train_dataset = YOLODataset(train_img_dir, train_ann_dir)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collate_db,
    shuffle=True,
    drop_last=True,
)
valid_dataset = YOLODataset(valid_img_dir, valid_ann_dir)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    collate_fn=collate_db,
    shuffle=False,
    drop_last=True,
)

# set model, loss and optimizer
model = YoloV2(num_classes=num_classes)
criterion = YoloV2_LOSS(num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# save init model
save_bin(model, "init_model")
torch.save(model.state_dict(), "./init_model.pt")

# train model
best_loss = 1e10
for epoch in range(epochs):
    epoch_train_loss = 0
    epoch_valid_loss = 0
    for idx, (img, bbox, cls) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        # model prediction
        hypothesis = model(img).permute((0, 2, 3, 1))
        hypothesis = hypothesis.reshape(
            (batch_size, out_size**2, num_anchors, 5 + num_classes)
        )
        # split each prediction(bbox, iou, class prob)
        bbox_pred_xy = torch.sigmoid(hypothesis[..., :2])
        bbox_pred_wh = torch.exp(hypothesis[..., 2:4])
        bbox_pred = torch.cat((bbox_pred_xy, bbox_pred_wh), 3)
        iou_pred = torch.sigmoid(hypothesis[..., 4:5])
        score_pred = hypothesis[..., 5:].contiguous()
        prob_pred = torch.softmax(score_pred.view(-1, num_classes), dim=1).view(
            score_pred.shape
        )
        # calc loss
        loss = criterion(
            torch.FloatTensor(bbox_pred),
            torch.FloatTensor(iou_pred),
            torch.FloatTensor(prob_pred),
            bbox,
            cls,
        )
        # back prop
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_train_loss += loss.item()

    for idx, (img, bbox, cls) in enumerate(valid_loader):
        model.eval()
        with torch.no_grad():
            # model prediction
            hypothesis = model(img).permute((0, 2, 3, 1))
            hypothesis = hypothesis.reshape(
                (hypothesis.shape[0], out_size**2, num_anchors, 5 + num_classes)
            )
            # split each prediction(bbox, iou, class prob)
            bbox_pred_xy = torch.sigmoid(hypothesis[..., :2])
            bbox_pred_wh = torch.exp(hypothesis[..., 2:4])
            bbox_pred = torch.cat((bbox_pred_xy, bbox_pred_wh), 3)
            iou_pred = torch.sigmoid(hypothesis[..., 4:5])
            score_pred = hypothesis[..., 5:].contiguous()
            prob_pred = torch.softmax(score_pred.view(-1, num_classes), dim=1).view(
                score_pred.shape
            )
            # calc loss
            loss = criterion(
                torch.FloatTensor(bbox_pred),
                torch.FloatTensor(iou_pred),
                torch.FloatTensor(prob_pred),
                bbox,
                cls,
            )
            epoch_valid_loss += loss.item()

    if epoch_valid_loss < best_loss:
        best_loss = epoch_valid_loss
        torch.save(model.state_dict(), "./best_model.pt")
        save_bin(model, "best_model")

    print(
        f"{epoch}epoch, train loss: {epoch_train_loss / len(train_loader):.4f},\
          valid loss: {epoch_valid_loss / len(valid_loader):.4f}"
    )

##
# @brief bbox post process function for inference


def post_process_for_bbox(bbox_p):
    """
    @param bbox_p shape(batch_size, cell_h x cell_w, num_anchors, 4)
    @return bbox_p shape(batch_size, cell_h x cell_w, num_anchors, 4)
    """
    anchors = torch.FloatTensor(
        [
            (1.3221, 1.73145),
            (3.19275, 4.00944),
            (5.05587, 8.09892),
            (9.47112, 4.84053),
            (11.2364, 10.0071),
        ]
    )

    outsize = (13, 13)
    width, height = outsize

    # restore cell pos to x, y
    for w in range(width):
        for h in range(height):
            bbox_p[:, height * h + w, :, 0] += w
            bbox_p[:, height * h + w, :, 1] += h
    bbox_p[:, :, :, :2] /= 13

    # apply anchors to w, h
    anchor_w = anchors[:, 0].contiguous().view(-1, 1)
    anchor_h = anchors[:, 1].contiguous().view(-1, 1)
    bbox_p[:, :, :, 2:3] *= anchor_w
    bbox_p[:, :, :, 3:4] *= anchor_h

    return bbox_p


# inference example using trained model
hypothesis = model(img).permute((0, 2, 3, 1))
hypothesis = hypothesis[0].reshape((1, out_size**2, num_anchors, 5 + num_classes))

# transform output
bbox_pred_xy = torch.sigmoid(hypothesis[..., :2])
bbox_pred_wh = torch.exp(hypothesis[..., 2:4])
bbox_pred = torch.cat((bbox_pred_xy, bbox_pred_wh), 3)
bbox_pred = post_process_for_bbox(bbox_pred)
iou_pred = torch.sigmoid(hypothesis[..., 4:5])
score_pred = hypothesis[..., 5:].contiguous()
prob_pred = torch.softmax(score_pred.view(-1, num_classes), dim=1).view(
    score_pred.shape
)

# result of inference (data range 0~1)
iou_mask = iou_pred > 0.5
print(bbox_pred * iou_mask, iou_pred * iou_mask, prob_pred * iou_mask)
