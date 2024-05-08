# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file dataset.py
# @date 8 March 2023
# @brief Define dataset class for yolo
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image


##
# @brief dataset class for yolo
# @note Need annotation text files corresponding to the name of the images.
class YOLODataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        super().__init__()
        img_list = glob.glob(img_dir)
        ann_list = glob.glob(ann_dir)
        img_list.sort()
        ann_list.sort()

        self.length = len(img_list)
        self.input_images = []
        self.bbox_gt = []
        self.cls_gt = []

        for i in range(len(img_list)):
            img = np.array(Image.open(img_list[i]).resize((416, 416))) / 255
            label_bbox = []
            label_cls = []
            with open(ann_list[i], "rt", encoding="utf-8") as f:
                for line in f.readlines():
                    line = [float(i) for i in line.split()]
                    label_bbox.append(np.array(line[1:], dtype=np.float32) / 416)
                    label_cls.append(int(line[0]))

            self.input_images.append(img)
            self.bbox_gt.append(label_bbox)
            self.cls_gt.append(label_cls)

        self.input_images = np.array(self.input_images)
        self.input_images = torch.FloatTensor(self.input_images).permute((0, 3, 1, 2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.input_images[idx], self.bbox_gt[idx], self.cls_gt[idx]


##
# @brief collate db function for yolo
def collate_db(batch):
    """
    @param batch list of batch, (img, bbox, cls)
    @return collated list of batch, (img, bbox, cls)
    """
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    items[2] = list(items[2])
    return items
