# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file dataset.py
# @date 8 March 2023
# @brief Define dataset class for yolo
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import glob
import re
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
        self.img_dir = img_dir
        pattern = re.compile("\/(\d+)\.")
        img_list = glob.glob(img_dir + "*")
        ann_list = glob.glob(ann_dir + "*")

        img_ids = list(map(lambda x: pattern.search(x).group(1), img_list))
        ann_ids = list(map(lambda x: pattern.search(x).group(1), ann_list))
        ids_list = list(set(img_ids) & set(ann_ids))

        self.ids_list = []
        self.bbox_gt = []
        self.cls_gt = []

        for ids in ids_list:
            label_bbox = []
            label_cls = []
            with open(ann_dir + ids + ".txt", "rt", encoding="utf-8") as f:
                for line in f.readlines():
                    line = [float(i) for i in line.split()]
                    label_bbox.append(np.array(line[1:], dtype=np.float32) / 416)
                    label_cls.append(int(line[0]))

            if len(label_cls) == 0:
                continue

            self.ids_list.append(ids)
            self.bbox_gt.append(label_bbox)
            self.cls_gt.append(label_cls)

        self.length = len(self.ids_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = (
            torch.FloatTensor(
                np.array(
                    Image.open(self.img_dir + self.ids_list[idx] + ".jpg").resize(
                        (416, 416)
                    )
                )
            ).permute((2, 0, 1))
            / 255
        )
        return img, self.bbox_gt[idx], self.cls_gt[idx]


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
