# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file yolo.py
# @date 8 March 2023
# @brief Define simple yolo model, but not original darknet.
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import torch
import torch.nn as nn

##
# @brief define simple yolo model (not original darknet)
class YoloV2_light(nn.Module): 
    def __init__(self, 
                 num_classes,
                 anchors=\
                 [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]):
        
        super(YoloV2_light, self).__init__()              
        self.num_classes = num_classes
        self.anchors = anchors
        self.stage1_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32, eps=1e-3),
                                          nn.LeakyReLU(), nn.MaxPool2d(2, 2))
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64, eps=1e-3),
                                          nn.LeakyReLU(), nn.MaxPool2d(2, 2))
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128, eps=1e-3),
                                          nn.LeakyReLU())
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0), nn.BatchNorm2d(64, eps=1e-3),
                                          nn.LeakyReLU())
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128, eps=1e-3),
                                          nn.LeakyReLU(), nn.MaxPool2d(2, 2))
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256, eps=1e-3),
                                          nn.LeakyReLU())
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0), nn.BatchNorm2d(128, eps=1e-3),
                                          nn.LeakyReLU())
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256, eps=1e-3),
                                          nn.LeakyReLU(), nn.MaxPool2d(2, 2))
        self.stage1_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512, eps=1e-3),
                                          nn.LeakyReLU())
        self.stage1_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0), nn.BatchNorm2d(256, eps=1e-3),
                                           nn.LeakyReLU(), nn.MaxPool2d(2, 2))
        self.out_conv = nn.Conv2d(256, len(self.anchors) * (5 + num_classes), 1, 1, 0)

    def forward(self, input):
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.out_conv(output)
        return output
