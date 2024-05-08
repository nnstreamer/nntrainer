# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file yolo.py
# @date 8 March 2023
# @brief Define simple yolo model, but not original darknet.
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import torch
from torch import nn


##
# @brief define yolo model (except for re-organization module)
class YoloV2(nn.Module):
    def __init__(self, num_classes, num_anchors=5):

        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64, eps=1e-3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128, eps=1e-3), nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0), nn.BatchNorm2d(64, eps=1e-3), nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, eps=1e-3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256, eps=1e-3), nn.LeakyReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0), nn.BatchNorm2d(128, eps=1e-3), nn.LeakyReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512, eps=1e-3), nn.LeakyReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0), nn.BatchNorm2d(256, eps=1e-3), nn.LeakyReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512, eps=1e-3), nn.LeakyReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0), nn.BatchNorm2d(256, eps=1e-3), nn.LeakyReLU()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512, eps=1e-3), nn.LeakyReLU()
        )

        self.conv_b = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, 0), nn.BatchNorm2d(64, eps=1e-3), nn.LeakyReLU()
        )

        self.maxpool_a = nn.MaxPool2d(2, 2)
        self.conv_a1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024, eps=1e-3),
            nn.LeakyReLU(),
        )
        self.conv_a2 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0), nn.BatchNorm2d(512, eps=1e-3), nn.LeakyReLU()
        )
        self.conv_a3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024, eps=1e-3),
            nn.LeakyReLU(),
        )
        self.conv_a4 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0), nn.BatchNorm2d(512, eps=1e-3), nn.LeakyReLU()
        )
        self.conv_a5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024, eps=1e-3),
            nn.LeakyReLU(),
        )
        self.conv_a6 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024, eps=1e-3),
            nn.LeakyReLU(),
        )
        self.conv_a7 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024, eps=1e-3),
            nn.LeakyReLU(),
        )

        self.conv_out1 = nn.Sequential(
            nn.Conv2d(1280, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024, eps=1e-3),
            nn.LeakyReLU(),
        )

        self.conv_out2 = nn.Conv2d(1024, self.num_anchors * (5 + num_classes), 1, 1, 0)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.conv9(output)
        output = self.conv10(output)
        output = self.conv11(output)
        output = self.conv12(output)
        output = self.conv13(output)

        output_a = self.maxpool_a(output)
        output_a = self.conv_a1(output_a)
        output_a = self.conv_a2(output_a)
        output_a = self.conv_a3(output_a)
        output_a = self.conv_a4(output_a)
        output_a = self.conv_a5(output_a)
        output_a = self.conv_a6(output_a)
        output_a = self.conv_a7(output_a)

        output_b = self.conv_b(output)
        b, c, h, w = output_b.size()
        output_b = output_b.view(b, int(c / 4), h, 2, w, 2).contiguous()
        output_b = output_b.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_b = output_b.view(b, -1, int(h / 2), int(w / 2))

        output = torch.cat((output_a, output_b), 1)
        output = self.conv_out1(output)
        output = self.conv_out2(output)
        return output
