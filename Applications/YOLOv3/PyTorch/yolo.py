#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file yolo.py
# @date 31 May 2023
# @brief Yolo v3 model
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import torch
import torch.nn as nn

import sys
import os

# get path of pyutils using relative path
def get_util_path():
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.abspath(os.path.dirname(current_path))
    target_path = os.path.abspath(os.path.dirname(parent_path))
    return os.path.dirname(target_path) + '/tools/pyutils/'

# add path of pyutils to sys.path
sys.path.append(get_util_path())
from torchconverter import save_bin

class ConvBlock(nn.Module):
    def __init__(self, num_channel_in, num_channel_out, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(num_channel_in, num_channel_out, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(num_channel_out, eps=1e-3)
        self.leaky_relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x


class DarkNetBlock(nn.Module):
    def __init__(self, num_channel, repeat=1):
        super(DarkNetBlock, self).__init__()
        self.repeat = repeat

        self.module_list = nn.ModuleList()
        for _ in range(self.repeat):
            self.module_list.add_module('block', nn.Sequential(
                ConvBlock(num_channel, num_channel // 2, 1, 1, 0),
                ConvBlock(num_channel // 2, num_channel, 3, 1, 1)
            ))

    def forward(self, x):        
        for module in self.module_list:
            out = module[0](x)
            out = module[1](out)
            x = x + out

        return x

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

        self.fe_modules = nn.Sequential()
        self.fe_modules.append(ConvBlock(3, 32, 3, 1, 1))
        self.fe_modules.append(ConvBlock(32, 64, 3, 2, 1))
        self.fe_modules.append(DarkNetBlock(64, 1))
        self.fe_modules.append(ConvBlock(64, 128, 3, 2, 1))
        self.fe_modules.append(DarkNetBlock(128, 2))
        self.fe_modules.append(ConvBlock(128, 256, 3, 2, 1))
        self.fe_modules.append(DarkNetBlock(256, 8))
        self.fe_modules.append(ConvBlock(256, 512, 3, 2, 1))
        self.fe_modules.append(DarkNetBlock(512, 8))
        self.fe_modules.append(ConvBlock(512, 1024, 3, 2, 1))
        self.fe_modules.append(DarkNetBlock(1024, 4))        

    def forward(self, x):
        for module in self.fe_modules:
            x = module(x)

        return x

if __name__ == '__main__':
    model = Darknet53()
    torch.save(model.state_dict(), './init_model.pt')
    save_bin(model, 'init_model')
