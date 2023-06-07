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
import numpy as np

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

        self.conv = nn.Conv2d(num_channel_in, num_channel_out, kernel_size, stride, padding, bias=False)
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

    def load_weights_of_convblock(self, block, weights, pointer):
        # load bias of batch norm
        num_weights = block.bn.bias.numel()
        bn_bias = torch.from_numpy(
            weights[pointer: pointer + num_weights]).view_as(block.bn.bias)
        block.bn.bias.data.copy_(bn_bias)
        pointer += num_weights
        # load weight of batch norm
        bn_weights = torch.from_numpy(
            weights[pointer: pointer + num_weights]).view_as(block.bn.weight)
        block.bn.weight.data.copy_(bn_weights)
        pointer += num_weights
        # load running mean of batch norm
        bn_running_mean = torch.from_numpy(
            weights[pointer: pointer + num_weights]).view_as(block.bn.running_mean)
        block.bn.running_mean.data.copy_(bn_running_mean)
        pointer += num_weights
        # load running var of batch norm
        bn_running_var = torch.from_numpy(
            weights[pointer: pointer + num_weights]).view_as(block.bn.running_var)
        block.bn.running_var.data.copy_(bn_running_var)
        pointer += num_weights
        
        # load weight of convolutional layer
        num_weights = block.conv.weight.numel()
        conv_weights = torch.from_numpy(
            weights[pointer: pointer + num_weights]).view_as(block.conv.weight)
        block.conv.weight.data.copy_(conv_weights)
        pointer += num_weights
        
        return pointer
        
    def load_pretrained_weights(self, path):
        with open(path, "rb") as f:
            # read header info (useless in this func)
            _ = np.fromfile(f, dtype=np.int32, count=5)
            # read weights
            weights = np.fromfile(f, dtype=np.float32)

        pointer = 0
        for module in self.module_list:            
            if isinstance(module, ConvBlock):    
                pointer = self.load_weights_of_convblock(module, weights, pointer)
            elif isinstance(module, DarkNetBlock):
                for res_block in module.module_list:
                    for conv_block in res_block:
                        pointer = self.load_weights_of_convblock(conv_block, weights, pointer)


if __name__ == '__main__':
    model = Darknet53()
    model.load_pretrained_weights('darknet53.conv.74')
    save_bin(model, 'pre_trained_darknet53.bin')
