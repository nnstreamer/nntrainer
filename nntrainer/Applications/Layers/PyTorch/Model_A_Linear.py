# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
#
# @file   main.cpp
# @date   01 Feb 2023
# @see    https://github.com/nnstreamer/nntrainer
# @author Donghak Park <donghak.park@samsung.com>
# @bug	  No known bugs except for NYI items
# @brief  This is Model_A_Linear Example for Pytorch (with Dummy Dataset)

import torch
import torch.nn.utils as torch_utils
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import random
import numpy as np

seed = 0
torch.manual_seed(seed)

device = "cpu"
print(f"Using {device} device")
print(f"torch version: {torch.__version__}")

EPOCH = 10
BATCH_SIZE = 2048
IMG_SIZE = 784
OUTPUT_SIZE = 100


class CustomDatset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = torch.rand((BATCH_SIZE, 1, 1, IMG_SIZE))
        self.y_data = torch.rand((BATCH_SIZE, 1, 1, OUTPUT_SIZE))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden_layer1 = nn.Linear(IMG_SIZE, 4096)
        self.hidden_layer2 = nn.Linear(4096, 2048)
        self.output_layer = nn.Linear(2048, OUTPUT_SIZE)

    def forward(self, x):
        out = self.hidden_layer1(x)
        out = self.hidden_layer2(out)
        out = self.output_layer(out)

        return out


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        max_norm = 1.0
        optimizer.zero_grad()
        loss.backward()
        torch_utils.clip_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    print(model)

    hidden_layer1_trainable = True
    hidden_layer2_trainable = True

    for i, param in enumerate(model.hidden_layer1.parameters()):
        if i == 0 or i == 1:
            param.requires_grad = hidden_layer1_trainable

    for i, param in enumerate(model.hidden_layer2.parameters()):
        if i == 0 or i == 1:
            param.requires_grad = hidden_layer2_trainable

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCH):
        print(f"\nEPOCH {t+1}\n-------------------------------")
        dataset = CustomDatset()
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        train(dataloader, model, loss_fn, optimizer)
    print("Training Finished!")
