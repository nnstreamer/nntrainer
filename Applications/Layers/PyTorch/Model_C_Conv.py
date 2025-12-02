# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
#
# @file   main.cpp
# @date   01 Feb 2023
# @see    https://github.com/nnstreamer/nntrainer
# @author Donghak Park <donghak.park@samsung.com>
# @bug	  No known bugs except for NYI items
# @brief  This is Model_C_Conv Example for Pytorch (with Dummy Dataset)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cpu"
print(f"Using {device} device")
print(f"torch version: {torch.__version__}")

EPOCH = 100
BATCH_SIZE = 64
IMG_SIZE = [3, 224, 224]
OUTPUT_SIZE = [1, 1, 37632]


class CustomDatset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = torch.rand((64, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
        self.y_data = torch.rand((64, OUTPUT_SIZE[2]))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.NN_stack = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Flatten()
        )

    def forward(self, x):
        logits = self.NN_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCH):
        print(f"\nEPOCH {t+1}\n-------------------------------")
        dataset = CustomDatset()
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        train(dataloader, model, loss_fn, optimizer)
    print("Training Finished!")
