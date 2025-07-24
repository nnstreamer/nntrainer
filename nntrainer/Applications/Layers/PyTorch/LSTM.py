# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file   main.cpp
# @date   30 Jan 2023
# @see    https://github.com/nnstreamer/nntrainer
# @author Seungbaek Hong <sb92.hong@samsung.com>
# @bug	  No known bugs except for NYI items
# @brief  This is LSTM Example for PyTorch (only training with dummy data)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch version: {torch.__version__}")

seed = 0
torch.manual_seed(seed)
torch.set_num_threads(1)

EPOCH = 10
DB_SIZE = 64
BATCH_SIZE = 64
IMG_SIZE = 224*224*3
OUTPUT_SIZE = 10


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()        
        self.lstm = nn.LSTM(IMG_SIZE, OUTPUT_SIZE, batch_first=True)
        
    def forward(self, x):                
        output, (_, _) = self.lstm(x)
        return output


def train(dataloader, model, loss_fn, optimizer):   
    epoch_loss, num_of_batch = 0, len(dataloader)

    model.train()
    for X_batch, y_batch in dataloader:
        # Compute prediction error
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / num_of_batch

    return epoch_loss


def make_dummy_database(num_of_samples):
    X = torch.randn((num_of_samples, 1, IMG_SIZE))
    y = torch.randn((num_of_samples, 1, OUTPUT_SIZE))
    return X, y

if __name__ == "__main__":
    model = LSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(EPOCH):
        X_train, y_train = make_dummy_database(DB_SIZE)        
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        epoch_loss = train(dataloader, model, loss_fn, optimizer)
        print(f"loss: {epoch_loss:>7f}  [{epoch+1:>5d}/{EPOCH}]")
        del X_train, y_train, dataset, dataloader, epoch_loss
