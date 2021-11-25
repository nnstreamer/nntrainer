#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
#
# @file genModelTests_v2.py
# @date 25 November 2021
# @brief Generate model tcs
# @author Parichay Kapoor <pk.kapoor@samsung.com>

from recorder_v2 import record_v2, inspect_file
import torch

class ReduceMeanLast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 7)
        self.loss = torch.nn.Identity()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = torch.mean(out, dim=-1)
        loss = self.loss(torch.sum(out))
        return out, loss

if __name__ == "__main__":
    record_v2(
        ReduceMeanLast(),
        iteration=2,
        input_dims=[(3, 2,)],
        label_dims=[(3, 1,)],
        name="reduce_mean_last",
    )

    # inspect_file("lstm_single.nnmodelgolden")
