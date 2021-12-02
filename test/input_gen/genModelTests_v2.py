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

class MolAttention(torch.nn.Module):
    def __init__(self, query_size):
        super(MolAttention, self).__init__()
        self.query_size = query_size
        self.units = 8
        self.K = 5 # number of mixtures
        self.dense1 = torch.nn.Linear(self.query_size, self.units)
        self.dense2 = torch.nn.Linear(self.units, 3 * self.K, bias=False)
        self.loss = torch.nn.Identity()

    def forward(self, inputs, labels):
        if len(inputs) == 4:
            query, values, attention_state, mask_len = inputs
        else:
            query, values, attention_state = inputs
            mask_len = None
        batch_size, timesteps, _ = values.size()

        dense1_out = torch.tanh(self.dense1(query.unsqueeze(1)))
        mlp_proj_out = self.dense2(dense1_out)
        kappa, beta, alpha = mlp_proj_out.chunk(chunks=3, dim=2)

        kappa = torch.exp(kappa)
        beta = torch.exp(beta)
        alpha = torch.softmax(alpha, dim=2)
        kappa = kappa + attention_state

        # Timesteps const array
        j = torch.arange(start=1, end=timesteps + 1).view(1, -1, 1).expand(batch_size, -1, self.K)

        integrals_left = torch.sigmoid(torch.div(j + 0.5 - kappa, beta + 1e-8))
        integrals_right = torch.sigmoid(torch.div(j - 0.5 - kappa, beta + 1e-8))
        integrals = alpha * (integrals_left - integrals_right)
        scores = torch.sum(integrals, dim=2)

        if mask_len is not None:
            max_len = max(int(mask_len.max()), scores.shape[1])
            mask = torch.arange(0, max_len)\
                    .type_as(mask_len)\
                    .unsqueeze(0).expand(mask_len.numel(), max_len)\
                    .lt(mask_len.unsqueeze(1))
            scores.masked_fill_(torch.logical_not(mask), 0.)

        output = torch.matmul(scores.unsqueeze(1), values).squeeze(dim=1)

        loss = self.loss(torch.sum(output))

        return output, loss

if __name__ == "__main__":
    record_v2(
        ReduceMeanLast(),
        iteration=2,
        input_dims=[(3, 2,)],
        label_dims=[(3, 1,)],
        name="reduce_mean_last",
    )

    record_v2(
        MolAttention(query_size=6),
        iteration=2,
        input_dims=[(3,6), (3,4,6), (3,1,5), (3)],
        input_dtype=[float, float, float, int],
        label_dims=[(3,1,6)],
        name="mol_attention_masked",
    )

    record_v2(
        MolAttention(query_size=6),
        iteration=2,
        input_dims=[(3,6), (3,4,6), (3,1,5)],
        input_dtype=[float, float, float],
        label_dims=[(3,1,6)],
        name="mol_attention",
    )

    # inspect_file("mol_attention_masked.nnmodelgolden")
