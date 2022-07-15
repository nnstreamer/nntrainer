#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
#
# @file genModelTests_v2.py
# @date 25 November 2021
# @brief Generate model tcs
# @author Parichay Kapoor <pk.kapoor@samsung.com>

from recorder_v2 import record_v2, inspect_file, _rand_like
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

        loss = self.loss(torch.sum(output)) + self.loss(torch.sum(kappa))

        return (output, kappa), loss

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, need_weights=True, provide_attention_mask=False):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_attention = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first=True)
        self.loss = torch.nn.MSELoss()
        self.need_weights = need_weights
        self.provide_attention_mask = provide_attention_mask

    def forward(self, inputs, labels):
        inputs, attn_mask = (inputs[:-1], inputs[-1]) if self.provide_attention_mask else (inputs, None)
        query, *left = inputs
        if len(left) == 0:
            key = value = query
        else:
            key, value = left

        output, attention_weight = self.multi_head_attention(query, key, value, need_weights=self.need_weights, attn_mask=attn_mask)
        loss = self.loss(output, labels[0])
        if attention_weight is not None:
            output = [output, attention_weight]

        return output, loss

    def input_label_reader(input_dims, label_dims, input_dtype):
        query_dim, key_dim, value_dim, *left_dim = input_dims
        query_dtype, key_dtype, value_dtype, *left_dtype = input_dtype
        assert(query_dtype == key_dtype == value_dtype)
        if left_dim != []:
            mask_dim = left_dim[0]
            mask_dtype = left_dtype[0]
            if mask_dtype == bool:
                # Since nntrainer does not support bool type tensor yet, convert mask to float type
                # todo: return bool type mask tensor
                mask = torch.randn(mask_dim) > 0.5
                new_attn_mask = torch.zeros_like(mask, dtype=torch.float32)
                new_attn_mask.masked_fill_(mask, float("-inf"))
                mask = [new_attn_mask]
            elif mask_dtype == int:
                mask = [torch.randint(0, 1, mask_dim, torch.int32)]
            else:
                mask = _rand_like([mask_dim], -1e9, mask_dtype)
        else:
            mask = []
        inputs = _rand_like([query_dim, key_dim, value_dim], dtype=input_dtype if input_dtype is not None else float) + mask
        labels = _rand_like(label_dims, dtype=float)
        return inputs, labels

class FCRelu(torch.nn.Module):
    def __init__(self, decay=False):
        super().__init__()
        self.fc = torch.nn.Linear(3, 10)
        self.fc1 = torch.nn.Linear(10, 2)
        self.loss = torch.nn.MSELoss()
        self.decay = decay

    def forward(self, inputs, labels):
        out = torch.relu(self.fc(inputs[0]))
        out = torch.sigmoid(self.fc1(out))
        loss = self.loss(out, labels[0])
        return out, loss

    def getOptimizer(self):
        if not self.decay:
            return torch.optim.SGD(self.parameters(), lr=0.1)
        else:
            decay_params = []
            non_decay_params = []
            for name, params in self.named_parameters():
                if name == 'fc.weight' or name == 'fc1.bias':
                    decay_params.append(params)
                else:
                    non_decay_params.append(params)
            return torch.optim.SGD([
                {'params': non_decay_params},
                {'params': decay_params, 'weight_decay': 0.9}], lr=0.1)


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
        label_dims=[(3,1,6), (3,1,5)],
        name="mol_attention_masked",
    )

    record_v2(
        MolAttention(query_size=6),
        iteration=2,
        input_dims=[(3,6), (3,4,6), (3,1,5)],
        input_dtype=[float, float, float],
        label_dims=[(3,1,6), (3,1,5)],
        name="mol_attention",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2, bias=False, need_weights=False),
        iteration=2,
        input_dims=[(3,3,6), (3,2,6), (3,2,6)],
        label_dims=[(3,3,6)],
        input_dtype=[float, float, float],
        name="multi_head_attention_disable_need_weights",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2),
        iteration=2,
        input_dims=[(3,3,6), (3,2,6), (3,2,6)],
        label_dims=[(3,3,6), (3,3,2)],
        input_dtype=[float, float, float],
        name="multi_head_attention",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2, kdim=4, vdim=5),
        iteration=2,
        input_dims=[(3,3,6), (3,2,4), (3,2,5)],
        label_dims=[(3,3,6), (3,3,2)],
        input_dtype=[float, float, float],
        name="multi_head_attention_kdim_vdim",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2, provide_attention_mask=True),
        iteration=2,
        input_dims=[(3,3,6), (3,2,6), (3,2,6), (6,3,2)],
        label_dims=[(3,3,6), (3,3,2)],
        input_dtype=[float, float, float, float],
        input_label_reader=MultiHeadAttention.input_label_reader,
        name="multi_head_attention_float_attn_mask",
    )

    # @todo: change this pseudo bool type tensor to actual bool tensor
    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2, provide_attention_mask=True),
        iteration=2,
        input_dims=[(3,3,6), (3,2,6), (3,2,6), (6,3,2)],
        label_dims=[(3,3,6), (3,3,2)],
        input_dtype=[float, float, float, bool],
        input_label_reader=MultiHeadAttention.input_label_reader,
        name="multi_head_attention_pseudo_bool_attn_mask",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2),
        iteration=2,
        input_dims=[(3,3,6)],
        label_dims=[(3,3,6), (3,3,3)],
        input_dtype=[float],
        name="multi_head_attention_self_attention",
    )

    fc_relu_decay = FCRelu(decay=True)
    record_v2(
        fc_relu_decay,
        iteration=2,
        input_dims=[(3,3)],
        input_dtype=[float],
        label_dims=[(3,2)],
        name="fc_relu_decay",
        optimizer=fc_relu_decay.getOptimizer()
    )

    inspect_file("fc_relu_decay.nnmodelgolden")
