#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
#
# @file genModelTests_v2.py
# @date 25 November 2021
# @brief Generate model tcs
# @author Parichay Kapoor <pk.kapoor@samsung.com>

import math
from recorder_v2 import record_v2, inspect_file, _rand_like
import torch
from torch import autocast


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
        self.K = 5  # number of mixtures
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
        j = (
            torch.arange(start=1, end=timesteps + 1)
            .view(1, -1, 1)
            .expand(batch_size, -1, self.K)
        )

        integrals_left = torch.sigmoid(torch.div(j + 0.5 - kappa, beta + 1e-8))
        integrals_right = torch.sigmoid(torch.div(j - 0.5 - kappa, beta + 1e-8))
        integrals = alpha * (integrals_left - integrals_right)
        scores = torch.sum(integrals, dim=2)

        if mask_len is not None:
            max_len = max(int(mask_len.max()), scores.shape[1])
            mask = (
                torch.arange(0, max_len)
                .type_as(mask_len)
                .unsqueeze(0)
                .expand(mask_len.numel(), max_len)
                .lt(mask_len.unsqueeze(1))
            )
            scores.masked_fill_(torch.logical_not(mask), 0.0)

        output = torch.matmul(scores.unsqueeze(1), values).squeeze(dim=1)

        loss = self.loss(torch.sum(output)) + self.loss(torch.sum(kappa))

        return (output, kappa), loss


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        need_weights=True,
        provide_attention_mask=False,
    ):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_attention = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first=True,
        )
        self.loss = torch.nn.MSELoss()
        self.need_weights = need_weights
        self.provide_attention_mask = provide_attention_mask

    def forward(self, inputs, labels):
        inputs, attn_mask = (
            (inputs[:-1], inputs[-1]) if self.provide_attention_mask else (inputs, None)
        )
        query, *left = inputs
        if len(left) == 0:
            key = value = query
        else:
            key, value = left

        output, attention_weight = self.multi_head_attention(
            query, key, value, need_weights=self.need_weights, attn_mask=attn_mask
        )
        loss = self.loss(output, labels[0])
        if attention_weight is not None:
            output = [output, attention_weight]

        return output, loss

    def input_label_reader(input_dims, label_dims, input_dtype):
        query_dim, key_dim, value_dim, *left_dim = input_dims
        query_dtype, key_dtype, value_dtype, *left_dtype = input_dtype
        assert query_dtype == key_dtype == value_dtype
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
        inputs = (
            _rand_like(
                [query_dim, key_dim, value_dim],
                dtype=input_dtype if input_dtype is not None else float,
            )
            + mask
        )
        labels = _rand_like(label_dims, dtype=float)
        return inputs, labels


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.multi_head_attention = torch.nn.MultiheadAttention(
            d_model, 2, batch_first=True
        )
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        output = inputs[0]
        output += self.pe[:, : output.size(1), :]
        output = self.multi_head_attention(output, output, output)
        loss = self.loss(output[0], labels[0])
        return output, loss


# class for test transformer encoder layer
class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, provide_attention_mask=False):
        super(TransformerEncoderLayer, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=0.0, batch_first=True
        )
        self.loss = torch.nn.MSELoss()
        # indicate attention mask will be given or not
        self.provide_attention_mask = provide_attention_mask

    def forward(self, inputs, labels):
        inputs, attn_mask = (
            (inputs[0], inputs[-1])
            if self.provide_attention_mask
            else (inputs[0], None)
        )
        output = self.encoder_layer(inputs, attn_mask)

        loss = self.loss(output, labels[0])

        return output, loss

    def input_label_reader(input_dims, label_dims, input_dtypes):
        input_dim, *left_dim = input_dims
        input_dtype, *left_dtype = input_dtypes
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
        inputs = (
            _rand_like(
                [input_dim], dtype=input_dtype if input_dtype is not None else float
            )
            + mask
        )
        labels = _rand_like(label_dims, dtype=float)
        return inputs, labels


# class for test transformer decoder layer
class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, provide_attention_mask=False):
        super(TransformerDecoderLayer, self).__init__()
        self.decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout=0.0, batch_first=True
        )
        self.loss = torch.nn.MSELoss()
        # indicate attention mask will be given or not
        self.provide_attention_mask = provide_attention_mask

    def forward(self, inputs, labels):
        tgt, memory, tgt_mask, memory_mask = (
            (inputs[0], inputs[1], inputs[-2], inputs[-1])
            if self.provide_attention_mask
            else (inputs[0], inputs[1], None, None)
        )
        output = self.decoder_layer(tgt, memory, tgt_mask, memory_mask)

        loss = self.loss(output, labels[0])

        return output, loss

    def input_label_reader(input_dims, label_dims, input_dtypes):
        tgt_dim, memory_dim, *mask_dims = input_dims
        tgt_dtype, memory_dtype, *mask_dtypes = input_dtypes
        if mask_dims != []:
            if mask_dtypes[0] == bool:
                # Since nntrainer does not support bool type tensor yet, convert mask to float type
                # todo: return bool type mask tensor
                masks = [torch.randn(dim) > 0.5 for dim in mask_dims]
                new_attn_masks = [
                    torch.zeros_like(mask, dtype=torch.float32) for mask in masks
                ]
                for mask, new_attn_mask in zip(masks, new_attn_masks):
                    new_attn_mask.masked_fill_(mask, float("-inf"))
                masks = new_attn_masks
            elif mask_dtypes[0] == int:
                masks = [
                    torch.randint(0, 1, mask_dim, torch.int32) for mask_dim in mask_dims
                ]
            else:
                masks = _rand_like(mask_dims, -1e9, mask_dtypes)
        else:
            masks = []
        inputs = (
            _rand_like(
                [tgt_dim, memory_dim],
                dtype=(
                    [tgt_dtype, memory_dtype]
                    if tgt_dtype is not None and memory_dtype is not None
                    else float
                ),
            )
            + masks
        )
        labels = _rand_like(label_dims, dtype=float)
        return inputs, labels


# class for test transformer.
# Transformer in this class consist of transformer encoder and transformer decoder
class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        provide_attention_mask=False,
    ):
        super(Transformer, self).__init__()
        self.transformer = torch.nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.loss = torch.nn.MSELoss()
        # indicate attention mask will be given or not
        self.provide_attention_mask = provide_attention_mask

    def forward(self, inputs, labels):
        src, tgt, src_mask, tgt_mask, memory_mask = (
            (inputs[0], inputs[1], inputs[-3], inputs[-2], inputs[-1])
            if self.provide_attention_mask
            else (inputs[0], inputs[1], None, None, None)
        )
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)

        loss = self.loss(output, labels[0])

        return output, loss

    def input_label_reader(input_dims, label_dims, input_dtypes):
        src_dim, tgt_dim, *mask_dims = input_dims
        src_dtype, tgt_dtype, *mask_dtypes = input_dtypes
        if mask_dims != []:
            if mask_dtypes[0] == bool:
                # Since nntrainer does not support bool type tensor yet, convert mask to float type
                # todo: return bool type mask tensor
                masks = [torch.randn(dim) > 0.5 for dim in mask_dims]
                new_attn_masks = [
                    torch.zeros_like(mask, dtype=torch.float32) for mask in masks
                ]
                for mask, new_attn_mask in zip(masks, new_attn_masks):
                    new_attn_mask.masked_fill_(mask, float("-inf"))
                masks = new_attn_masks
            elif mask_dtypes[0] == int:
                masks = [
                    torch.randint(0, 1, mask_dim, torch.int32) for mask_dim in mask_dims
                ]
            else:
                masks = _rand_like(mask_dims, -1e9, mask_dtypes)
        else:
            masks = []
        inputs = (
            _rand_like(
                [src_dim, tgt_dim],
                dtype=(
                    [src_dtype, tgt_dtype]
                    if src_dtype is not None and tgt_dtype is not None
                    else float
                ),
            )
            + masks
        )
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
                if name == "fc.weight" or name == "fc1.bias":
                    decay_params.append(params)
                else:
                    non_decay_params.append(params)
            return torch.optim.SGD(
                [
                    {"params": non_decay_params},
                    {"params": decay_params, "weight_decay": 0.9},
                ],
                lr=0.1,
            )


# class for test non-trainable fc layer
class NonTrainableFC(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 2)
        self.loss = torch.nn.MSELoss()
        # determine which layer to set to non-trainable
        fc_layer_list = [self.fc1, self.fc2, self.fc3]
        for param in fc_layer_list[idx - 1].parameters():
            param.requires_grad = False

    def forward(self, inputs, labels):
        out = torch.relu(self.fc1(inputs[0]))
        out = torch.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        loss = self.loss(out, labels[0])
        return out, loss


class LinearMixedPrecision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 10)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        with autocast(device_type="cuda", dtype=torch.float16):
            input = inputs[0].to("cuda")
            label = labels[0].to("cuda")
            out = self.fc(input)
        return out

    def getOptimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


class AddOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = inputs[0] + out
        loss = self.loss(out, labels[0])
        return out, loss


class LinearMixedPrecisionNaNSGD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(1, 1)
        self.fc1 = torch.nn.Linear(1, 1)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        with autocast(device_type="cuda", dtype=torch.float16):
            input = inputs[0].to("cuda")
            label = labels[0].to("cuda")
            out = self.fc0(input)
            out = self.fc1(out)
        return out

    def getOptimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class SubtractOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = inputs[0] - out
        loss = self.loss(out, labels[0])
        return out, loss


class MultiplyOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = inputs[0] * out
        loss = self.loss(out, labels[0])
        return out, loss


class DivideOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = inputs[0] / out
        loss = self.loss(out, labels[0])
        return out, loss


class PowOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = out.pow(3)
        loss = self.loss(out, labels[0])
        return out, loss

class SQRTOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = torch.sqrt(out)
        loss = self.loss(out, labels[0])
        return out, loss

class NegOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = torch.neg(out)
        loss = self.loss(out, labels[0])
        return out, loss


class CosineOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = torch.cos(out)
        loss = self.loss(out, labels[0])
        return out, loss

class SineOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = torch.sin(out)
        loss = self.loss(out, labels[0])
        return out, loss

class TangentOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = torch.tan(out)
        loss = self.loss(out, labels[0])
        return out, loss

class MatMulOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = torch.matmul(inputs[0], out)
        loss = self.loss(out, labels[0])
        return out, loss


class ChannelShuffle(torch.nn.Module):
    def __init__(self, split_number):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(8, 8, 1)  # 1x1 convolution
        self.split_number = split_number
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        # Channel shuffle operation
        batch_size, channels, height, width = inputs[0].shape
        channels_per_group = channels // self.split_number
        
        out = self.conv2d(inputs[0])
        out = out.view(batch_size, self.split_number, channels_per_group, height * width)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, height, width)
        loss = self.loss(out, labels[0])
        
        return out, loss

class UnsqueezeOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 12)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        out = torch.unsqueeze(out, 1)
        loss = self.loss(out, labels[0])
        return out, loss
    
if __name__ == "__main__":
    record_v2(
        ReduceMeanLast(),
        iteration=2,
        input_dims=[
            (
                3,
                2,
            )
        ],
        label_dims=[
            (
                3,
                1,
            )
        ],
        name="reduce_mean_last",
    )

    record_v2(
        MolAttention(query_size=6),
        iteration=2,
        input_dims=[(3, 6), (3, 4, 6), (3, 1, 5), (3)],
        input_dtype=[float, float, float, int],
        label_dims=[(3, 1, 6), (3, 1, 5)],
        name="mol_attention_masked",
    )

    record_v2(
        MolAttention(query_size=6),
        iteration=2,
        input_dims=[(3, 6), (3, 4, 6), (3, 1, 5)],
        input_dtype=[float, float, float],
        label_dims=[(3, 1, 6), (3, 1, 5)],
        name="mol_attention",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2, bias=False, need_weights=False),
        iteration=2,
        input_dims=[(3, 3, 6), (3, 2, 6), (3, 2, 6)],
        label_dims=[(3, 3, 6)],
        input_dtype=[float, float, float],
        name="multi_head_attention_disable_need_weights",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2),
        iteration=2,
        input_dims=[(3, 3, 6), (3, 2, 6), (3, 2, 6)],
        label_dims=[(3, 3, 6), (3, 3, 2)],
        input_dtype=[float, float, float],
        name="multi_head_attention",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2, kdim=4, vdim=5),
        iteration=2,
        input_dims=[(3, 3, 6), (3, 2, 4), (3, 2, 5)],
        label_dims=[(3, 3, 6), (3, 3, 2)],
        input_dtype=[float, float, float],
        name="multi_head_attention_kdim_vdim",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2, provide_attention_mask=True),
        iteration=2,
        input_dims=[(3, 3, 6), (3, 2, 6), (3, 2, 6), (6, 3, 2)],
        label_dims=[(3, 3, 6), (3, 3, 2)],
        input_dtype=[float, float, float, float],
        input_label_reader=MultiHeadAttention.input_label_reader,
        name="multi_head_attention_float_attn_mask",
    )

    # @todo: change this pseudo bool type tensor to actual bool tensor
    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2, provide_attention_mask=True),
        iteration=2,
        input_dims=[(3, 3, 6), (3, 2, 6), (3, 2, 6), (6, 3, 2)],
        label_dims=[(3, 3, 6), (3, 3, 2)],
        input_dtype=[float, float, float, bool],
        input_label_reader=MultiHeadAttention.input_label_reader,
        name="multi_head_attention_pseudo_bool_attn_mask",
    )

    record_v2(
        MultiHeadAttention(embed_dim=6, num_heads=2),
        iteration=2,
        input_dims=[(3, 3, 6)],
        label_dims=[(3, 3, 6), (3, 3, 3)],
        input_dtype=[float],
        name="multi_head_attention_self_attention",
    )

    record_v2(
        PositionalEncoding(d_model=6, max_len=7),
        iteration=1,
        input_dims=[(3, 5, 6)],
        input_dtype=[float],
        label_dims=[(3, 5, 6)],
        name="positional_encoding",
    )

    record_v2(
        TransformerEncoderLayer(d_model=6, nhead=2, dim_feedforward=7),
        iteration=2,
        input_dims=[(3, 5, 6)],
        label_dims=[(3, 5, 6)],
        input_dtype=[float],
        name="transformer_encoder_layer",
    )

    record_v2(
        TransformerEncoderLayer(
            d_model=6, nhead=2, dim_feedforward=7, provide_attention_mask=True
        ),
        iteration=2,
        input_dims=[(3, 5, 6), (6, 5, 5)],
        label_dims=[(3, 5, 6)],
        input_dtype=[float, float],
        input_label_reader=TransformerEncoderLayer.input_label_reader,
        name="transformer_encoder_layer_float_attn_mask",
    )

    record_v2(
        TransformerEncoderLayer(
            d_model=6, nhead=2, dim_feedforward=7, provide_attention_mask=True
        ),
        iteration=2,
        input_dims=[(3, 5, 6), (6, 5, 5)],
        label_dims=[(3, 5, 6)],
        input_dtype=[float, bool],
        input_label_reader=TransformerEncoderLayer.input_label_reader,
        name="transformer_encoder_layer_pseudo_bool_attn_mask",
    )

    record_v2(
        TransformerDecoderLayer(d_model=6, nhead=2, dim_feedforward=7),
        iteration=2,
        input_dims=[(3, 5, 6), (3, 4, 6)],
        label_dims=[(3, 5, 6)],
        input_dtype=[float, float],
        name="transformer_decoder_layer",
    )

    record_v2(
        TransformerDecoderLayer(
            d_model=6, nhead=2, dim_feedforward=7, provide_attention_mask=True
        ),
        iteration=2,
        input_dims=[(3, 5, 6), (3, 4, 6), (6, 5, 5), (6, 5, 4)],
        label_dims=[(3, 5, 6)],
        input_dtype=[float, float, float, float],
        input_label_reader=TransformerDecoderLayer.input_label_reader,
        name="transformer_decoder_layer_float_attn_mask",
    )

    record_v2(
        TransformerDecoderLayer(
            d_model=6, nhead=2, dim_feedforward=7, provide_attention_mask=True
        ),
        iteration=2,
        input_dims=[(3, 5, 6), (3, 4, 6), (6, 5, 5), (6, 5, 4)],
        label_dims=[(3, 5, 6)],
        input_dtype=[float, float, bool, bool],
        input_label_reader=TransformerDecoderLayer.input_label_reader,
        name="transformer_decoder_layer_pseudo_bool_attn_mask",
    )

    record_v2(
        Transformer(
            d_model=6,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=7,
        ),
        iteration=2,
        input_dims=[(3, 5, 6), (3, 4, 6)],
        label_dims=[(3, 4, 6)],
        input_dtype=[float, float],
        name="transformer_single",
    )

    record_v2(
        Transformer(
            d_model=6,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=7,
        ),
        iteration=2,
        input_dims=[(3, 5, 6), (3, 4, 6)],
        label_dims=[(3, 4, 6)],
        input_dtype=[float, float],
        name="transformer_stack",
    )

    record_v2(
        Transformer(
            d_model=6,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=7,
            provide_attention_mask=True,
        ),
        iteration=2,
        input_dims=[(3, 5, 6), (3, 4, 6), (6, 5, 5), (6, 4, 4), (6, 4, 5)],
        label_dims=[(3, 4, 6)],
        input_dtype=[float, float, float, float, float],
        input_label_reader=Transformer.input_label_reader,
        name="transformer_float_attn_mask",
    )

    record_v2(
        Transformer(
            d_model=6,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=7,
            provide_attention_mask=True,
        ),
        iteration=2,
        input_dims=[(3, 5, 6), (3, 4, 6), (6, 5, 5), (6, 4, 4), (6, 4, 5)],
        label_dims=[(3, 4, 6)],
        input_dtype=[float, float, bool, bool, bool],
        input_label_reader=Transformer.input_label_reader,
        name="transformer_pseudo_bool_attn_mask",
    )

    fc_relu_decay = FCRelu(decay=True)
    record_v2(
        fc_relu_decay,
        iteration=2,
        input_dims=[(3, 3)],
        input_dtype=[float],
        label_dims=[(3, 2)],
        name="fc_relu_decay",
        optimizer=fc_relu_decay.getOptimizer(),
    )

    non_trainable_fc_idx1 = NonTrainableFC(idx=1)
    record_v2(
        non_trainable_fc_idx1,
        iteration=2,
        input_dims=[(3, 3)],
        input_dtype=[float],
        label_dims=[(3, 2)],
        name="non_trainable_fc_idx1",
    )

    non_trainable_fc_idx2 = NonTrainableFC(idx=2)
    record_v2(
        non_trainable_fc_idx2,
        iteration=2,
        input_dims=[(3, 3)],
        input_dtype=[float],
        label_dims=[(3, 2)],
        name="non_trainable_fc_idx2",
    )

    non_trainable_fc_idx3 = NonTrainableFC(idx=3)
    record_v2(
        non_trainable_fc_idx3,
        iteration=2,
        input_dims=[(3, 3)],
        input_dtype=[float],
        label_dims=[(3, 2)],
        name="non_trainable_fc_idx3",
    )

    fc_mixed_training = LinearMixedPrecision()
    record_v2(
        fc_mixed_training,
        iteration=3,
        input_dims=[(1, 3)],
        input_dtype=[float],
        label_dims=[(1, 10)],
        name="fc_mixed_training",
        optimizer=fc_mixed_training.getOptimizer(),
        type="mixed",
    )

    inspect_file("fc_mixed_training.nnmodelgolden")

    add_operation = AddOperation()
    record_v2(
        add_operation,
        iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="add_operation",
    )

    subtract_operation = SubtractOperation()
    record_v2(
        subtract_operation,
        iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="subtract_operation",
    )

    multiply_operation = MultiplyOperation()
    record_v2(
        multiply_operation,
        iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="multiply_operation",
    )

    pow_operation = PowOperation()
    record_v2(
        pow_operation,
        iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="pow_operation",
    )

    sqrt_operation = SQRTOperation()
    record_v2(
        sqrt_operation,
        iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="sqrt_operation",
    )

    neg_operation = NegOperation()
    record_v2(
        neg_operation,
        iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="neg_operation",
    )

    cosine_operation = CosineOperation()
    record_v2(
        cosine_operation,
        iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="cosine_operation",
    )

    sine_operation = SineOperation()
    record_v2(
        sine_operation,
                iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="cosine_operation",
    )

    tangent_operation = TangentOperation()
    record_v2(
        tangent_operation,
        iteration=2,
        input_dims=[(1, 2)],
        input_dtype=[float],
        label_dims=[(1, 2)],
        name="tangent_operation",
    )

    matmul_operation = MatMulOperation()
    record_v2(
        matmul_operation,
        iteration=2,
        input_dims=[(2, 2)],
        input_dtype=[float],
        label_dims=[(2, 2)],
        name="matmul_operation",
    )

    # Function to check the created golden test file
    inspect_file("add_operation.nnmodelgolden")
    fc_mixed_training_nan_sgd = LinearMixedPrecisionNaNSGD()
    record_v2(
        fc_mixed_training_nan_sgd,
        iteration=5,
        input_dims=[(1, 1)],
        input_dtype=[float],
        label_dims=[(1, 1)],
        name="fc_mixed_training_nan_sgd",
        optimizer=fc_mixed_training_nan_sgd.getOptimizer(),
        type="mixed",
    )

    #    Function to check the created golden test file
    inspect_file("non_trainable_fc_idx3.nnmodelgolden")

    # Add ChannelShuffle test
    channel_shuffle = ChannelShuffle(split_number=4)
    record_v2(
        channel_shuffle,
        iteration=2,
        input_dims=[(1, 8, 4, 4)],  # batch_size=1, channels=8, height=4, width=4
        input_dtype=[float],
        label_dims=[(1, 8, 4, 4)],
        name="channel_shuffle",
    )

    #    Function to check the created golden test file
    inspect_file("channel_shuffle.nnmodelgolden")
    
    unsqueeze_operation = UnsqueezeOperation()
    record_v2(
        unsqueeze_operation,
        iteration=2,
        input_dims=[(2, 3)],
        input_dtype=[float],
        label_dims=[(2, 1, 12)],
        name="unsqueeze_operation",
    )
    
    inspect_file("unsqueeze_operation.nnmodelgolden")
