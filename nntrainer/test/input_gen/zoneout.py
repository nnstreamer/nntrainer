#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
#
# @file zoneout.py
# @date 02 December 2021
# @brief Generate Zoneout LSTM cell using torch lstmcell
# @author hyeonseok lee <hs89.lee@samsung.com>

import torch

# Note: Each iteration share the same zoneout mask
class Zoneout(torch.nn.LSTMCell):
    def __init__(self, batch_size, input_size, hidden_size, num_roll=2, hidden_state_zoneout_rate=1, cell_state_zoneout_rate=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        super().__init__(self.input_size, self.hidden_size, bias=True)
        self.hidden_state_zoneout_mask = [ torch.zeros([batch_size, self.hidden_size]).bernoulli_(1. - hidden_state_zoneout_rate) for _ in range(num_roll)]
        self.cell_state_zoneout_mask = [ torch.zeros([batch_size, self.hidden_size]).bernoulli_(1. - cell_state_zoneout_rate) for _ in range(num_roll)]

    def zoneout(self, prev_state, next_state, mask):
        return prev_state * (1. - mask) + next_state * mask

    def forward(self, out, states):
        hidden_state, cell_state, num_unroll = states
        next_hidden_state, next_cell_state = super().forward(out, (hidden_state, cell_state))
        next_hidden_state = self.zoneout(hidden_state, next_hidden_state, self.hidden_state_zoneout_mask[num_unroll])
        next_cell_state = self.zoneout(cell_state, next_cell_state, self.cell_state_zoneout_mask[num_unroll])

        return (next_hidden_state, next_cell_state)
