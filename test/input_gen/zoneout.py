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

    def zoneout(self, prev_state, next_state, mask):
        return prev_state * (1. - mask) + next_state * mask

    def forward(self, out, states, masks):
        hidden_state, cell_state = states
        hidden_state_mask, cell_state_mask = masks
        next_hidden_state, next_cell_state = super().forward(out, (hidden_state, cell_state))
        next_hidden_state = self.zoneout(hidden_state, next_hidden_state, hidden_state_mask)
        next_cell_state = self.zoneout(cell_state, next_cell_state, cell_state_mask)

        return (next_hidden_state, next_cell_state)
