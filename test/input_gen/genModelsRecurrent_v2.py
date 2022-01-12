#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file genModelsRecurrent_v2.py
# @date 19 October 2021
# @brief Generate recurrent model tcs
# @author Jihoon lee <jhoon.it.lee@samsung.com>

from recorder_v2 import record_v2, inspect_file
from zoneout import Zoneout
import torch

class FCUnroll(torch.nn.Module):
    def __init__(self, unroll_for=1, num_fc=1):
        super().__init__()
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(1, 1) for i in range(num_fc)])
        self.unroll_for = unroll_for
        # self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.Identity()

    def forward(self, inputs, labels):
        output = inputs[0]
        for i in range(self.unroll_for):
            for fc in self.fcs:
                output = fc(output)
        loss = self.loss(output)
        # loss = self.loss(output, labels[0])
        return output, loss

class RNNCellStacked(torch.nn.Module):
    def __init__(self, unroll_for=1, num_rnn=1, input_size=1, hidden_size=1):
        super().__init__()
        self.rnns = torch.nn.ModuleList(
            [
                torch.nn.RNNCell(input_size, hidden_size)
                for _ in range(num_rnn)
            ]
        )
        self.unroll_for = unroll_for
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        hs = [torch.zeros_like(inputs[0]) for _ in self.rnns]
        out = inputs[0]
        ret = []
        for _ in range(self.unroll_for):
            for i, rnn in enumerate(self.rnns):
                hs[i] = rnn(out, hs[i])
                out = hs[i]
            ret.append(out)

        ret = torch.stack(ret, dim=1)
        loss = self.loss(ret, labels[0])
        return ret, loss

class LSTMStacked(torch.nn.Module):
    def __init__(self, num_lstm=1):
        super().__init__()
        self.input_size = self.hidden_size = 2
        self.num_lstm = num_lstm
        self.lstms = torch.nn.ModuleList(
            [
                torch.nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
                # torch.nn.LSTM(self.input_size, self.hidden_size, num_layers=num_lstm, batch_first=True)
                for _ in range(num_lstm)
            ]
        )
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = inputs[0]
        states = inputs[1:]
        # hs = [states[2 * i] for i in range(self.num_lstm)]
        hs = [torch.zeros((1, 3, 2)) for _ in range(self.num_lstm)]
        # cs = [states[2 * i + 1] for i in range(self.num_lstm)]
        cs = [torch.zeros((1, 3, 2)) for _ in range(self.num_lstm)]
        for i, (lstm, h, c) in enumerate(zip(self.lstms, hs, cs)):
            out, (hs[i], cs[i]) = lstm(out, (h, c))

        loss = self.loss(out, labels[0])
        return out, loss

class LSTMCellStacked(torch.nn.Module):
    def __init__(self, unroll_for=2, num_lstmcell=1):
        super().__init__()
        self.input_size = self.hidden_size = 2
        self.lstmcells = torch.nn.ModuleList(
            [
                torch.nn.LSTMCell(self.input_size, self.hidden_size)
                for _ in range(num_lstmcell)
            ]
        )
        self.unroll_for = unroll_for
        self.num_lstmcell = num_lstmcell
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = inputs[0]
        states = inputs[1:]
        hs = [states[2 * i] for i in range(self.num_lstmcell)]
        cs = [states[2 * i + 1] for i in range(self.num_lstmcell)]
        ret = []
        for _ in range(self.unroll_for):
            for i, (lstm, h, c) in enumerate(zip(self.lstmcells, hs, cs)):
                hs[i], cs[i] = lstm(out, (h, c))
                out = hs[i]
            ret.append(out)

        ret = torch.stack(ret, dim=1)
        loss = self.loss(ret, labels[0])
        return ret, loss

class ZoneoutLSTMStacked(torch.nn.Module):
    def __init__(self, batch_size=3, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=1, cell_state_zoneout_rate=1):
        super().__init__()
        self.input_size = self.hidden_size = 2
        self.cell_state_zoneout_rate = cell_state_zoneout_rate
        self.zoneout_lstms = torch.nn.ModuleList(
            [
                Zoneout(batch_size, self.input_size, self.hidden_size, unroll_for, hidden_state_zoneout_rate, cell_state_zoneout_rate)
                for _ in range(num_lstm)
            ]
        )
        self.unroll_for = unroll_for
        self.num_lstm = num_lstm
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = inputs[0]
        states = inputs[1:]
        hs = [states[2 * i] for i in range(self.num_lstm)]
        cs = [states[2 * i + 1] for i in range(self.num_lstm)]
        ret = []
        for num_unroll in range(self.unroll_for):
            for i, (zoneout_lstm, h, c) in enumerate(zip(self.zoneout_lstms, hs, cs)):
                hs[i], cs[i] = zoneout_lstm(out, (h, c, num_unroll))
                out = hs[i]
            ret.append(out)

        ret = torch.stack(ret, dim=1)
        loss = self.loss(ret, labels[0])
        return ret, loss

class GRUCellStacked(torch.nn.Module):
    def __init__(self, unroll_for=2, num_grucell=1):
        super().__init__()
        self.input_size = self.hidden_size = 2
        self.grus = torch.nn.ModuleList(
            [
                torch.nn.GRUCell(self.input_size, self.hidden_size, bias=True)
                for _ in range(num_grucell)
            ]
        )
        self.unroll_for = unroll_for
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = inputs[0]
        hs = inputs[1:]
        ret = []
        for _ in range(self.unroll_for):
            for i, (gru, h) in enumerate(zip(self.grus, hs)):
                hs[i] = gru(out, h)
                out = hs[i]
            ret.append(out)

        ret = torch.stack(ret, dim=1)
        loss = self.loss(ret, labels[0])
        return ret, loss

if __name__ == "__main__":
    record_v2(
        FCUnroll(unroll_for=5),
        iteration=2,
        input_dims=[(1,)],
        label_dims=[(1,)],
        name="fc_unroll_single",
    )

    record_v2(
        FCUnroll(unroll_for=2, num_fc=2),
        iteration=2,
        input_dims=[(1,)],
        label_dims=[(1,)],
        name="fc_unroll_stacked",
    )

    record_v2(
        FCUnroll(unroll_for=2, num_fc=2),
        iteration=2,
        input_dims=[(1,)],
        label_dims=[(1,)],
        name="fc_unroll_stacked_clipped",
        clip=True
    )

    record_v2(
        RNNCellStacked(unroll_for=2, num_rnn=1, input_size=2, hidden_size=2),
        iteration=2,
        input_dims=[(3, 2)],
        label_dims=[(3, 2, 2)],
        name="rnncell_single",
    )

    record_v2(
        RNNCellStacked(unroll_for=2, num_rnn=2, input_size=2, hidden_size=2),
        iteration=2,
        input_dims=[(3, 2)],
        label_dims=[(3, 2, 2)],
        name="rnncell_stacked",
    )

    unroll_for, num_lstm, batch_size, unit, feature_size, iteration = [2, 1, 3, 2, 2, 2]
    record_v2(
        LSTMStacked(num_lstm=num_lstm),
        iteration=iteration,
        input_dims=[(batch_size, unroll_for, feature_size)],
        # input_dims=[(batch_size, unroll_for, feature_size)] + [(1, batch_size, unit) for _ in range(2 * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="lstm_single",
    )

    unroll_for, num_lstm, batch_size, unit, feature_size, iteration = [2, 2, 3, 2, 2, 2]
    record_v2(
        LSTMStacked(num_lstm=num_lstm),
        iteration=iteration,
        input_dims=[(batch_size, unroll_for, feature_size)],
        # input_dims=[(batch_size, unroll_for, feature_size)] + [(1, batch_size, unit) for _ in range(2 * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="lstm_stacked",
    )

    unroll_for, num_lstmcell, state_num, batch_size, unit, feature_size, iteration = [2, 1, 2, 3, 2, 2, 2]
    record_v2(
        LSTMCellStacked(unroll_for=unroll_for, num_lstmcell=num_lstmcell),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstmcell)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="lstmcell_single",
    )

    unroll_for, num_lstmcell, state_num, batch_size, unit, feature_size, iteration = [2, 2, 2, 3, 2, 2, 2]
    record_v2(
        LSTMCellStacked(unroll_for=unroll_for, num_lstmcell=num_lstmcell),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstmcell)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="lstmcell_stacked",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 0.0, 0.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_000_000",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 0.0, 0.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_000_000",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 0.5, 0.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_050_000",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 0.5, 0.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_050_000",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 1.0, 0.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_100_000",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 1.0, 0.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_100_000",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 0.0, 0.5]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_000_050",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 0.0, 0.5]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_000_050",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 0.5, 0.5]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_050_050",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 0.5, 0.5]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_050_050",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 1.0, 0.5]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_100_050",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 1.0, 0.5]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_100_050",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 0.0, 1.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_000_100",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 0.0, 1.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_000_100",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 0.5, 1.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_050_100",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 0.5, 1.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_050_100",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 1, 2, 1, 2, 2, 2, 1.0, 1.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_single_100_100",
    )

    unroll_for, num_lstm, state_num, batch_size, unit, feature_size, iteration, hidden_state_zoneout_rate, cell_state_zoneout_rate = [2, 2, 2, 1, 2, 2, 2, 1.0, 1.0]
    record_v2(
        ZoneoutLSTMStacked(batch_size=batch_size, unroll_for=unroll_for, num_lstm=num_lstm, hidden_state_zoneout_rate=hidden_state_zoneout_rate, cell_state_zoneout_rate=cell_state_zoneout_rate),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num * num_lstm)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="zoneout_lstm_stacked_100_100",
    )

    unroll_for, num_grucell, batch_size, unit, feature_size, iteration, = [2, 1, 3, 2, 2, 2]
    record_v2(
        GRUCellStacked(unroll_for=unroll_for, num_grucell=num_grucell),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(num_grucell)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="grucell_single",
    )

    unroll_for, num_grucell, batch_size, unit, feature_size, iteration, = [2, 2, 3, 2, 2, 2]
    record_v2(
        GRUCellStacked(unroll_for=unroll_for, num_grucell=num_grucell),
        iteration=iteration,
        input_dims=[(batch_size, feature_size)] + [(batch_size, unit) for _ in range(num_grucell)],
        label_dims=[(batch_size, unroll_for, unit)],
        name="grucell_stacked",
    )

    # inspect_file("lstm_single.nnmodelgolden")
