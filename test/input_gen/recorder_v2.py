#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file recorder_v2.py
# @date 18 October 2021
# @brief Generate tc from given torch model
# @author Jihoon lee <jhoon.it.lee@samsung.com>

import os
import random
import torch  # torch used here is torch==1.9.1
import numpy as np

from transLayer_v2 import params_translated

if torch.__version__ != "1.9.1":
    print(
        "the script was tested at version 1.9.1 it might not work if torch version is different"
    )

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

__all__ = ["record_v2", "inspect_file"]


def _get_writer(file):
    def write_fn(items):
        if not isinstance(items, (list, tuple)):
            items = [items]

        for item in items:
            np.array([item.numel()], dtype="int32").tofile(file)
            item.detach().cpu().numpy().tofile(file)

        return items

    return write_fn


def _rand_like(shapes, scale=1, dtype=None):
    def shape_to_np(shape, dtype=int):
        if dtype == int:
            return np.random.randint(0, 4, shape).astype(dtype=np.int32)
        else:
            return np.random.rand(*shape).astype(dtype=np.float32)

    if not isinstance(dtype, list):
        dtype = [dtype] * len(shapes)
    np_array = list([shape_to_np(s,t) for s,t in zip(shapes, dtype)])
    return list([torch.tensor(t * scale) for t in np_array])


##
# @brief record a torch model
# @param iteration number of iteration to record
# @param input_dims dimensions to record including batch (list of tuple)
# @param label_dims dimensions to record including batch (list of tuple)
# @param name golden name
def record_v2(model, iteration, input_dims, label_dims, name, clip=False,
              input_dtype=None, input_label_reader=None, input_label_reader_params=None, optimizer=None):
    ## file format is as below
    # [<number of iteration(int)> <Iteration> <Iteration>...<Iteration>]
    # Each iteration contains
    # [<input(Tensors)><Label(Tensors)><Parameters(Tensors)><Output(Tensors)>]
    # Each tensor contains
    # [<num_elements(int32)><data_point(float32)>...<data_point(float32)>]

    file_name = name + ".nnmodelgolden"
    if os.path.isfile(file_name):
        print("Warning: the file %s is being truncated and overwritten" % file_name)

    if optimizer == None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def record_iteration(write_fn):
        if input_label_reader != None:
            inputs, labels = input_label_reader(input_dims, label_dims, input_label_reader_params=input_label_reader_params)
        else:
            inputs = _rand_like(input_dims, dtype=input_dtype if input_dtype is not None else float)
            labels = _rand_like(label_dims, dtype=float)
        write_fn(inputs)
        write_fn(labels)
        write_fn(list(t for _, t in params_translated(model)))
        output, loss = model(inputs, labels)
        write_fn(output)

        optimizer.zero_grad()
        loss.backward()
        if clip:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        optimizer.step()

    with open(file_name, "wb") as f:
        # write number of iterations
        np.array([iteration], dtype="int32").tofile(f)

        write_fn = _get_writer(f)
        for _ in range(iteration):
            record_iteration(write_fn)


##
# @brief inpsect if file is created correctly
# @note this just checks if offset is corretly set, The result have to inspected
# manually
def inspect_file(file_name, show_content=True):
    with open(file_name, "rb") as f:
        sz = int.from_bytes(f.read(4), byteorder="little")
        if not sz:
            return
        print("num_iter: ", sz)
        while True:
            sz = int.from_bytes(f.read(4), byteorder="little")
            if not sz:
                break
            print("size: ", sz)
            t = np.fromfile(f, dtype="float32", count=sz)
            if show_content:
                print(t)

