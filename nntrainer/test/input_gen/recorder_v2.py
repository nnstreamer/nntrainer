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
import torch.cuda.amp as amp
from torch import autocast

from transLayer_v2 import params_translated

if torch.__version__ != "2.4":
    print(
        "the script was tested at version 2.4 it might not work if torch version is different"
    )

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

__all__ = ["record_v2", "inspect_file"]


def _get_writer(file):
    def write_fn(items, type="float32"):
        if not isinstance(items, (list, tuple)):
            items = [items]

        for item in items:
            print(item.numel(), " -0-----")
            print(item)
            np.array([item.numel()], dtype="int32").tofile(file)
            a = np.array(item.detach().cpu(), dtype=type)
            a.tofile(file)
            print(a.dtype)

        return items

    return write_fn


def _get_writer_mixed(file):
    def write_fn(items, num_type="int32", type="float32"):
        if not isinstance(items, (list, tuple)):
            items = [items]

        for item in items:
            np.array([item.numel()], dtype=num_type).tofile(file)
            a = np.array(item.detach().cpu(), dtype=type)
            a.tofile(file)

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
    np_array = list([shape_to_np(s, t) for s, t in zip(shapes, dtype)])
    return list([torch.tensor(t * scale) for t in np_array])


##
# @brief record a torch model
# @param iteration number of iteration to record
# @param input_dims dimensions to record including batch (list of tuple)
# @param label_dims dimensions to record including batch (list of tuple)
# @param name golden name
def record_v2(
    model,
    iteration,
    input_dims,
    label_dims,
    name,
    clip=False,
    input_dtype=None,
    input_label_reader=None,
    optimizer=None,
    type="default",
):
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
            inputs, labels = input_label_reader(input_dims, label_dims, input_dtype)
        else:
            inputs = _rand_like(
                input_dims, dtype=input_dtype if input_dtype is not None else float
            )
            labels = _rand_like(label_dims, dtype=float)
        write_fn(inputs)
        write_fn(labels)
        write_fn(list(t for _, t in params_translated(model)))
        output, *losses = model(inputs, labels)
        write_fn(output)

        optimizer.zero_grad()
        for loss in losses:
            loss.backward()
        if clip:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        optimizer.step()

    def record_iteration_with_amp(write_fn, inputs, labels, is_nan, scaler):
        model_ = model.cuda()

        print(inputs[0], " inputs inside")
        output = model_(inputs[0].cuda(), labels[0].cuda())

        print("model output type: ", output.dtype)

        with autocast(device_type="cuda", dtype=torch.float16):
            l = model_.loss(output, labels[0].to("cuda"))

        optimizer.zero_grad()

        scaler.scale(l).backward()
        print("Gradient      ---------------")
        for param in model_.parameters():
            print(param.grad)
            is_nan = torch.any(torch.isnan(param.grad) | torch.isinf(param.grad))
            if is_nan:
                print("nan or inf detected in gradient")
                break

        if not is_nan:
            print("------------------------------- not nan")
            write_fn(output, "int32", "float32")
        return output, is_nan

    if type == "default":
        with open(file_name, "wb") as f:
            # write number of iterations
            np.array([iteration], dtype="int32").tofile(f)

            write_fn = _get_writer(f)
            for _ in range(iteration):
                record_iteration(write_fn)

    elif type == "mixed":
        with open(file_name, "wb") as f:
            # write number of iterations
            print("iteration : ", iteration)
            np.array([iteration], dtype="int32").tofile(f)

            write_fn = _get_writer_mixed(f)
            for i in range(iteration):
                if input_label_reader != None:
                    inputs, labels = input_label_reader(
                        input_dims, label_dims, input_dtype
                    )
                else:
                    inputs = _rand_like(
                        input_dims,
                        dtype=input_dtype if input_dtype is not None else float,
                    )
                    labels = _rand_like(label_dims, dtype=float)
                print("inputs ==============")
                write_fn(inputs, "int32", "float32")
                print("labels ==============")
                write_fn(labels, "int32", "float32")
                is_nan = True
                print("=========================== ", i)
                scaler = amp.GradScaler()
                print("weights ==============")
                write_fn(
                    list(t for _, t in params_translated(model)), "int16", "float16"
                )
                print("\n\n")
                while is_nan:
                    print("before is_nan_", is_nan)
                    output, is_nan_ = record_iteration_with_amp(
                        write_fn, inputs, labels, is_nan, scaler
                    )
                    is_nan = is_nan_
                    print("after is_nan_", is_nan)
                    scaler.step(optimizer)
                    scaler.update()


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
