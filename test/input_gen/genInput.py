#!/usr/bin/env python3
##
# Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
#
# SPDX-License-Identifier: Apache-2.0-only
#
# @file genTestInput.py
# @brief Generate test input
# @author Jijoong Moon <jijoong.moon@samsung.com>

import sys
import os
import numpy
import struct

def gen_input(outfile_name, batch, channel, height, width):
    if os.path.isfile(outfile_name):
        os.remove(outfile_name)

    for i in range(0,batch):
        imarray=numpy.random.rand(channel, height, width)*255
        with open(outfile_name, 'ab') as outfile:
            numpy.array(imarray, dtype=numpy.float32).tofile(outfile)
        print(i+1, " x ", imarray.shape, " data is generated")
        print(imarray)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print('wrong argument : filename, batch, channel, height, width')
    else:
        gen_input(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
