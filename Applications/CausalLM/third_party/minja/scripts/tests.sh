#!/bin/bash
#
# Copyright 2024 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
#
#
# Builds & run tests. Fuzzing tests are run in "unit test" mode. To run them in "fuzzing mode", use ./run_fuzzing_mode.sh
#
set -euo pipefail

cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j --config Release && \
    ctest --test-dir build -j -C Release --output-on-failure "$@"
