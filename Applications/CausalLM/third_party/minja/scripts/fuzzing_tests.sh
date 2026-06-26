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
# Builds & run each of the fuzztest tests in the project, with a configurable timeout (default = 60 seconds).
#
# Needs to be run on Linux or a Linux docker image.
#
set -euo pipefail

TIMEOUT=${TIMEOUT:-60}
echo "# TIMEOUT = ${TIMEOUT} seconds (for each test)"

CC=clang CXX=clang++ cmake -B build-fuzz -DCMAKE_BUILD_TYPE=RelWithDebug -DFUZZTEST_FUZZING_MODE=on && \
    cmake --build build-fuzz

TESTS=( $( ./build-fuzz/tests/test-fuzz --gtest_list_tests | awk '/[.] *$/{prefix=$0}/^  (.*)/{print prefix $1 }' ) )

for test in "${TESTS[@]}" ; do
    echo "# Running fuzz test: ${test}"
    timeout "$TIMEOUT" ./build-fuzz/tests/test-fuzz --fuzz="${test}" || true
done
