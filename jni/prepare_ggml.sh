#! /bin/bash

set -e
GGML_SRC=$1

REVISION=489716ba99ecd51164f79e8c6fec0b5bf634eac9
PATCH_PATH="${GGML_SRC}"/../packagefiles/ggml/0001-Export-some-internal-methods.patch

echo "[GGML] PREPARING GGML..."

pushd "${GGML_SRC}"
git checkout "${REVISION}"
git restore CMakeLists.txt
git restore include
git restore src
git apply "${PATCH_PATH}"
cp ./src/ggml-cpu/ggml-cpu.c ./src/ggml-cpu/ggml-cpu_c.c
popd

echo "[GGML] PREPARING GGML FINISHED"
