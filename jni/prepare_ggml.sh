#!/bin/bash

set -e
GGML_SRC=$1

REVISION=489716ba99ecd51164f79e8c6fec0b5bf634eac9
PATCH_FILE="${GGML_SRC}"/../packagefiles/ggml/0001-nntrainer-ggml-patch.patch

is_patch_applied() {
    # check if the patch is already applied
    if git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
        return 0  
    else
        return 1  
    fi
}

echo "[GGML] PREPARING GGML..."

pushd "${GGML_SRC}"

if is_patch_applied; then
    echo "[GGML] Patch is already applied. Only performing file copy."
else
    echo "[GGML] Patch not applied. Performing full setup..."
    
    # checkout to deisgnated version
    git checkout "${REVISION}"
    
    # restore files
    git restore CMakeLists.txt
    git restore include
    git restore src
    
    # apply patch
    echo "[GGML] Applying patch..."
    git apply "${PATCH_FILE}"
    echo "[GGML] Patch applied successfully."
fi

# copy ggml files
if [ -f "./src/ggml-cpu/ggml-cpu.c" ]; then
    cp ./src/ggml-cpu/ggml-cpu.c ./src/ggml-cpu/ggml-cpu_c.c
    echo "[GGML] Copied ggml-cpu.c to ggml-cpu_c.c"
else
    echo "[GGML] WARNING: ./src/ggml-cpu/ggml-cpu.c not found. Skipping file copy."
fi

popd

echo "[GGML] PREPARING GGML FINISHED"
