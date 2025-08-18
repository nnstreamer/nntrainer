#! /bin/bash

set -e
GGML_SRC=$1

REVISION=489716ba99ecd51164f79e8c6fec0b5bf634eac9
PATCH_FILE="${GGML_SRC}"/../packagefiles/ggml/0001-nntrainer-ggml-patch.patch

is_patch_applied() {  
    # save patch file to tmp path
    temp_file=$(mktemp)  
    cat "$PATCH_FILE" | grep -v '^$' > "$temp_file"  

    diff_output=$(diff "$temp_file" <(git diff HEAD))  
    rm "$temp_file"  

    if [ -z "$diff_output" ]; then  
        return 1
    else  
        return 0
    fi  

}  

echo "[GGML] PREPARING GGML..."

pushd "${GGML_SRC}"
if !is_patch_applied; then
    git checkout "${REVISION}"
    git restore CMakeLists.txt
    git restore include
    git restore src
    git apply "${PATCH_FILE}"
fi
cp ./src/ggml-cpu/ggml-cpu.c ./src/ggml-cpu/ggml-cpu_c.c
popd

echo "[GGML] PREPARING GGML FINISHED"
