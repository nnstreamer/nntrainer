#!/bin/bash

set -e

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
NNTRAINER_ROOT=${SCRIPT_PATH}/../../../../../../../

echo "NNTRAINER_ROOT: ${NNTRAINER_ROOT}"

if [ ! -f ${NNTRAINER_ROOT}/nntrainer_for_android.tar.gz ]; then
    echo "nntrainer_for_android.tar.gz not found. Please run package_android.sh first."
    exit 1
fi

if [ -d ${SCRIPT_PATH}/nntrainer ]; then
    echo "Removing existing nntrainer directory"
    rm -rf ${SCRIPT_PATH}/nntrainer
fi

echo "Extracting nntrainer_for_android.tar.gz"
cd ${SCRIPT_PATH}
tar -xzf ${NNTRAINER_ROOT}/nntrainer_for_android.tar.gz
mv android_build_result nntrainer

echo "Android dependencies prepared successfully"