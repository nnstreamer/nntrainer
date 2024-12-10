#!/bin/bash

if [ "x${HEXAGON_SDK_ROOT}" = "x" ]; then
    echo "HEXAGON_SDK_ROOT is not set, we will set evn using /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.2.0/setup_sdk_env.source"
    ln -s /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.2.0/ HexagonSDK
    source HexagonSDK/setup_sdk_env.source
fi

echo "QNN_SDK_ROOT is not set, we will set /opt/qcom/aistack/qairt/2.28.2.241116/"
ln -s /opt/qcom/aistack/qairt/2.28.2.241116/ qairt
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.28.2.241116/
source ${QNN_SDK_ROOT}/bin/envsetup.sh

echo "QNN_SDK_ROOT=./qairt"
echo "HEXAGON_SDK_ROOT=./HexagonSDK"

echo "ANDROID_ROOT_DIR=${ANDROID_ROOT_DIR}"
echo "ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}"
echo "QNX_BIN_DIR=${QNX_BIN_DIR}"
echo "LV_TOOLS_DIR=${LV_TOOLS_DIR}"
echo "LRH_TOOLS_DIR=${LRH_TOOLS_DIR}"

echo "DEFAULT_HEXAGON_TOOLS_ROOT=${DEFAULT_HEXAGON_TOOLS_ROOT}"
echo "DEFAULT_DSP_ARCH=${DEFAULT_DSP_ARCH}"
ehco "DEFAULT_BUILD=${DEFAULT_BUILD}"
echo "DEFAULT_HLOS_ARCH=${DEFAULT_HLOS_ARCH}"
echo "DEFAULT_TOOLS_VARIANT=${DEFAULT_TOOLS_VARIANT}"
echo "DEFAULT_NO_OURT_INC=${DEFAULT_NO_QURT_INC}"
echo "DEFAULT_TREE=${DEFAULT_TREE}"
echo "CMAKE_ROOT_PATH=${CMAKE_ROOT_PATH}"
echo "DEBUGGER_UTILS=${DEBUGGER_UTILS}"
echo "HEXAGONSDK_TELEMATICS_ROOT=$HEXAGONSDK_TELEMATICS_ROOT}"

echo "AISW_SDK_ROOT=${AISW_SDK_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "PATH=${PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "HEXAGON_TOOLS_DIR=${HEXAGON_TOOLS_DIR}"
echo "SNPE_ROOT=${SNPE_ROOT}"

cd LLaMAPackage

make htp_v75 && make htp_aarch64

