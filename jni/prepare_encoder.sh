#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2023 Donghak Park <donghak.park@samsung.com>
#
# @file prepare_encoder.sh
# @date 10 october 2023
# @brief This file is a helper tool to build encoder at LLM
# @author Donghak Park <donghak.park@samsung.com>
#
# usage: ./prepare_encoder.sh target version

set -e
TARGET=$1
TARGET_VERSION=$2
TAR_PREFIX=encoder

TAR_NAME=${TAR_PREFIX}-${TARGET_VERSION}.tar.gz

URL="https://github.com/nnstreamer/nnstreamer-android-resource/raw/main/external/${TAR_NAME}"

echo "PREPARING Encoder at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}

pushd ${TARGET}

function _download_encoder {
  [ -f $TAR_NAME ] && echo "${TAR_NAME} exists, skip downloading" && return 0

  echo "[Encoder] downloading ${TAR_NAME}\n"
  if ! wget -q ${URL}; then
    echo "[Encoder] Download failed, please check url\n"
    exit $?
  fi
  echo "[Encoder] Finish downloading encoder\n"

}

function _untar_encoder {

  echo "[Encoder] untar encoder\n"
  tar -zxvf ${TAR_NAME} -C ${TARGET}
  rm -f ${TAR_NAME}

  if [ ${TARGET_VERSION} == "0.1" ]; then
    cp -f json.hpp ../nntrainer/utils/
    mv -f ctre-unicode.hpp json.hpp encoder.hpp ../Applications/PicoGPT/jni/
    echo "[Encoder] Finish moving encoder to PicoGPT\n"
  fi

  if [ ${TARGET_VERSION} == "0.2" ]; then
    cp -f json.hpp ../nntrainer/utils/
    mv -f ctre-unicode.hpp json.hpp encoder.hpp ../Applications/LLaMA/jni/
    echo "[Encoder] Finish moving encoder to LLaMA\n"
  fi

}

[ ! -d "${TAR_PREFIX}" ] && _download_encoder && _untar_encoder

popd
