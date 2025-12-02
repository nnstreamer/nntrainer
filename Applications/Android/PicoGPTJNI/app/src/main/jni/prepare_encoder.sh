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
# usage: ./prepare_encoder.sh target

set -e
TARGET=encoder
TAR_PREFIX=encoder
TAR_NAME=${TAR_PREFIX}-0.1.tar.gz
URL="https://github.com/nnstreamer/nnstreamer-android-resource/raw/master/external/${TAR_NAME}"

echo "PREPARING Encoder at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}

function _download_encoder {
  [ -f $TAR_NAME ] && echo "${TAR_NAME} exists, skip downloading" && return 0
  echo "[Encoder] downloading ${TAR_NAME}\n"
  if ! wget -q ${URL} ; then
    echo "[Encoder] Download failed, please check url\n"
    exit $?
  fi
  echo "[Encoder] Finish downloading openblas\n"
}

function _untar_encoder {
  echo "[Encoder] untar encoder\n"
  tar xzvf ${TAR_NAME} -C ${TARGET}
  rm -f ${TAR_NAME}
  cd ${TARGET}
  mv -f ctre-unicode.hpp json.hpp encoder.hpp ..
  cd ..
  rm -r ${TARGET}
  echo "[Encoder] Finish moving encoder to PicoGPT\n"
}

[ ! -d "${TAR_PREFIX}" ] && _download_encoder
_untar_encoder

