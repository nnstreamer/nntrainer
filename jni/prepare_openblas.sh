#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file prepare_openblas.sh
# @date 08 December 2020
# @brief This file is a helper tool to build android
# @author Jihoon lee <jhoon.it.lee@samsung.com>
#
# usage: ./prepare_openblas.sh target

set -e
TARGET=$1
TAR_PREFIX=openblas

TAR_NAME=${TAR_PREFIX}-0.2.20.tar.gz
URL="https://raw.githubusercontent.com/nnstreamer/nnstreamer-android-resource/main/external/${TAR_NAME}"

echo "PREPARING OPENBLAS at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}

pushd ${TARGET}

function _download_cblas {
  [ -f $TAR_NAME ] && echo "${TAR_NAME} exists, skip downloading" && return 0
  echo "[OPENBLAS] downloading ${TAR_NAME}\n"
  if ! wget -q ${URL} ; then
    echo "[OPENBLAS] Download failed, please check url\n"
    exit $?
  fi
  echo "[OPENBLAS] Finish downloading openblas\n"
}

function _untar_cblas {
  echo "[OPENBLAS] untar openblas\n"
  tar -zxvf ${TAR_NAME} -C ${TARGET}
  rm -f ${TAR_NAME}
}

[ ! -d "${TAR_PREFIX}" ] && _download_cblas && _untar_cblas

popd
