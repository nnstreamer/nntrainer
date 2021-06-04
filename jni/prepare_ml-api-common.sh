#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
#
# @file prepare_ml-api-common.sh
# @date 10 June 2021
# @brief This file is a helper tool to ml-api-common dependency for android build
# @author Parichay Kapoor <pk.kapoor@samsung.com>
#
# usage: ./prepare_ml-api-common.sh target

TARGET=$1
# Note: zip name can be nnstreamer-native-*.zip but this file is heavier to download
FILE_PREFIX=nnstreamer-single-native
ZIP_NAME_REGEX=${FILE_PREFIX}-*.zip
ZIP_NAME=${FILE_PREFIX}.zip
URL="http://nnstreamer.mooo.com/nnstreamer/ci/daily-build/build_result/latest/android"

echo "PREPARING ml_api_common at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}

pushd ${TARGET}

function _download_ml_api_common {
  [ -f $ZIP_NAME ] && echo "${ZIP_NAME} exists, skip downloading" && return 0
  echo "[ml_api_common] downloading ${ZIP_NAME}\n"
  if ! wget -r -l1 -nH --cut-dirs=6 ${URL} -A ${ZIP_NAME_REGEX} -O ${ZIP_NAME} ; then
    echo "[ml_api_common] Download failed, please check url\n"
    exit $?
  fi
  echo "[ml_api_common] Finish downloading ml_api_common\n"
}

function _extract_ml_api_common {
  echo "[ml_api_common] unzip ml_api_common\n"
  unzip -q ${ZIP_NAME} -d ${FILE_PREFIX}
  rm -f ${ZIP_NAME}
}

function _cleanup_ml_api_common {
  echo "[ml_api_common] cleanup ml_api_common \n"
  # move include to the target location
  mv ${FILE_PREFIX}/main/jni/nnstreamer/include .
  # remove all directories/files other than include
  rm -rf ${FILE_PREFIX}
  # cleanup all files other than ml_api_common and tizen_error
  find include ! \( -name 'ml-api-common.h' -o -name 'tizen_error.h' \) -type f -exec rm -f {} +
}

[ ! -d "${FILE_PREFIX}" ] && _download_ml_api_common && _extract_ml_api_common \
  && _cleanup_ml_api_common

popd
