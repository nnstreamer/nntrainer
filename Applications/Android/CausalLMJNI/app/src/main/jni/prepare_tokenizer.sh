#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file prepare_tokenizer.sh
# @date 25 January 2025
# @brief This file is a helper tool to prepare tokenizer library for CausalLM Android build
# @author Samsung Electronics Co., Ltd.
#
# usage: ./prepare_tokenizer.sh

set -e

TARGET=lib
TAR_PREFIX=tokenizers_android
TAR_NAME=${TAR_PREFIX}.tar.gz
URL="https://github.com/nnstreamer/nnstreamer-android-resource/raw/master/external/${TAR_NAME}"

echo "PREPARING Tokenizer library at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}
[ ! -d ${TARGET}/arm64-v8a ] && mkdir -p ${TARGET}/arm64-v8a

function _download_tokenizer {
  [ -f $TAR_NAME ] && echo "${TAR_NAME} exists, skip downloading" && return 0
  echo "[Tokenizer] downloading ${TAR_NAME}"
  if ! wget -q ${URL} ; then
    echo "[Tokenizer] Download failed, please check url"
    exit $?
  fi
  echo "[Tokenizer] Finish downloading tokenizer"
}

function _untar_tokenizer {
  echo "[Tokenizer] untar ${TAR_NAME}"
  if ! tar -zxf ${TAR_NAME} ; then
    echo "[Tokenizer] untar failed"
    exit $?
  fi
  rm -f ${TAR_NAME}

  # Move tokenizer library to appropriate directory
  mv -f libtokenizers_c.a ${TARGET}/arm64-v8a/
  echo "[Tokenizer] Finish moving tokenizer library"
}

_download_tokenizer
_untar_tokenizer

echo "[Tokenizer] Preparation completed"