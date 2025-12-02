#!/usr/bin/env bash
##
## @file prepare_android_deps.sh
## @author Jihoon Lee <jhoon.it.lee@samsung.com>
## @date 22 October 2020
## @brief prepare android dependency from internet

set -e

TARGET=$1
[ -z $1 ] && TARGET=$(pwd)
echo $TARGET

if [ ! -d $TARGET ]; then
  mkdir -p $TARGET
  wget -r -np -nH --cut-dirs=8 -R "*html*" http://3.37.7.125/nntrainer/ci/daily-build/build_result/lts/latest/android/v0.4.y/ -P $TARGET
fi

