#!/usr/bin/env bash
##
## @file prepare_android_deps.sh
## @author Jijoong Moon <jijoong.moon@samsung.com>
## @date 20 March 2022
## @brief prepare android dependency

set -e

TARGET=$(pwd)/nntrainer
echo $TARGET

if [ -L "$TARGET" ]
then
    rm $TARGET
fi    

ln -s ../../../../../../../builddir/android_build_result $TARGET

ndk-build

if [ -d "../jniLibs" ]
then
    rm -r ../jniLibs
fi

mv ../libs ../jniLibs
