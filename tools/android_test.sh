#!/usr/bin/env bash
#
# This is a script to run NNTrainer unit tests on Android devices
# Note that this script assumes to be run on the nntrainer root path.

./tools/package_android.sh

# You can modify test/jni/Android.mk to choose module that you wish to build
cd test/jni

# Perequisite: Install and configure the NDK
ndk-build

if [ $? != 0 ]; then
  echo "ndk-build failed"
  exit 1
fi

cd ../libs/arm64-v8a

adb root

if [ $? != 0 ]; then
  echo "$0: adb root failed"
  exit 1
fi

adb shell mkdir -p /data/local/tmp/nntr_android_test

adb push . /data/local/tmp/nntr_android_test

if [ $? != 0 ]; then
  echo "$0: adb push failed to write to /data/local/tmp/nntr_android_test"
  exit 1
fi

# To test unittest_layer, unittest_model, etc., golden data is required for the layer.
# The steps are as follows.

# $ meson build [flags...]
# $ cd build
# $ adb push res/ /data/local/tmp/nntr_android_test
