#!/usr/bin/env bash
#
# This is a script to run NNTrainer unit tests on Android devices
# Note that this script assumes to be run on the nntrainer root path.

./tools/package_android.sh

# You can modify test/jni/Android.mk to choose module that you wish to build
pushd test/jni

if [ ! -d $ANDROID_NDK ]; then
  echo "Error: ANDROID_NDK not found."
  exit 1
fi

ndk-build -j$(nproc)

if [ $? != 0 ]; then
  echo "ndk-build failed"
  exit 1
fi

popd
pushd test/libs/arm64-v8a

if [ -v ADB_IP ]; then
  echo "Variable is set"
  ADB_CMD="adb -H ${ADB_IP}"
else
  echo "Variable is not set"
  ADB_CMD="adb"
fi

$ADB_CMD root

if [ $? != 0 ]; then
  echo "$0: adb root failed"
  exit 1
fi

$ADB_CMD shell mkdir -p /data/local/tmp/nntr_android_test/res
$ADB_CMD shell mkdir -p /data/local/tmp/nntr_android_test/nntrainer_opencl_kernels

$ADB_CMD push . /data/local/tmp/nntr_android_test

if [ $? != 0 ]; then
  echo "$0: adb push failed to write to /data/local/tmp/nntr_android_test"
  exit 1
fi

# To test unittest_layer, unittest_model, etc., golden data is required for the layer.
# The steps are as follows.

# $ meson build [flags...]
# meson build will unzip golden data for the unit tests
popd
if [ ! -d build ]; then
  meson build -Dopenblas-num-threads=1  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Domp-num-threads=1 -Denable-opencl=true -Denable-ggml=true -Dhgemm-experimental-kernel=false
else
  echo "warning: build has already been taken, this script tries to reconfigure and try building"
  pushd build
  meson configure -Dopenblas-num-threads=1  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Domp-num-threads=1 -Denable-opencl=true -Denable-ggml=true -Dhgemm-experimental-kernel=false
  popd
fi

cd build
$ADB_CMD push res/ /data/local/tmp/nntr_android_test
