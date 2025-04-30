#!/usr/bin/env bash

set -e

TARGET=$1
[ -z $1 ] && TARGET=$(pwd)
echo $TARGET


if [ ! -d $TARGET ]; then
    if [[ $1 == -D* ]]; then
	TARGET=$(pwd)
	echo $TARGET
    else
	echo $TARGET is not a directory. please put project root of nntrainer
	exit 1
    fi
fi

pushd $TARGET

filtered_args=()

for arg in "$@"; do
    if [[ $arg == -D* ]]; then
	filtered_args+=("$arg")
    fi
done

if [ ! -d builddir ]; then
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
  meson builddir -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Domp-num-threads=1 -Denable-opencl=true -Dhgemm-experimental-kernel=false -Denable-ggml=true ${filtered_args[@]}
else
  echo "warning: $TARGET/builddir has already been taken, this script tries to reconfigure and try building"
  pushd builddir
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel  
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
    meson configure -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Domp-num-threads=1 -Denable-opencl=true -Dhgemm-experimental-kernel=false -Denable-ggml=true ${filtered_args[@]}
    meson --wipe
  popd
fi

pushd builddir
ninja install

tar -czvf $TARGET/nntrainer_for_android.tar.gz --directory=android_build_result .

popd
popd

