#!/usr/bin/env bash

set -e

BASEDIR=$(dirname $0)

BiQGEMM_NUM_THREADS=$1
[ -z $1 ] && BiQGEMM_NUM_THREADS=6
GEMM_NUM_THREADS=$2
[ -z $2 ] && GEMM_NUM_THREADS=4
GEMV_NUM_THREADS=$3
[ -z $3 ] && GEMV_NUM_THREADS=2

$BASEDIR/../build_for_llama.sh $BiQGEMM_NUM_THREADS $GEMM_NUM_THREADS $GEMV_NUM_THREADS

TARGET=$4
[ -z $4 ] && TARGET=$(pwd)
echo $TARGET

if [ ! -d $TARGET ]; then
  echo $TARGET is not a directory. please put project root of nntrainer
  exit 1
fi

pushd $TARGET

if [ ! -d builddir ]; then
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
  meson builddir -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Denable-neon=true -Domp-num-threads=1 -Denable-opencl=true
else
  echo "warning: $TARGET/builddir has already been taken, this script tries to reconfigure and try building"
  pushd builddir
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel  
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
    meson configure -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Denable-neon=true -Domp-num-threads=1 -Denable-opencl=true
    meson --wipe
  popd
fi

pushd builddir
ninja install

tar -czvf $TARGET/nntrainer_for_android.tar.gz --directory=android_build_result .

popd
popd

