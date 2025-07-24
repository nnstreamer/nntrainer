#!/bin/bash

# Clean previous build
rm -rf build_android
rm -f jni/Android.mk

# Configure meson for Android with FP16 support
meson build_android \
    -Dplatform=android \
    -Denable-fp16=true \
    -Dopenblas-num-threads=1 \
    -Denable-tflite-interpreter=false \
    -Denable-tflite-backbone=false \
    -Domp-num-threads=1 \
    -Denable-opencl=true \
    -Denable-ggml=true \
    -Dhgemm-experimental-kernel=false

# Build using ninja (this will generate Android.mk and run ndk-build)
cd build_android
ninja

echo "Android build with FP16 support completed!"