#!/bin/bash

# Exit on error
set -e

echo "=== Building nntrainer for Android with FP16 support ==="

# Step 1: Build nntrainer library with FP16 support
echo "Step 1: Building nntrainer library..."

# Clean previous build
rm -rf builddir
rm -f jni/Android.mk

# Configure meson for Android with FP16 support
meson builddir \
    -Dplatform=android \
    -Denable-fp16=true \
    -Dopenblas-num-threads=1 \
    -Denable-tflite-interpreter=false \
    -Denable-tflite-backbone=false \
    -Domp-num-threads=1 \
    -Denable-opencl=true \
    -Denable-ggml=true \
    -Dhgemm-experimental-kernel=false

# Build using ninja
cd builddir
ninja
ninja install
cd ..

echo "Step 1 completed: nntrainer library built with FP16 support"

# Step 2: Create libs directory structure for applications
echo "Step 2: Setting up library directories..."

# Create libs directory if it doesn't exist
mkdir -p libs/arm64-v8a
mkdir -p libs/armeabi-v7a

# Copy built libraries to libs directory
if [ -d "builddir/android_build_result/lib" ]; then
    cp builddir/android_build_result/lib/*.so libs/arm64-v8a/ || true
    cp builddir/android_build_result/lib/*.so libs/armeabi-v7a/ || true
else
    echo "Warning: Could not find built libraries in builddir/android_build_result/lib"
    echo "Looking for libraries in other locations..."
    find builddir -name "*.so" -type f | while read lib; do
        echo "Found library: $lib"
        cp "$lib" libs/arm64-v8a/ || true
        cp "$lib" libs/armeabi-v7a/ || true
    done
fi

echo "Step 2 completed: Libraries copied to libs directory"

# Step 3: Build LLaMA application
echo "Step 3: Building LLaMA application..."

cd Applications/LLaMA/jni

# Set required environment variables
export ANDROID_NDK=${ANDROID_NDK:-/opt/android-ndk}
export NNTRAINER_ROOT=$(pwd)/../../..

# Check if NDK exists
if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: Android NDK not found at $ANDROID_NDK"
    echo "Please set ANDROID_NDK environment variable to point to your NDK installation"
    exit 1
fi

# Build using ndk-build
ndk-build

cd ../../..

echo "=== Build completed successfully! ==="
echo "Libraries are available in libs/ directory"
echo "LLaMA application binaries are in Applications/LLaMA/libs/"