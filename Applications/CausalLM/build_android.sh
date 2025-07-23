#!/bin/bash

# Build script for CausalLM Android application
set -e

# Check if NDK path is set
if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "Error: ANDROID_NDK_ROOT is not set. Please set it to your Android NDK path."
    exit 1
fi

# Configuration
BUILD_DIR="build-android"
INSTALL_DIR="install-android"
NNTRAINER_ROOT=$(realpath "$(dirname "$0")/../..")

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf $BUILD_DIR $INSTALL_DIR

# First, build nntrainer for Android using meson
echo "Building nntrainer for Android..."
cd $NNTRAINER_ROOT

# Create Android cross-compilation file
cat > android-cross.ini << EOF
[binaries]
c = '$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang'
cpp = '$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++'
ar = '$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar'
strip = '$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip'

[properties]
sys_root = '$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot'

[host_machine]
system = 'android'
cpu_family = 'aarch64'
cpu = 'aarch64'
endian = 'little'
EOF

# Configure meson build
meson setup $BUILD_DIR \
    --cross-file android-cross.ini \
    --prefix=$PWD/$INSTALL_DIR \
    -Denable-app=false \
    -Denable-test=false \
    -Denable-nnstreamer-plugin=false \
    -Denable-nnstreamer-backbone=false \
    -Dplatform=android

# Build nntrainer
ninja -C $BUILD_DIR
ninja -C $BUILD_DIR install

# Now build CausalLM application using ndk-build
echo "Building CausalLM application..."
cd Applications/CausalLM

# Copy tokenizer library for the target architecture
mkdir -p lib
if [ ! -f "lib/libtokenizers_c.a" ]; then
    echo "Warning: libtokenizers_c.a not found in lib directory."
    echo "Please build or download the tokenizer library for Android arm64-v8a and place it in Applications/CausalLM/lib/"
fi

# Run ndk-build
$ANDROID_NDK_ROOT/ndk-build \
    APP_BUILD_SCRIPT=jni/Android.mk \
    APP_PLATFORM=android-29 \
    NDK_PROJECT_PATH=. \
    NDK_APPLICATION_MK=jni/Application.mk

echo "Build completed successfully!"
echo "Output files are in: libs/arm64-v8a/"