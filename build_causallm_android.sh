#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file build_causallm_android.sh
# @date 25 January 2025
# @brief Build script for CausalLM Android application
# @author Samsung Electronics Co., Ltd.
#
# Usage: ./build_causallm_android.sh [clean|install]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="${SCRIPT_DIR}"
ANDROID_PROJECT_DIR="${SCRIPT_DIR}/Applications/Android/CausalLMJNI"
BUILD_DIR="${SCRIPT_DIR}/build"
ANDROID_BUILD_DIR="${BUILD_DIR}/android"

# Android NDK settings
export ANDROID_NDK="${ANDROID_NDK:-$HOME/Android/Sdk/ndk/25.2.9519653}"
export ANDROID_SDK="${ANDROID_SDK:-$HOME/Android/Sdk}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Android NDK
    if [ ! -d "$ANDROID_NDK" ]; then
        print_error "Android NDK not found at $ANDROID_NDK"
        print_error "Please set ANDROID_NDK environment variable or install NDK"
        exit 1
    fi
    
    # Check Android SDK
    if [ ! -d "$ANDROID_SDK" ]; then
        print_error "Android SDK not found at $ANDROID_SDK"
        print_error "Please set ANDROID_SDK environment variable or install SDK"
        exit 1
    fi
    
    # Check meson
    if ! command -v meson &> /dev/null; then
        print_error "meson not found. Please install meson build system"
        exit 1
    fi
    
    # Check ninja
    if ! command -v ninja &> /dev/null; then
        print_error "ninja not found. Please install ninja build system"
        exit 1
    fi
    
    print_info "Prerequisites check passed"
}

function clean_build() {
    print_info "Cleaning build directories..."
    rm -rf "${BUILD_DIR}"
    rm -rf "${ANDROID_PROJECT_DIR}/app/src/main/jni/nntrainer"
    print_info "Clean completed"
}

function build_nntrainer_android() {
    print_info "Building NNTrainer for Android..."
    
    mkdir -p "${ANDROID_BUILD_DIR}"
    cd "${ANDROID_BUILD_DIR}"
    
    # Configure cross compilation for Android ARM64
    meson setup \
        --cross-file="${NNTRAINER_ROOT}/cross_file_android_arm64.txt" \
        --buildtype=release \
        --prefix="${ANDROID_BUILD_DIR}/install" \
        -Dplatform=android \
        -Denable-test=false \
        -Denable-app=false \
        -Denable-capi=true \
        -Denable-ccapi=true \
        -Denable-tizen=false \
        -Dml-api-support=disabled \
        -Dopenblas-support=disabled \
        -Dcublas-support=disabled \
        -Dcudnn-support=disabled \
        -Dblas=false \
        -Dopenmp-support=true \
        "${NNTRAINER_ROOT}"
    
    # Build
    ninja
    ninja install
    
    print_info "NNTrainer Android build completed"
}

function build_causallm_library() {
    print_info "Building CausalLM library for Android..."
    
    cd "${ANDROID_BUILD_DIR}"
    
    # Build CausalLM application as shared library
    ninja Applications/CausalLM/causallm
    
    # Copy CausalLM library to install directory
    cp Applications/CausalLM/libcausallm.so install/lib/
    
    print_info "CausalLM library build completed"
}

function prepare_android_project() {
    print_info "Preparing Android project..."
    
    local jni_dir="${ANDROID_PROJECT_DIR}/app/src/main/jni"
    local nntrainer_dir="${jni_dir}/nntrainer"
    
    # Create directories
    mkdir -p "${nntrainer_dir}/lib/arm64-v8a"
    mkdir -p "${nntrainer_dir}/include"
    
    # Copy NNTrainer libraries
    cp "${ANDROID_BUILD_DIR}/install/lib/libnntrainer.so" "${nntrainer_dir}/lib/arm64-v8a/"
    cp "${ANDROID_BUILD_DIR}/install/lib/libccapi-nntrainer.so" "${nntrainer_dir}/lib/arm64-v8a/"
    cp "${ANDROID_BUILD_DIR}/install/lib/libcausallm.so" "${nntrainer_dir}/lib/arm64-v8a/"
    
    # Copy headers
    cp -r "${ANDROID_BUILD_DIR}/install/include/nntrainer" "${nntrainer_dir}/include/"
    
    # Copy CausalLM headers
    cp "${NNTRAINER_ROOT}/Applications/CausalLM/"*.h "${nntrainer_dir}/include/nntrainer/"
    cp -r "${NNTRAINER_ROOT}/Applications/CausalLM/layers" "${nntrainer_dir}/include/nntrainer/"
    
    print_info "Android project preparation completed"
}

function build_android_app() {
    print_info "Building Android application..."
    
    cd "${ANDROID_PROJECT_DIR}"
    
    # Make scripts executable
    chmod +x app/src/main/jni/prepare_tokenizer.sh
    
    # Build APK
    ./gradlew assembleDebug
    
    print_info "Android application build completed"
    print_info "APK location: ${ANDROID_PROJECT_DIR}/app/build/outputs/apk/debug/app-debug.apk"
}

function install_app() {
    print_info "Installing application to connected device..."
    
    cd "${ANDROID_PROJECT_DIR}"
    
    # Check if device is connected
    if ! adb devices | grep -q "device$"; then
        print_error "No Android device connected. Please connect a device and enable USB debugging."
        exit 1
    fi
    
    # Install APK
    adb install -r app/build/outputs/apk/debug/app-debug.apk
    
    print_info "Application installed successfully"
}

function create_cross_file() {
    print_info "Creating cross compilation file..."
    
    cat > "${NNTRAINER_ROOT}/cross_file_android_arm64.txt" << EOF
[binaries]
c = '${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang'
cpp = '${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++'
ar = '${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar'
strip = '${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip'
pkgconfig = 'false'

[host_machine]
system = 'android'
cpu_family = 'aarch64'
cpu = 'arm64'
endian = 'little'

[properties]
sys_root = '${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot'
c_args = ['-fPIC', '-DANDROID', '-D__ANDROID_API__=24']
cpp_args = ['-fPIC', '-DANDROID', '-D__ANDROID_API__=24', '-std=c++17']
c_link_args = ['-fPIC']
cpp_link_args = ['-fPIC']
EOF
    
    print_info "Cross compilation file created"
}

function main() {
    local command="${1:-build}"
    
    case "$command" in
        "clean")
            clean_build
            ;;
        "install")
            install_app
            ;;
        "build")
            check_prerequisites
            create_cross_file
            build_nntrainer_android
            build_causallm_library
            prepare_android_project
            build_android_app
            ;;
        *)
            print_error "Unknown command: $command"
            print_info "Usage: $0 [clean|build|install]"
            exit 1
            ;;
    esac
}

main "$@"