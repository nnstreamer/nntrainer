#!/bin/bash

set -e

# Script to build CausalLM Android application
# This script first calls package_android.sh and then builds the CausalLM Android app

SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
NNTRAINER_ROOT="${SCRIPT_DIR}/../../.."

echo "========================================"
echo "Building CausalLM Android Application"
echo "========================================"

# Step 1: Build nntrainer for Android
echo "Step 1: Building nntrainer for Android..."
cd "${NNTRAINER_ROOT}"

if [ ! -f "${NNTRAINER_ROOT}/tools/package_android.sh" ]; then
    echo "Error: package_android.sh not found in tools directory"
    exit 1
fi

# Run package_android.sh to build nntrainer for Android
echo "Running package_android.sh..."
bash "${NNTRAINER_ROOT}/tools/package_android.sh" "${NNTRAINER_ROOT}"

if [ ! -f "${NNTRAINER_ROOT}/nntrainer_for_android.tar.gz" ]; then
    echo "Error: nntrainer_for_android.tar.gz was not created"
    exit 1
fi

echo "nntrainer for Android built successfully"

# Step 2: Prepare Android dependencies for CausalLM
echo "Step 2: Preparing Android dependencies for CausalLM..."
cd "${SCRIPT_DIR}/app/src/main/jni"

if [ ! -f "prepare_android_deps.sh" ]; then
    echo "Error: prepare_android_deps.sh not found"
    exit 1
fi

echo "Running prepare_android_deps.sh..."
bash prepare_android_deps.sh

echo "Android dependencies prepared successfully"

# Step 3: Build CausalLM JNI
echo "Step 3: Building CausalLM JNI..."
cd "${SCRIPT_DIR}/app/src/main/jni"

# Check if NDK is available
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK environment variable is not set"
    echo "Please set ANDROID_NDK to point to your Android NDK installation"
    exit 1
fi

if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK directory does not exist: $ANDROID_NDK"
    exit 1
fi

echo "Using Android NDK: $ANDROID_NDK"

# Build using ndk-build
echo "Running ndk-build..."
$ANDROID_NDK/ndk-build

if [ $? -eq 0 ]; then
    echo "CausalLM JNI built successfully"
else
    echo "Error: CausalLM JNI build failed"
    exit 1
fi

# Step 4: Build Android APK (optional)
echo "Step 4: Building Android APK..."
cd "${SCRIPT_DIR}"

if [ -f "gradlew" ]; then
    echo "Running gradlew assembleDebug..."
    ./gradlew assembleDebug
    
    if [ $? -eq 0 ]; then
        echo "Android APK built successfully"
        echo "APK location: ${SCRIPT_DIR}/app/build/outputs/apk/debug/"
    else
        echo "Warning: Android APK build failed, but JNI build was successful"
    fi
else
    echo "Warning: gradlew not found, skipping APK build"
    echo "You can manually build the APK using Android Studio or gradle"
fi

echo "========================================"
echo "CausalLM Android build completed!"
echo "========================================"
echo "JNI libraries are available in: ${SCRIPT_DIR}/app/src/main/libs/"
echo "You can now use Android Studio to complete the application development"