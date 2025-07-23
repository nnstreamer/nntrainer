#!/bin/bash

echo "========================================"
echo "CausalLM Android Build Setup Verification"
echo "========================================"

# Check environment variables
echo "Checking environment variables..."

if [ -z "$ANDROID_NDK" ]; then
    echo "❌ ANDROID_NDK is not set"
    echo "   Please set: export ANDROID_NDK=/path/to/your/android-ndk"
    exit 1
else
    echo "✅ ANDROID_NDK: $ANDROID_NDK"
    if [ ! -d "$ANDROID_NDK" ]; then
        echo "❌ ANDROID_NDK directory does not exist: $ANDROID_NDK"
        exit 1
    fi
fi

# Check required files
echo ""
echo "Checking required files..."

SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
NNTRAINER_ROOT="${SCRIPT_DIR}/../../.."

# Check package_android.sh
if [ -f "${NNTRAINER_ROOT}/tools/package_android.sh" ]; then
    echo "✅ package_android.sh found"
else
    echo "❌ package_android.sh not found at: ${NNTRAINER_ROOT}/tools/package_android.sh"
    exit 1
fi

# Check JNI files
JNI_DIR="${SCRIPT_DIR}/app/src/main/jni"
required_files=(
    "Android.mk"
    "Application.mk"
    "causallm_jni.h"
    "causallm_jni.cpp"
    "prepare_android_deps.sh"
)

for file in "${required_files[@]}"; do
    if [ -f "${JNI_DIR}/${file}" ]; then
        echo "✅ ${file} found"
    else
        echo "❌ ${file} not found in JNI directory"
        exit 1
    fi
done

# Check Android project files
android_files=(
    "app/build.gradle"
    "build.gradle"
    "settings.gradle"
    "gradle.properties"
    "gradlew"
)

for file in "${android_files[@]}"; do
    if [ -f "${SCRIPT_DIR}/${file}" ]; then
        echo "✅ ${file} found"
    else
        echo "❌ ${file} not found"
        exit 1
    fi
done

# Check NDK tools
echo ""
echo "Checking NDK tools..."

if [ -f "$ANDROID_NDK/ndk-build" ]; then
    echo "✅ ndk-build found"
else
    echo "❌ ndk-build not found in ANDROID_NDK"
    exit 1
fi

# Check Java files
echo ""
echo "Checking Java source files..."

if [ -f "${SCRIPT_DIR}/app/src/main/java/com/applications/causallmjni/MainActivity.java" ]; then
    echo "✅ MainActivity.java found"
else
    echo "❌ MainActivity.java not found"
    exit 1
fi

# Check Android resources
if [ -f "${SCRIPT_DIR}/app/src/main/res/layout/activity_main.xml" ]; then
    echo "✅ activity_main.xml found"
else
    echo "❌ activity_main.xml not found"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ All checks passed!"
echo "========================================"
echo ""
echo "Build setup is ready. You can now run:"
echo "  ./build_causallm_android.sh"
echo ""
echo "Or build manually:"
echo "  1. ${NNTRAINER_ROOT}/tools/package_android.sh"
echo "  2. cd app/src/main/jni && ./prepare_android_deps.sh"
echo "  3. \$ANDROID_NDK/ndk-build"
echo "  4. ./gradlew assembleDebug"