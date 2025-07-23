#!/bin/bash

# Installation script for CausalLM Android application
set -e

# Configuration
PACKAGE_NAME="org.nnstreamer.nntrainer.causallm"
INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
MODEL_DIR="$INSTALL_DIR/models"
LIB_DIR="/system/lib64"

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected. Please connect a device and try again."
    exit 1
fi

# Check if build was successful
if [ ! -f "libs/arm64-v8a/nntr_causallm" ]; then
    echo "Error: nntr_causallm not found. Please run build_android.sh first."
    exit 1
fi

echo "Installing CausalLM to Android device..."

# Create directories on device
adb shell "mkdir -p $INSTALL_DIR"
adb shell "mkdir -p $MODEL_DIR"

# Push executable
echo "Pushing executable..."
adb push libs/arm64-v8a/nntr_causallm $INSTALL_DIR/
adb shell "chmod 755 $INSTALL_DIR/nntr_causallm"

# Push shared libraries
echo "Pushing shared libraries..."
adb push libs/arm64-v8a/libcausallm.so $INSTALL_DIR/
adb push libs/arm64-v8a/libnntrainer.so $INSTALL_DIR/
adb push libs/arm64-v8a/libccapi-nntrainer.so $INSTALL_DIR/

# Push C++ STL library
if [ -f "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" ]; then
    adb push $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $INSTALL_DIR/
fi

# Create run script on device
adb shell "cat > $INSTALL_DIR/run_causallm.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
cd $INSTALL_DIR
./nntr_causallm \$@
EOF"

adb shell "chmod 755 $INSTALL_DIR/run_causallm.sh"

echo "Installation completed!"
echo ""
echo "To run CausalLM on the device:"
echo "1. Push your model files to: $MODEL_DIR/"
echo "   Example: adb push res/qwen3-4b/* $MODEL_DIR/qwen3-4b/"
echo ""
echo "2. Run the application:"
echo "   adb shell $INSTALL_DIR/run_causallm.sh $MODEL_DIR/qwen3-4b"
echo ""
echo "For interactive shell:"
echo "   adb shell"
echo "   cd $INSTALL_DIR"
echo "   ./run_causallm.sh $MODEL_DIR/qwen3-4b"