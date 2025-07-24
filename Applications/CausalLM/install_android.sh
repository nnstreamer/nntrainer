#!/bin/bash

# Installation script for CausalLM Android application
set -e

# Configuration
INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
MODEL_DIR="$INSTALL_DIR/models"

# Set SCRIPT_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected. Please connect a device and try again."
    exit 1
fi

# Check if build was successful
if [ ! -f "$SCRIPT_DIR/jni/libs/arm64-v8a/nntrainer_causallm" ]; then
    echo "Error: nntrainer_causallm not found. Please run build_android.sh first."
    exit 1
fi

echo "Installing CausalLM to Android device..."

# Create directories on device
adb shell "mkdir -p $INSTALL_DIR"
adb shell "mkdir -p $MODEL_DIR"

# Push executable
echo "Pushing executable..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/nntrainer_causallm" $INSTALL_DIR/
adb shell "chmod 755 $INSTALL_DIR/nntrainer_causallm"

# Push shared libraries
echo "Pushing shared libraries..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libnntrainer.so" $INSTALL_DIR/
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libccapi-nntrainer.so" $INSTALL_DIR/
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libc++_shared.so" $INSTALL_DIR/

# Create run script on device
adb shell "cat > $INSTALL_DIR/run_causallm.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
cd $INSTALL_DIR
./nntrainer_causallm \$@
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