#!/bin/bash

# Test script for CausalLM Android installation
set -e

INSTALL_DIR="/data/local/tmp/nntrainer/causallm"

echo "Testing CausalLM Android installation..."

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected."
    exit 1
fi

# Check if executable exists
echo "Checking executable..."
if adb shell "[ -f $INSTALL_DIR/nntrainer_causallm ]"; then
    echo "✓ Executable found"
else
    echo "✗ Executable not found"
    exit 1
fi

# Check libraries
echo "Checking libraries..."
for lib in libnntrainer.so libccapi-nntrainer.so libc++_shared.so; do
    if adb shell "[ -f $INSTALL_DIR/$lib ]"; then
        echo "✓ $lib found"
    else
        echo "✗ $lib not found"
        exit 1
    fi
done

# Check executable permissions
echo "Checking permissions..."
if adb shell "[ -x $INSTALL_DIR/nntrainer_causallm ]"; then
    echo "✓ Executable has execute permissions"
else
    echo "✗ Executable missing execute permissions"
    exit 1
fi

# Try to run with --help (if supported)
echo "Testing execution..."
adb shell "cd $INSTALL_DIR && LD_LIBRARY_PATH=$INSTALL_DIR ./nntrainer_causallm --help 2>&1 || echo 'Note: --help flag might not be supported'"

echo ""
echo "Installation test completed!"
echo "To run with a model, use:"
echo "  adb shell $INSTALL_DIR/run_causallm.sh /path/to/model"