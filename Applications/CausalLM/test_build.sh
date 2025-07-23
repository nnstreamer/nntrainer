#!/bin/bash

# Simple test build script for CausalLM
# This script tests if the basic build structure works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/test_build"

echo "=== CausalLM Test Build ==="

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Copy files for test build
echo "Setting up test build..."
cp "$SCRIPT_DIR/main.cpp" .
cp "$SCRIPT_DIR/meson_android.build" meson.build

# Create minimal directories
mkdir -p layers lib/android

echo "Running test build..."
meson setup build_test -Dbuildtype=debug
cd build_test
meson compile

if [ -f "nntr_causallm_android" ]; then
    echo "✅ Test build successful!"
    echo "Testing executable..."
    ./nntr_causallm_android test_mode
    echo "✅ Executable runs successfully!"
else
    echo "❌ Test build failed - executable not found"
    exit 1
fi

echo "=== Test completed successfully ==="