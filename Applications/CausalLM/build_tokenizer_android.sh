#!/bin/bash

# Script to build tokenizers library for Android
set -e

echo "Building tokenizers library for Android arm64-v8a..."

# Check prerequisites
if ! command -v rustc &> /dev/null; then
    echo "Error: Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set. Please set it to your Android NDK path."
    exit 1
fi

# Clone tokenizers repository if not exists
if [ ! -d "tokenizers" ]; then
    echo "Cloning tokenizers repository..."
    git clone https://github.com/huggingface/tokenizers.git
fi

cd tokenizers

# Install Android targets for Rust
echo "Installing Android targets for Rust..."
rustup target add aarch64-linux-android

# Set up cargo config for Android
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[target.aarch64-linux-android]
ar = "$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
linker = "$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang"

[build]
target = "aarch64-linux-android"
EOF

# Build tokenizers C library
echo "Building tokenizers C library..."
cd tokenizers-c

# Set environment variables
export CC_aarch64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang"
export CXX_aarch64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
export AR_aarch64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang"

# Build the library
cargo build --release --target aarch64-linux-android

# Copy the built library
echo "Copying built library..."
mkdir -p ../../lib
cp target/aarch64-linux-android/release/libtokenizers_c.a ../../lib/

echo "Build completed successfully!"
echo "Library copied to: Applications/CausalLM/lib/libtokenizers_c.a"