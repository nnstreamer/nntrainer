#!/bin/bash

# CausalLM Android Build Script
# This script builds CausalLM for Android using the main nntrainer build system
# Following reference patch style: ae24db6e9c018a819841f5884defb2c9c1fc3a14

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_android_causallm"
NDK_PATH=${ANDROID_NDK_ROOT:-$ANDROID_NDK}
ABI=${ANDROID_ABI:-arm64-v8a}
API_LEVEL=${ANDROID_API_LEVEL:-21}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."
    
    if [ -z "$NDK_PATH" ]; then
        echo_error "Android NDK not found. Please set ANDROID_NDK_ROOT or ANDROID_NDK environment variable."
        exit 1
    fi
    
    if [ ! -d "$NDK_PATH" ]; then
        echo_error "Android NDK directory not found: $NDK_PATH"
        exit 1
    fi
    
    if ! command -v meson &> /dev/null; then
        echo_error "Meson build system not found. Please install meson."
        exit 1
    fi
    
    echo_info "Prerequisites check passed."
}

# Setup build directory
setup_build_dir() {
    echo_info "Setting up build directory..."
    
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    
    echo_info "Build directory created: $BUILD_DIR"
}

# Create Android cross-compilation file
create_cross_file() {
    echo_info "Creating Android cross-compilation file..."
    
    local cross_file="$BUILD_DIR/android-cross-file.txt"
    local target_triple=""
    
    case "$ABI" in
        arm64-v8a)
            target_triple="aarch64-linux-android"
            ;;
        armeabi-v7a)
            target_triple="armv7a-linux-androideabi"
            ;;
        x86)
            target_triple="i686-linux-android"
            ;;
        x86_64)
            target_triple="x86_64-linux-android"
            ;;
        *)
            echo_error "Unsupported ABI: $ABI"
            exit 1
            ;;
    esac
    
    local toolchain_dir="$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64"
    
    cat > "$cross_file" << EOF
[binaries]
c = '$toolchain_dir/bin/${target_triple}${API_LEVEL}-clang'
cpp = '$toolchain_dir/bin/${target_triple}${API_LEVEL}-clang++'
ar = '$toolchain_dir/bin/llvm-ar'
strip = '$toolchain_dir/bin/llvm-strip'
pkgconfig = 'pkg-config'

[host_machine]
system = 'android'
cpu_family = '$(echo $ABI | cut -d'-' -f1)'
cpu = '$(echo $ABI | cut -d'-' -f1)'
endian = 'little'

[properties]
android_ndk = '$NDK_PATH'
android_api_level = '$API_LEVEL'
android_abi = '$ABI'
EOF
    
    echo_info "Cross-compilation file created: $cross_file"
}

# Build for Android
build_android() {
    echo_info "Building CausalLM for Android..."
    echo_info "ABI: $ABI, API Level: $API_LEVEL"
    
    local cross_file="$BUILD_DIR/android-cross-file.txt"
    
    # Configure meson for Android
    cd "$PROJECT_ROOT"
    meson setup "$BUILD_DIR" \
        --cross-file="$cross_file" \
        -Dplatform=android \
        -Denable-app=true \
        -Denable-test=false \
        -Denable-logging=true \
        -Denable-openmp=true \
        -Dbuildtype=release
    
    # Build
    meson compile -C "$BUILD_DIR"
    
    echo_info "Android build completed successfully!"
    echo_info "Build artifacts are in: $BUILD_DIR"
}

# Package Android build
package_android() {
    echo_info "Packaging Android build..."
    
    local package_dir="$BUILD_DIR/package"
    mkdir -p "$package_dir/bin"
    mkdir -p "$package_dir/lib"
    
    # Find and copy CausalLM executable
    local causallm_exe=$(find "$BUILD_DIR" -name "nntrainer_causallm" -type f | head -1)
    if [ -n "$causallm_exe" ] && [ -f "$causallm_exe" ]; then
        cp "$causallm_exe" "$package_dir/bin/"
        echo_info "CausalLM executable copied to package"
    else
        echo_warn "CausalLM executable not found"
    fi
    
    # Copy shared libraries
    find "$BUILD_DIR" -name "*.so" -exec cp {} "$package_dir/lib/" \; 2>/dev/null || true
    
    echo_info "Android package created in: $package_dir"
    echo_info ""
    echo_info "To deploy to Android device:"
    echo_info "  adb push $package_dir/bin/nntrainer_causallm /data/local/tmp/"
    echo_info "  adb push $package_dir/lib/*.so /data/local/tmp/"
    echo_info "  adb shell chmod +x /data/local/tmp/nntrainer_causallm"
    echo_info "  adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=. ./nntrainer_causallm'"
}

# Main execution
main() {
    echo_info "Starting CausalLM Android build..."
    echo_info "Project root: $PROJECT_ROOT"
    echo_info "Script directory: $SCRIPT_DIR"
    
    check_prerequisites
    setup_build_dir
    create_cross_file
    build_android
    package_android
    
    echo_info "CausalLM Android build completed successfully!"
    echo_info "You can find the build artifacts in: $BUILD_DIR"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        echo "CausalLM Android Build Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Environment Variables:"
        echo "  ANDROID_NDK_ROOT    Path to Android NDK (required)"
        echo "  ANDROID_ABI         Target ABI (default: arm64-v8a)"
        echo "  ANDROID_API_LEVEL   Target API level (default: 21)"
        echo ""
        echo "Supported ABIs: arm64-v8a, armeabi-v7a, x86, x86_64"
        exit 0
        ;;
    clean)
        echo_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        echo_info "Clean completed."
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac