#!/bin/bash

# CausalLM Android Build Script
# This script builds CausalLM for Android independently from the main build system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_android"
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
    local toolchain_prefix=""
    local target_triple=""
    
    case "$ABI" in
        arm64-v8a)
            toolchain_prefix="aarch64-linux-android"
            target_triple="aarch64-linux-android"
            ;;
        armeabi-v7a)
            toolchain_prefix="armv7a-linux-androideabi"
            target_triple="arm-linux-androideabi"
            ;;
        x86)
            toolchain_prefix="i686-linux-android"
            target_triple="i686-linux-android"
            ;;
        x86_64)
            toolchain_prefix="x86_64-linux-android"
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
    echo_info "Building CausalLM executable for Android..."
    echo_info "ABI: $ABI, API Level: $API_LEVEL"
    
    local cross_file="$BUILD_DIR/android-cross-file.txt"
    local android_meson_file="$SCRIPT_DIR/meson_android.build"
    
    # Check if Android meson file exists
    if [ ! -f "$android_meson_file" ]; then
        echo_error "Android meson build file not found: $android_meson_file"
        exit 1
    fi
    
    # Configure meson for Android using the separate build file
    cd "$BUILD_DIR"
    
    # Copy the Android meson file as meson.build in build directory
    cp "$android_meson_file" "$BUILD_DIR/meson.build"
    
    # Copy source files to build directory
    echo_info "Copying source files..."
    cp "$SCRIPT_DIR/main.cpp" "$BUILD_DIR/" 2>/dev/null || true
    cp -r "$SCRIPT_DIR"/*.h "$BUILD_DIR/" 2>/dev/null || true
    cp -r "$SCRIPT_DIR/layers" "$BUILD_DIR/" 2>/dev/null || true
    cp -r "$SCRIPT_DIR/lib" "$BUILD_DIR/" 2>/dev/null || true
    
    # Setup meson build
    meson setup build_temp \
        --cross-file="$cross_file" \
        -Dbuildtype=release
    
    # Build
    cd build_temp
    meson compile
    
    echo_info "Android executable build completed successfully!"
    echo_info "Executable location: $BUILD_DIR/build_temp/nntr_causallm_android"
}

# Copy executable and create package
package_android() {
    echo_info "Packaging Android executable..."
    
    local package_dir="$BUILD_DIR/package"
    mkdir -p "$package_dir/bin"
    
    # Copy executable
    local exe_path="$BUILD_DIR/build_temp/nntr_causallm_android"
    if [ -f "$exe_path" ]; then
        cp "$exe_path" "$package_dir/bin/"
        echo_info "Executable copied to package"
    else
        echo_warn "Executable not found at: $exe_path"
    fi
    
    # Copy any shared libraries if they exist
    find "$BUILD_DIR/build_temp" -name "*.so" -exec cp {} "$package_dir/bin/" \; 2>/dev/null || true
    
    echo_info "Android package created in: $package_dir"
    echo_info "To deploy to Android device:"
    echo_info "  adb push $package_dir/bin/nntr_causallm_android /data/local/tmp/"
    echo_info "  adb shell chmod +x /data/local/tmp/nntr_causallm_android"
    echo_info "  adb shell /data/local/tmp/nntr_causallm_android"
}

# Main execution
main() {
    echo_info "Starting CausalLM Android build..."
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