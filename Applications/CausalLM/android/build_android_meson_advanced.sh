#!/bin/bash

set -e

# Advanced CausalLM Android Build Script using Meson with Cross-compilation
# This script fully leverages Meson's cross-compilation capabilities

SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
CAUSALLM_ROOT="${SCRIPT_DIR}/.."
NNTRAINER_ROOT="${SCRIPT_DIR}/../../.."

echo "========================================"
echo "Advanced CausalLM Android Build (Meson)"
echo "========================================"

# Check environment variables
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK environment variable is not set"
    echo "Please set: export ANDROID_NDK=/path/to/your/android-ndk"
    exit 1
fi

if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK directory does not exist: $ANDROID_NDK"
    exit 1
fi

echo "Using Android NDK: $ANDROID_NDK"
echo "NNTrainer Root: $NNTRAINER_ROOT"

# Default configuration
CLEAN_BUILD=false
INSTALL_APPS=true
ENABLE_CAUSALLM=true
BUILD_TYPE="release"
API_LEVEL="30"
ARCH="aarch64"
MESON_ARGS=""
CROSS_FILE=""
USE_CROSS_FILE=true
VERBOSE=false
PARALLEL_JOBS=$(nproc)

# Parse command line arguments
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --clean                 Clean previous build
    --debug                 Build in debug mode
    --no-install           Don't install applications
    --disable-causallm     Disable CausalLM build
    --api-level LEVEL      Android API level (default: 30)
    --arch ARCH            Target architecture (default: aarch64)
    --no-cross-file        Don't use cross-compilation file
    --verbose              Enable verbose output
    --jobs N               Number of parallel jobs (default: $(nproc))
    -D option=value        Pass option to Meson
    --help                 Show this help

EXAMPLES:
    $0                                    # Basic build
    $0 --clean --debug                    # Clean debug build
    $0 --api-level 29 --arch armv7        # Build for API 29, ARMv7
    $0 -Denable-opencl=false              # Disable OpenCL
    $0 --verbose --jobs 8                 # Verbose build with 8 jobs

SUPPORTED ARCHITECTURES:
    aarch64 (arm64-v8a)    - Default, recommended
    armv7 (armeabi-v7a)    - 32-bit ARM
    x86_64                 - Intel 64-bit (emulator)
    x86                    - Intel 32-bit (emulator)
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            BUILD_TYPE="debug"
            shift
            ;;
        --no-install)
            INSTALL_APPS=false
            shift
            ;;
        --disable-causallm)
            ENABLE_CAUSALLM=false
            shift
            ;;
        --api-level)
            API_LEVEL="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --no-cross-file)
            USE_CROSS_FILE=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -D*)
            MESON_ARGS="$MESON_ARGS $1"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Architecture mapping
case $ARCH in
    aarch64|arm64|arm64-v8a)
        ARCH="aarch64"
        ANDROID_ABI="arm64-v8a"
        TOOLCHAIN_PREFIX="aarch64-linux-android"
        ;;
    armv7|arm|armeabi-v7a)
        ARCH="armv7"
        ANDROID_ABI="armeabi-v7a"
        TOOLCHAIN_PREFIX="armv7a-linux-androideabi"
        ;;
    x86_64)
        ARCH="x86_64"
        ANDROID_ABI="x86_64"
        TOOLCHAIN_PREFIX="x86_64-linux-android"
        ;;
    x86|i686)
        ARCH="x86"
        ANDROID_ABI="x86"
        TOOLCHAIN_PREFIX="i686-linux-android"
        ;;
    *)
        echo "Error: Unsupported architecture: $ARCH"
        echo "Supported: aarch64, armv7, x86_64, x86"
        exit 1
        ;;
esac

echo "Build Configuration:"
echo "  Architecture: $ARCH ($ANDROID_ABI)"
echo "  API Level: $API_LEVEL"
echo "  Build Type: $BUILD_TYPE"
echo "  Parallel Jobs: $PARALLEL_JOBS"
echo "  CausalLM: $ENABLE_CAUSALLM"

# Go to nntrainer root
cd "$NNTRAINER_ROOT"

# Clean previous build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo ""
    echo "Cleaning previous build..."
    if [ -d "builddir" ]; then
        rm -rf builddir
    fi
    if [ -f "nntrainer_for_android.tar.gz" ]; then
        rm -f "nntrainer_for_android.tar.gz"
    fi
fi

# Step 1: Generate cross-compilation file
if [ "$USE_CROSS_FILE" = true ]; then
    echo ""
    echo "Step 1: Generating Meson cross-compilation file..."
    
    CROSS_FILE="$SCRIPT_DIR/android-cross-${ARCH}-api${API_LEVEL}.txt"
    
    # Generate cross-compilation file
    cat > "$CROSS_FILE" << EOF
[binaries]
c = '$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/${TOOLCHAIN_PREFIX}${API_LEVEL}-clang'
cpp = '$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/${TOOLCHAIN_PREFIX}${API_LEVEL}-clang++'
ar = '$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar'
strip = '$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip'
objcopy = '$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objcopy'
ld = '$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/ld'
pkgconfig = 'pkg-config'

[host_machine]
system = 'android'
cpu_family = '$([ "$ARCH" = "aarch64" ] && echo "aarch64" || [ "$ARCH" = "armv7" ] && echo "arm" || echo "$ARCH")'
cpu = '$([ "$ARCH" = "aarch64" ] && echo "armv8" || [ "$ARCH" = "armv7" ] && echo "armv7" || echo "$ARCH")'
endian = 'little'

[target_machine]
system = 'android'
cpu_family = '$([ "$ARCH" = "aarch64" ] && echo "aarch64" || [ "$ARCH" = "armv7" ] && echo "arm" || echo "$ARCH")'
cpu = '$([ "$ARCH" = "aarch64" ] && echo "armv8" || [ "$ARCH" = "armv7" ] && echo "armv7" || echo "$ARCH")'
endian = 'little'

[properties]
# Android-specific properties
android_api_level = '$API_LEVEL'
android_ndk_api_level = '$API_LEVEL'
android_abi = '$ANDROID_ABI'

# Compiler and linker flags
c_args = ['-DANDROID', '-fPIC', '-ffunction-sections', '-funwind-tables', '-fstack-protector-strong', '-no-canonical-prefixes']
cpp_args = ['-DANDROID', '-fPIC', '-ffunction-sections', '-funwind-tables', '-fstack-protector-strong', '-no-canonical-prefixes', '-std=c++17', '-DPLUGGABLE']
c_link_args = ['-Wl,--exclude-libs,libgcc.a', '-Wl,--exclude-libs,libatomic.a', '-static-libstdc++', '-Wl,--build-id', '-Wl,--warn-shared-textrel', '-Wl,--fatal-warnings']
cpp_link_args = ['-Wl,--exclude-libs,libgcc.a', '-Wl,--exclude-libs,libatomic.a', '-static-libstdc++', '-Wl,--build-id', '-Wl,--warn-shared-textrel', '-Wl,--fatal-warnings']

# Additional Android NDK paths
sys_root = '$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot'
EOF

    echo "✅ Cross-compilation file generated: $CROSS_FILE"
    
    if [ "$VERBOSE" = true ]; then
        echo "Cross-file contents:"
        cat "$CROSS_FILE"
    fi
fi

# Step 2: Configure Meson for Android with CausalLM
echo ""
echo "Step 2: Configuring Meson build for Android..."

# Base meson configuration for Android
MESON_CONFIG=(
    -Dplatform=android
    -Dopenblas-num-threads=1
    -Denable-tflite-interpreter=false
    -Denable-tflite-backbone=false
    -Denable-fp16=true
    -Domp-num-threads=1
    -Denable-opencl=true
    -Dhgemm-experimental-kernel=false
    -Denable-ggml=true
    -Denable-app=$INSTALL_APPS
    -Dinstall-app=$INSTALL_APPS
    -Denable-causallm-app=$ENABLE_CAUSALLM
    --buildtype=$BUILD_TYPE
)

# Add cross-compilation file if enabled
if [ "$USE_CROSS_FILE" = true ] && [ -n "$CROSS_FILE" ]; then
    MESON_CONFIG+=(--cross-file="$CROSS_FILE")
fi

# Add user-provided meson arguments
if [ -n "$MESON_ARGS" ]; then
    echo "Additional Meson arguments: $MESON_ARGS"
    MESON_CONFIG+=($MESON_ARGS)
fi

# Configure or reconfigure build
if [ ! -d builddir ]; then
    echo "Creating new build directory..."
    if [ "$VERBOSE" = true ]; then
        echo "Meson setup command: meson setup builddir ${MESON_CONFIG[*]}"
    fi
    meson setup builddir "${MESON_CONFIG[@]}"
else
    echo "Reconfiguring existing build directory..."
    if [ "$VERBOSE" = true ]; then
        echo "Meson configure command: meson configure builddir ${MESON_CONFIG[*]}"
    fi
    meson configure builddir "${MESON_CONFIG[@]}"
    # Wipe build if configuration changed significantly
    meson setup --wipe builddir
fi

if [ $? -ne 0 ]; then
    echo "❌ Meson configuration failed"
    exit 1
fi

echo "✅ Meson configuration completed"

# Show build information
if [ "$VERBOSE" = true ]; then
    echo ""
    echo "Build information:"
    meson introspect builddir --buildoptions | grep -E "(platform|buildtype|enable-causallm-app)" || true
fi

# Step 3: Build with Meson
echo ""
echo "Step 3: Building with Meson..."

cd builddir

# Build everything with parallel jobs
echo "Running meson compile with $PARALLEL_JOBS jobs..."
if [ "$VERBOSE" = true ]; then
    meson compile -j "$PARALLEL_JOBS" -v
else
    meson compile -j "$PARALLEL_JOBS"
fi

if [ $? -ne 0 ]; then
    echo "❌ Meson build failed"
    exit 1
fi

echo "✅ Meson build completed"

# Step 4: Install to staging area
echo ""
echo "Step 4: Installing to staging area..."

echo "Running meson install..."
if [ "$VERBOSE" = true ]; then
    meson install --destdir android_build_result -v
else
    meson install --destdir android_build_result
fi

if [ $? -ne 0 ]; then
    echo "❌ Meson install failed"
    exit 1
fi

echo "✅ Meson install completed"

# Step 5: Create deployment package
echo ""
echo "Step 5: Creating deployment package..."

cd "$NNTRAINER_ROOT"

# Create tarball
if [ -f "nntrainer_for_android.tar.gz" ]; then
    rm -f "nntrainer_for_android.tar.gz"
fi

echo "Creating nntrainer_for_android.tar.gz..."
tar -czvf nntrainer_for_android.tar.gz --directory=builddir/android_build_result .

if [ $? -ne 0 ]; then
    echo "❌ Failed to create deployment tarball"
    exit 1
fi

echo "✅ Deployment tarball created: nntrainer_for_android.tar.gz"

# Step 6: Extract and prepare CausalLM deployment
echo ""
echo "Step 6: Preparing CausalLM deployment..."

DEPLOY_DIR="${SCRIPT_DIR}/deploy_meson_${ARCH}"
if [ -d "$DEPLOY_DIR" ]; then
    rm -rf "$DEPLOY_DIR"
fi
mkdir -p "$DEPLOY_DIR"

cd "$DEPLOY_DIR"

# Extract the built artifacts
echo "Extracting build artifacts..."
tar -xzf "$NNTRAINER_ROOT/nntrainer_for_android.tar.gz"

# Find CausalLM executable
CAUSALLM_EXE=""
SEARCH_PATHS=("bin" "Applications" "usr/local/bin" ".")

for path in "${SEARCH_PATHS[@]}"; do
    if [ -f "$path/nntr_causallm" ]; then
        CAUSALLM_EXE="$path/nntr_causallm"
        break
    fi
done

# Fallback: search everywhere
if [ -z "$CAUSALLM_EXE" ]; then
    CAUSALLM_EXE=$(find . -name "nntr_causallm" -type f | head -1)
fi

if [ -z "$CAUSALLM_EXE" ] || [ ! -f "$CAUSALLM_EXE" ]; then
    echo "❌ CausalLM executable not found in build artifacts"
    echo "Available files:"
    find . -name "*causallm*" -type f
    echo ""
    echo "All executables:"
    find . -name "nntr_*" -type f
    exit 1
fi

echo "✅ Found CausalLM executable: $CAUSALLM_EXE"

# Copy executable to deployment root
cp "$CAUSALLM_EXE" "./nntr_causallm_android"
chmod +x "./nntr_causallm_android"

# Copy shared libraries based on architecture
echo "Copying shared libraries for $ANDROID_ABI..."
LIB_COPIED=0

# Try different library paths
LIB_PATHS=("lib/$ANDROID_ABI" "lib" "usr/local/lib" "lib64")

for lib_path in "${LIB_PATHS[@]}"; do
    if [ -d "$lib_path" ]; then
        echo "  Checking $lib_path..."
        for lib in "$lib_path"/*.so; do
            if [ -f "$lib" ]; then
                cp "$lib" .
                echo "    Copied: $(basename $lib)"
                ((LIB_COPIED++))
            fi
        done
    fi
done

if [ $LIB_COPIED -eq 0 ]; then
    echo "⚠️  No shared libraries found"
    echo "Available library directories:"
    find . -name "lib*" -type d
else
    echo "✅ Copied $LIB_COPIED shared libraries"
fi

# Step 7: Create enhanced deployment scripts
echo ""
echo "Step 7: Creating deployment scripts..."

# Create deployment script with architecture info
cat > "deploy_to_device.sh" << EOF
#!/bin/bash

# Meson-built CausalLM Android Deployment Script
# Architecture: $ARCH ($ANDROID_ABI)
# API Level: $API_LEVEL
# Build Type: $BUILD_TYPE

DEVICE_PATH=\${1:-/data/local/tmp/causallm}

echo "========================================"
echo "Deploying CausalLM to Android Device"
echo "========================================"
echo "Architecture: $ARCH ($ANDROID_ABI)"
echo "API Level: $API_LEVEL"
echo "Build Type: $BUILD_TYPE"
echo "Target path: \$DEVICE_PATH"

# Check prerequisites
if ! command -v adb &> /dev/null; then
    echo "❌ adb not found. Please install Android SDK platform-tools"
    exit 1
fi

if ! adb devices | grep -q "device\$"; then
    echo "❌ No Android device connected"
    echo "Please connect device and enable USB debugging"
    exit 1
fi

echo "✅ Device connected"

# Check device architecture compatibility
DEVICE_ARCH=\$(adb shell getprop ro.product.cpu.abi)
echo "Device architecture: \$DEVICE_ARCH"

case "\$DEVICE_ARCH" in
    arm64-v8a|aarch64)
        if [ "$ANDROID_ABI" != "arm64-v8a" ]; then
            echo "⚠️  Warning: Built for $ANDROID_ABI but device is \$DEVICE_ARCH"
        fi
        ;;
    armeabi-v7a)
        if [ "$ANDROID_ABI" != "armeabi-v7a" ]; then
            echo "⚠️  Warning: Built for $ANDROID_ABI but device is \$DEVICE_ARCH"
        fi
        ;;
    x86_64)
        if [ "$ANDROID_ABI" != "x86_64" ]; then
            echo "⚠️  Warning: Built for $ANDROID_ABI but device is \$DEVICE_ARCH"
        fi
        ;;
    x86)
        if [ "$ANDROID_ABI" != "x86" ]; then
            echo "⚠️  Warning: Built for $ANDROID_ABI but device is \$DEVICE_ARCH"
        fi
        ;;
esac

# Deploy files
echo ""
echo "Deploying files..."
adb shell "mkdir -p \$DEVICE_PATH"

echo "  Pushing CausalLM executable..."
if [ -f "nntr_causallm_android" ]; then
    adb push nntr_causallm_android "\$DEVICE_PATH/"
    adb shell "chmod +x \$DEVICE_PATH/nntr_causallm_android"
    echo "    ✅ Executable deployed"
else
    echo "    ❌ Executable not found"
    exit 1
fi

echo "  Pushing shared libraries..."
lib_count=0
for lib in *.so; do
    if [ -f "\$lib" ]; then
        adb push "\$lib" "\$DEVICE_PATH/"
        echo "    Pushed: \$lib"
        ((lib_count++))
    fi
done

if [ \$lib_count -eq 0 ]; then
    echo "    ⚠️  No shared libraries found"
else
    echo "    ✅ \$lib_count libraries deployed"
fi

# Create run script with build info
echo "  Creating run script..."
adb shell "cat > \$DEVICE_PATH/run_causallm.sh << 'EOFSCRIPT'
#!/system/bin/sh

# CausalLM Android Runner (Meson-built)
# Architecture: $ARCH ($ANDROID_ABI)
# API Level: $API_LEVEL
# Build Type: $BUILD_TYPE

CAUSALLM_DIR=\\\$(dirname \\\"\\\$0\\\")
cd \\\"\\\$CAUSALLM_DIR\\\"

# Set library path
export LD_LIBRARY_PATH=\\\".\\\$LD_LIBRARY_PATH\\\"

# Check if model path is provided
if [ \\\$# -eq 0 ]; then
    echo \\\"CausalLM Android Runner (Meson-built)\\\"
    echo \\\"Architecture: $ARCH ($ANDROID_ABI)\\\"
    echo \\\"API Level: $API_LEVEL\\\"
    echo \\\"Build Type: $BUILD_TYPE\\\"
    echo \\\"\\\"
    echo \\\"Usage: \\\$0 <model_directory> [options]\\\"
    echo \\\"Example: \\\$0 /sdcard/causallm_models/qwen3-4b\\\"
    echo \\\"\\\"
    echo \\\"Options:\\\"
    echo \\\"  --help    Show this help\\\"
    echo \\\"  --version Show version info\\\"
    echo \\\"  --info    Show build information\\\"
    exit 1
fi

MODEL_PATH=\\\"\\\$1\\\"
shift

# Handle special options
case \\\"\\\$MODEL_PATH\\\" in
    --help)
        echo \\\"CausalLM Android Runner (Meson-built)\\\"
        echo \\\"Usage: \\\$0 <model_directory> [additional_args]\\\"
        echo \\\"\\\"
        echo \\\"Model directory should contain:\\\"
        echo \\\"  - config.json\\\"
        echo \\\"  - generation_config.json\\\"
        echo \\\"  - nntr_config.json\\\"
        echo \\\"  - tokenizer.json\\\"
        echo \\\"  - *.bin (weight file)\\\"
        exit 0
        ;;
    --version)
        echo \\\"CausalLM Android (Meson-built)\\\"
        ./nntr_causallm_android --version 2>/dev/null || echo \\\"Version info not available\\\"
        exit 0
        ;;
    --info)
        echo \\\"Build Information:\\\"
        echo \\\"  Architecture: $ARCH ($ANDROID_ABI)\\\"
        echo \\\"  API Level: $API_LEVEL\\\"
        echo \\\"  Build Type: $BUILD_TYPE\\\"
        echo \\\"  Build System: Meson\\\"
        echo \\\"  Working Directory: \\\$(pwd)\\\"
        echo \\\"  Library Path: \\\$LD_LIBRARY_PATH\\\"
        exit 0
        ;;
esac

# Check if model directory exists
if [ ! -d \\\"\\\$MODEL_PATH\\\" ]; then
    echo \\\"❌ Error: Model directory not found: \\\$MODEL_PATH\\\"
    echo \\\"\\\"
    echo \\\"Available models:\\\"
    ls -la /sdcard/causallm_models/ 2>/dev/null || echo \\\"No models found in /sdcard/causallm_models/\\\"
    exit 1
fi

echo \\\"🚀 Running CausalLM with model: \\\$MODEL_PATH\\\"
echo \\\"📁 Working directory: \\\$(pwd)\\\"
echo \\\"📚 Library path: \\\$LD_LIBRARY_PATH\\\"
echo \\\"🏗️  Built with: Meson ($ARCH/$API_LEVEL/$BUILD_TYPE)\\\"
echo \\\"\\\"

# Run CausalLM with all remaining arguments
./nntr_causallm_android \\\"\\\$MODEL_PATH\\\" \\\$@
EOFSCRIPT"

adb shell "chmod +x \$DEVICE_PATH/run_causallm.sh"

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📱 Usage on device:"
echo "  adb shell"
echo "  cd \$DEVICE_PATH"
echo "  ./run_causallm.sh /path/to/model"
echo ""
echo "🚀 Quick test:"
echo "  adb shell '\$DEVICE_PATH/run_causallm.sh /sdcard/causallm_models/your_model'"
EOF

chmod +x "deploy_to_device.sh"

# Create build info file
cat > "build_info.txt" << EOF
CausalLM Android Build Information

Build System: Meson (Advanced)
Target Architecture: $ARCH ($ANDROID_ABI)
Android API Level: $API_LEVEL
Build Type: $BUILD_TYPE
Parallel Jobs: $PARALLEL_JOBS
Cross-compilation: $USE_CROSS_FILE
CausalLM Enabled: $ENABLE_CAUSALLM

Build Date: $(date)
Build Host: $(hostname)
NDK Path: $ANDROID_NDK
Meson Version: $(meson --version 2>/dev/null || echo "Unknown")

Files:
- nntr_causallm_android: Main executable
- *.so: Shared libraries ($LIB_COPIED files)
- deploy_to_device.sh: Deployment script
- build_info.txt: This file

Cross-compilation File: $CROSS_FILE
EOF

echo ""
echo "========================================"
echo "✅ Advanced Meson build completed!"
echo "========================================"
echo ""
echo "📊 Build Summary:"
echo "  🔧 Build System: Meson (Advanced)"
echo "  📱 Target: $ARCH ($ANDROID_ABI) API $API_LEVEL"
echo "  🏗️  Build Type: $BUILD_TYPE"
echo "  📦 Executable: $DEPLOY_DIR/nntr_causallm_android"
echo "  📚 Libraries: $LIB_COPIED shared libraries"
echo "  📁 Deploy Package: $DEPLOY_DIR/"
echo "  🔄 Cross-compilation: $USE_CROSS_FILE"
echo ""
echo "🚀 Next Steps:"
echo "  1. cd $DEPLOY_DIR"
echo "  2. ./deploy_to_device.sh"
echo "  3. adb push /path/to/model /sdcard/causallm_models/"
echo "  4. adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/your_model'"
echo ""
echo "📖 Build info: cat $DEPLOY_DIR/build_info.txt"

# Clean up cross-file if generated
if [ "$USE_CROSS_FILE" = true ] && [ -f "$CROSS_FILE" ]; then
    echo ""
    echo "🧹 Cleaning up generated cross-file: $CROSS_FILE"
    rm -f "$CROSS_FILE"
fi