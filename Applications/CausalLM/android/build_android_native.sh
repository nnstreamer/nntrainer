#!/bin/bash

set -e

# CausalLM Android Native Build Script
# This script builds CausalLM main.cpp as a native Android executable

SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
CAUSALLM_ROOT="${SCRIPT_DIR}/.."
NNTRAINER_ROOT="${SCRIPT_DIR}/../../.."

echo "========================================"
echo "Building CausalLM for Android (Native)"
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

# Build target configuration
TARGET_ARCH="aarch64-linux-android"
API_LEVEL="30"
TOOLCHAIN_PREFIX="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64"
CC="${TOOLCHAIN_PREFIX}/bin/${TARGET_ARCH}${API_LEVEL}-clang"
CXX="${TOOLCHAIN_PREFIX}/bin/${TARGET_ARCH}${API_LEVEL}-clang++"
AR="${TOOLCHAIN_PREFIX}/bin/llvm-ar"
STRIP="${TOOLCHAIN_PREFIX}/bin/llvm-strip"

echo "Target Architecture: ${TARGET_ARCH}"
echo "API Level: ${API_LEVEL}"

# Step 1: Build nntrainer for Android
echo ""
echo "Step 1: Building nntrainer for Android..."
cd "${NNTRAINER_ROOT}"

if [ ! -f "${NNTRAINER_ROOT}/tools/package_android.sh" ]; then
    echo "Error: package_android.sh not found"
    exit 1
fi

# Run package_android.sh to build nntrainer
echo "Running package_android.sh..."
bash "${NNTRAINER_ROOT}/tools/package_android.sh" "${NNTRAINER_ROOT}"

if [ ! -f "${NNTRAINER_ROOT}/nntrainer_for_android.tar.gz" ]; then
    echo "Error: nntrainer_for_android.tar.gz was not created"
    exit 1
fi

echo "nntrainer for Android built successfully"

# Step 2: Extract Android dependencies
echo ""
echo "Step 2: Extracting Android dependencies..."
cd "${SCRIPT_DIR}"

if [ -d "android_deps" ]; then
    echo "Removing existing android_deps directory"
    rm -rf android_deps
fi

echo "Extracting nntrainer_for_android.tar.gz"
mkdir -p android_deps
cd android_deps
tar -xzf "${NNTRAINER_ROOT}/nntrainer_for_android.tar.gz"

ANDROID_DEPS_DIR="${SCRIPT_DIR}/android_deps"
NNTRAINER_INCLUDE="${ANDROID_DEPS_DIR}/include/nntrainer"
NNTRAINER_LIB="${ANDROID_DEPS_DIR}/lib/arm64-v8a"

echo "Dependencies extracted to: ${ANDROID_DEPS_DIR}"

# Step 3: Build CausalLM native executable
echo ""
echo "Step 3: Building CausalLM native executable..."

if [ ! -f "${CAUSALLM_ROOT}/main.cpp" ]; then
    echo "Error: main.cpp not found in CausalLM directory"
    echo "Please ensure CausalLM sources from PR #3344 are available"
    exit 1
fi

# Create build directory
BUILD_DIR="${SCRIPT_DIR}/build"
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

echo "Compiling CausalLM main.cpp..."

# Compiler flags
CXXFLAGS="-std=c++17 -O3 -DANDROID -fPIC"
CXXFLAGS="$CXXFLAGS -I${NNTRAINER_INCLUDE}"
CXXFLAGS="$CXXFLAGS -I${CAUSALLM_ROOT}"
CXXFLAGS="$CXXFLAGS -I${CAUSALLM_ROOT}/layers"
CXXFLAGS="$CXXFLAGS -fopenmp -static-openmp"

# Linker flags
LDFLAGS="-L${NNTRAINER_LIB}"
LDFLAGS="$LDFLAGS -lnntrainer -lccapi-nntrainer -lcausallm"
LDFLAGS="$LDFLAGS -fopenmp -static-openmp"
LDFLAGS="$LDFLAGS -static-libgcc -static-libstdc++"

# Source files
SOURCES="${CAUSALLM_ROOT}/main.cpp"

# Additional CausalLM source files if needed
if [ -f "${CAUSALLM_ROOT}/causal_lm.cpp" ]; then
    SOURCES="$SOURCES ${CAUSALLM_ROOT}/causal_lm.cpp"
fi
if [ -f "${CAUSALLM_ROOT}/llm_util.cpp" ]; then
    SOURCES="$SOURCES ${CAUSALLM_ROOT}/llm_util.cpp"
fi

echo "Using compiler: $CXX"
echo "Compiler flags: $CXXFLAGS"
echo "Linker flags: $LDFLAGS"
echo "Source files: $SOURCES"

# Compile
$CXX $CXXFLAGS $SOURCES $LDFLAGS -o nntr_causallm_android

if [ $? -eq 0 ]; then
    echo "✅ CausalLM compiled successfully!"
else
    echo "❌ Compilation failed"
    exit 1
fi

# Strip binary to reduce size
echo "Stripping binary..."
$STRIP nntr_causallm_android

# Check binary
echo ""
echo "Binary information:"
ls -lh nntr_causallm_android
file nntr_causallm_android

# Step 4: Create deployment package
echo ""
echo "Step 4: Creating deployment package..."

DEPLOY_DIR="${SCRIPT_DIR}/deploy"
if [ -d "$DEPLOY_DIR" ]; then
    rm -rf "$DEPLOY_DIR"
fi
mkdir -p "$DEPLOY_DIR"

# Copy binary
cp nntr_causallm_android "$DEPLOY_DIR/"

# Copy required libraries
echo "Copying required libraries..."
cp "${NNTRAINER_LIB}"/*.so "$DEPLOY_DIR/" 2>/dev/null || true

# Create deployment script
cat > "$DEPLOY_DIR/deploy_to_device.sh" << 'EOF'
#!/bin/bash

# Deploy CausalLM to Android device
# Usage: ./deploy_to_device.sh [device_path]

DEVICE_PATH=${1:-/data/local/tmp/causallm}

echo "Deploying CausalLM to Android device..."
echo "Target path: $DEVICE_PATH"

# Check if adb is available
if ! command -v adb &> /dev/null; then
    echo "Error: adb not found. Please install Android SDK platform-tools"
    exit 1
fi

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No Android device connected"
    echo "Please connect device and enable USB debugging"
    exit 1
fi

# Create directory on device
echo "Creating directory on device..."
adb shell "mkdir -p $DEVICE_PATH"

# Push binary
echo "Pushing binary..."
adb push nntr_causallm_android "$DEVICE_PATH/"
adb shell "chmod +x $DEVICE_PATH/nntr_causallm_android"

# Push libraries
echo "Pushing libraries..."
for lib in *.so; do
    if [ -f "$lib" ]; then
        adb push "$lib" "$DEVICE_PATH/"
    fi
done

# Create run script on device
echo "Creating run script..."
adb shell "cat > $DEVICE_PATH/run_causallm.sh << 'EOFSCRIPT'
#!/system/bin/sh
cd $DEVICE_PATH
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./nntr_causallm_android \$@
EOFSCRIPT"

adb shell "chmod +x $DEVICE_PATH/run_causallm.sh"

echo ""
echo "✅ Deployment completed!"
echo ""
echo "To run CausalLM on device:"
echo "  adb shell"
echo "  cd $DEVICE_PATH"
echo "  ./run_causallm.sh /path/to/model/directory"
echo ""
echo "Or directly:"
echo "  adb shell '$DEVICE_PATH/run_causallm.sh /path/to/model/directory'"
EOF

chmod +x "$DEPLOY_DIR/deploy_to_device.sh"

# Create usage instructions
cat > "$DEPLOY_DIR/README.md" << 'EOF'
# CausalLM Android Native Deployment

## Files
- `nntr_causallm_android` - Main executable
- `*.so` - Required shared libraries
- `deploy_to_device.sh` - Deployment script
- `README.md` - This file

## Usage

### 1. Deploy to Device
```bash
./deploy_to_device.sh [target_path]
```
Default target path: `/data/local/tmp/causallm`

### 2. Prepare Model Files
Copy your CausalLM model files to device:
```bash
adb push /path/to/model/directory /sdcard/causallm_models/
```

### 3. Run CausalLM
```bash
adb shell
cd /data/local/tmp/causallm
./run_causallm.sh /sdcard/causallm_models/your_model
```

## Model Directory Structure
Your model directory should contain:
- config.json
- generation_config.json  
- nntr_config.json
- tokenizer.json
- *.bin (weight file)

## Troubleshooting
- Ensure device has sufficient storage and memory
- Check that all model files are present
- Verify library dependencies are satisfied
EOF

echo ""
echo "========================================"
echo "✅ Build completed successfully!"
echo "========================================"
echo ""
echo "Build artifacts:"
echo "  Binary: ${BUILD_DIR}/nntr_causallm_android"
echo "  Deploy package: ${DEPLOY_DIR}/"
echo ""
echo "Next steps:"
echo "  1. cd ${DEPLOY_DIR}"
echo "  2. ./deploy_to_device.sh"
echo "  3. Copy model files to device"
echo "  4. Run on device: adb shell '/data/local/tmp/causallm/run_causallm.sh /path/to/model'"