#!/bin/bash

set -e

# Advanced CausalLM Android Native Build Script
# This script builds CausalLM with all dependencies as a standalone Android executable

SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
CAUSALLM_ROOT="${SCRIPT_DIR}/.."
NNTRAINER_ROOT="${SCRIPT_DIR}/../../.."

echo "========================================"
echo "Building CausalLM for Android (Advanced)"
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

# Build configuration
TARGET_ARCH="aarch64-linux-android"
API_LEVEL="30"
TOOLCHAIN_PREFIX="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64"
CC="${TOOLCHAIN_PREFIX}/bin/${TARGET_ARCH}${API_LEVEL}-clang"
CXX="${TOOLCHAIN_PREFIX}/bin/${TARGET_ARCH}${API_LEVEL}-clang++"
AR="${TOOLCHAIN_PREFIX}/bin/llvm-ar"
STRIP="${TOOLCHAIN_PREFIX}/bin/llvm-strip"

echo "Target: ${TARGET_ARCH} API ${API_LEVEL}"

# Step 1: Build nntrainer for Android (including CausalLM)
echo ""
echo "Step 1: Building nntrainer + CausalLM for Android..."
cd "${NNTRAINER_ROOT}"

# Check if CausalLM is in the build
if [ ! -d "Applications/CausalLM" ]; then
    echo "Error: CausalLM directory not found"
    echo "Please ensure PR #3344 CausalLM implementation is available"
    exit 1
fi

# Modify package_android.sh to include CausalLM
echo "Building nntrainer with CausalLM support..."
bash "${NNTRAINER_ROOT}/tools/package_android.sh" "${NNTRAINER_ROOT}" -Denable-causallm=true

if [ ! -f "${NNTRAINER_ROOT}/nntrainer_for_android.tar.gz" ]; then
    echo "Error: nntrainer_for_android.tar.gz was not created"
    exit 1
fi

echo "nntrainer with CausalLM built successfully"

# Step 2: Extract dependencies
echo ""
echo "Step 2: Extracting dependencies..."
cd "${SCRIPT_DIR}"

if [ -d "android_deps" ]; then
    rm -rf android_deps
fi

mkdir -p android_deps
cd android_deps
tar -xzf "${NNTRAINER_ROOT}/nntrainer_for_android.tar.gz"

ANDROID_DEPS_DIR="${SCRIPT_DIR}/android_deps"
NNTRAINER_INCLUDE="${ANDROID_DEPS_DIR}/include/nntrainer"
NNTRAINER_LIB="${ANDROID_DEPS_DIR}/lib/arm64-v8a"

echo "Dependencies extracted to: ${ANDROID_DEPS_DIR}"

# Step 3: Prepare CausalLM sources
echo ""
echo "Step 3: Preparing CausalLM sources..."

# Create a comprehensive source list
CAUSALLM_SOURCES=""

# Core CausalLM files
CORE_FILES=(
    "main.cpp"
    "causal_lm.cpp"
    "llm_util.cpp"
    "huggingface_tokenizer.cpp"
)

# CausalLM model implementations
MODEL_FILES=(
    "qwen3_causallm.cpp"
    "qwen3_moe_causallm.cpp"
)

# Layer implementations
LAYER_DIR="${CAUSALLM_ROOT}/layers"
LAYER_FILES=(
    "embedding_layer.cpp"
    "rms_norm.cpp"
    "swiglu.cpp"
    "mha_core.cpp"
    "tie_word_embedding.cpp"
    "reshaped_rms_norm.cpp"
    "qwen_moe_layer.cpp"
)

echo "Checking CausalLM source files..."

# Check and add core files
for file in "${CORE_FILES[@]}"; do
    if [ -f "${CAUSALLM_ROOT}/${file}" ]; then
        echo "✅ Found: ${file}"
        CAUSALLM_SOURCES="$CAUSALLM_SOURCES ${CAUSALLM_ROOT}/${file}"
    else
        echo "⚠️  Missing: ${file}"
    fi
done

# Check and add model files
for file in "${MODEL_FILES[@]}"; do
    if [ -f "${CAUSALLM_ROOT}/${file}" ]; then
        echo "✅ Found: ${file}"
        CAUSALLM_SOURCES="$CAUSALLM_SOURCES ${CAUSALLM_ROOT}/${file}"
    else
        echo "⚠️  Missing: ${file}"
    fi
done

# Check and add layer files
for file in "${LAYER_FILES[@]}"; do
    if [ -f "${LAYER_DIR}/${file}" ]; then
        echo "✅ Found: layers/${file}"
        CAUSALLM_SOURCES="$CAUSALLM_SOURCES ${LAYER_DIR}/${file}"
    else
        echo "⚠️  Missing: layers/${file}"
    fi
done

if [ -z "$CAUSALLM_SOURCES" ]; then
    echo "Error: No CausalLM source files found"
    exit 1
fi

echo "Source files to compile: $CAUSALLM_SOURCES"

# Step 4: Build CausalLM executable
echo ""
echo "Step 4: Building CausalLM executable..."

BUILD_DIR="${SCRIPT_DIR}/build"
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Comprehensive compiler flags
CXXFLAGS="-std=c++17 -O3 -DANDROID -fPIC -DPLUGGABLE"
CXXFLAGS="$CXXFLAGS -I${NNTRAINER_INCLUDE}"
CXXFLAGS="$CXXFLAGS -I${CAUSALLM_ROOT}"
CXXFLAGS="$CXXFLAGS -I${CAUSALLM_ROOT}/layers"
CXXFLAGS="$CXXFLAGS -I${NNTRAINER_ROOT}/nntrainer/utils"  # for json.hpp
CXXFLAGS="$CXXFLAGS -fopenmp -static-openmp"
CXXFLAGS="$CXXFLAGS -pthread"

# Linker flags
LDFLAGS="-L${NNTRAINER_LIB}"
LDFLAGS="$LDFLAGS -lnntrainer -lccapi-nntrainer"
LDFLAGS="$LDFLAGS -fopenmp -static-openmp"
LDFLAGS="$LDFLAGS -static-libgcc -static-libstdc++"
LDFLAGS="$LDFLAGS -llog"  # Android logging

# Check if causallm library exists
if [ -f "${NNTRAINER_LIB}/libcausallm.so" ]; then
    echo "Using libcausallm.so"
    LDFLAGS="$LDFLAGS -lcausallm"
    # Only compile main.cpp if library exists
    CAUSALLM_SOURCES="${CAUSALLM_ROOT}/main.cpp"
else
    echo "Building with all source files (no libcausallm.so found)"
fi

echo ""
echo "Build configuration:"
echo "Compiler: $CXX"
echo "CXXFLAGS: $CXXFLAGS"
echo "LDFLAGS: $LDFLAGS"
echo "Sources: $CAUSALLM_SOURCES"

# Compile
echo ""
echo "Compiling..."
$CXX $CXXFLAGS $CAUSALLM_SOURCES $LDFLAGS -o nntr_causallm_android

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
else
    echo "❌ Compilation failed"
    
    # Try alternative compilation without some problematic flags
    echo "Trying alternative compilation..."
    CXXFLAGS_ALT="-std=c++17 -O2 -DANDROID -fPIC"
    CXXFLAGS_ALT="$CXXFLAGS_ALT -I${NNTRAINER_INCLUDE}"
    CXXFLAGS_ALT="$CXXFLAGS_ALT -I${CAUSALLM_ROOT}"
    CXXFLAGS_ALT="$CXXFLAGS_ALT -I${CAUSALLM_ROOT}/layers"
    
    LDFLAGS_ALT="-L${NNTRAINER_LIB}"
    LDFLAGS_ALT="$LDFLAGS_ALT -lnntrainer -lccapi-nntrainer"
    LDFLAGS_ALT="$LDFLAGS_ALT -llog"
    
    $CXX $CXXFLAGS_ALT $CAUSALLM_SOURCES $LDFLAGS_ALT -o nntr_causallm_android
    
    if [ $? -ne 0 ]; then
        echo "❌ Alternative compilation also failed"
        exit 1
    fi
fi

# Strip and check binary
echo "Stripping binary..."
$STRIP nntr_causallm_android

echo ""
echo "Binary information:"
ls -lh nntr_causallm_android
file nntr_causallm_android

# Step 5: Create deployment package
echo ""
echo "Step 5: Creating deployment package..."

DEPLOY_DIR="${SCRIPT_DIR}/deploy"
if [ -d "$DEPLOY_DIR" ]; then
    rm -rf "$DEPLOY_DIR"
fi
mkdir -p "$DEPLOY_DIR"

# Copy binary
cp nntr_causallm_android "$DEPLOY_DIR/"

# Copy all required libraries
echo "Copying required libraries..."
if [ -d "${NNTRAINER_LIB}" ]; then
    for lib in "${NNTRAINER_LIB}"/*.so; do
        if [ -f "$lib" ]; then
            cp "$lib" "$DEPLOY_DIR/"
            echo "  Copied: $(basename $lib)"
        fi
    done
fi

# Create enhanced deployment script
cat > "$DEPLOY_DIR/deploy_to_device.sh" << 'EOF'
#!/bin/bash

# Enhanced CausalLM Android Deployment Script
DEVICE_PATH=${1:-/data/local/tmp/causallm}

echo "========================================"
echo "Deploying CausalLM to Android Device"
echo "========================================"
echo "Target path: $DEVICE_PATH"

# Check prerequisites
if ! command -v adb &> /dev/null; then
    echo "❌ adb not found. Please install Android SDK platform-tools"
    exit 1
fi

if ! adb devices | grep -q "device$"; then
    echo "❌ No Android device connected"
    echo "Please connect device and enable USB debugging"
    exit 1
fi

echo "✅ Device connected"

# Deploy files
echo ""
echo "Deploying files..."
adb shell "mkdir -p $DEVICE_PATH"

echo "  Pushing binary..."
adb push nntr_causallm_android "$DEVICE_PATH/"
adb shell "chmod +x $DEVICE_PATH/nntr_causallm_android"

echo "  Pushing libraries..."
for lib in *.so; do
    if [ -f "$lib" ]; then
        adb push "$lib" "$DEVICE_PATH/"
        echo "    Pushed: $lib"
    fi
done

# Create run script
echo "  Creating run script..."
adb shell "cat > $DEVICE_PATH/run_causallm.sh << 'EOFSCRIPT'
#!/system/bin/sh

# CausalLM Runner Script
CAUSALLM_DIR=\$(dirname \"\$0\")
cd \"\$CAUSALLM_DIR\"

# Set library path
export LD_LIBRARY_PATH=\".:\$LD_LIBRARY_PATH\"

# Check if model path is provided
if [ \$# -eq 0 ]; then
    echo \"Usage: \$0 <model_directory>\"
    echo \"Example: \$0 /sdcard/causallm_models/qwen3-4b\"
    exit 1
fi

MODEL_PATH=\"\$1\"

# Check if model directory exists
if [ ! -d \"\$MODEL_PATH\" ]; then
    echo \"Error: Model directory not found: \$MODEL_PATH\"
    exit 1
fi

echo \"Running CausalLM with model: \$MODEL_PATH\"
echo \"Working directory: \$(pwd)\"
echo \"Library path: \$LD_LIBRARY_PATH\"

# Run CausalLM
./nntr_causallm_android \"\$MODEL_PATH\"
EOFSCRIPT"

adb shell "chmod +x $DEVICE_PATH/run_causallm.sh"

# Create model deployment helper
adb shell "cat > $DEVICE_PATH/deploy_model.sh << 'EOFSCRIPT'
#!/system/bin/sh

# Model Deployment Helper
if [ \$# -ne 2 ]; then
    echo \"Usage: \$0 <source_model_dir> <target_name>\"
    echo \"Example: \$0 /sdcard/Download/qwen3-4b qwen3-4b\"
    exit 1
fi

SOURCE=\"\$1\"
TARGET=\"/sdcard/causallm_models/\$2\"

echo \"Copying model from \$SOURCE to \$TARGET\"
mkdir -p \"/sdcard/causallm_models\"
cp -r \"\$SOURCE\" \"\$TARGET\"
echo \"Model deployed to: \$TARGET\"
EOFSCRIPT"

adb shell "chmod +x $DEVICE_PATH/deploy_model.sh"

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "Usage on device:"
echo "  adb shell"
echo "  cd $DEVICE_PATH"
echo "  ./run_causallm.sh /path/to/model"
echo ""
echo "Quick test:"
echo "  adb shell '$DEVICE_PATH/run_causallm.sh /sdcard/causallm_models/your_model'"
echo ""
echo "To deploy a model:"
echo "  adb push /local/model/path /sdcard/causallm_models/"
echo "  or use: adb shell '$DEVICE_PATH/deploy_model.sh /sdcard/source /target_name'"
EOF

chmod +x "$DEPLOY_DIR/deploy_to_device.sh"

# Create comprehensive README
cat > "$DEPLOY_DIR/README.md" << 'EOF'
# CausalLM Android Native Deployment

## Quick Start

1. **Deploy to device:**
   ```bash
   ./deploy_to_device.sh
   ```

2. **Copy model to device:**
   ```bash
   adb push /path/to/your/model /sdcard/causallm_models/model_name
   ```

3. **Run CausalLM:**
   ```bash
   adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/model_name'
   ```

## Files Included

- `nntr_causallm_android` - Main executable (ARM64)
- `*.so` - Required shared libraries
- `deploy_to_device.sh` - Deployment script
- `README.md` - This documentation

## Model Requirements

Your model directory must contain:
- `config.json` - Model configuration
- `generation_config.json` - Generation parameters
- `nntr_config.json` - NNTrainer configuration
- `tokenizer.json` - Tokenizer data
- `*.bin` - Model weights

## Device Requirements

- Android device with ARM64 architecture
- Android API level 24+ (Android 7.0+)
- Sufficient RAM (depends on model size)
- Storage space for model files
- USB debugging enabled

## Troubleshooting

### Common Issues

1. **"Permission denied"**
   - Ensure USB debugging is enabled
   - Try: `adb shell 'su -c "chmod +x /data/local/tmp/causallm/*"'`

2. **"Library not found"**
   - Check that all .so files are deployed
   - Verify LD_LIBRARY_PATH is set correctly

3. **"Model loading failed"**
   - Verify model files are complete
   - Check available device memory
   - Ensure model path is correct

4. **"Segmentation fault"**
   - Model may be too large for device
   - Try a smaller model
   - Check device logs: `adb logcat`

### Performance Tips

- Use smaller models for better performance
- Ensure device has sufficient free RAM
- Close other applications before running
- Consider using quantized models

## Advanced Usage

### Custom Deployment Path
```bash
./deploy_to_device.sh /data/local/tmp/my_causallm
```

### Running with Different Parameters
Edit the model's `nntr_config.json` to adjust:
- `batch_size` - Reduce for lower memory usage
- `max_seq_len` - Adjust sequence length
- `num_to_generate` - Control output length

### Monitoring Performance
```bash
# Monitor memory usage
adb shell 'top | grep causallm'

# Monitor logs
adb logcat | grep -i causallm
```

## Model Conversion

If you need to convert models to NNTrainer format:
1. Use the weight conversion scripts in the CausalLM/res/ directories
2. Ensure tokenizer files are compatible
3. Test the model on desktop before deploying to Android

## Support

For issues related to:
- **Model conversion**: Check CausalLM documentation
- **Android deployment**: Verify NDK and device compatibility
- **Performance**: Try smaller models or adjust configuration
EOF

echo ""
echo "========================================"
echo "✅ Build and packaging completed!"
echo "========================================"
echo ""
echo "Build artifacts:"
echo "  📱 Binary: ${BUILD_DIR}/nntr_causallm_android"
echo "  📦 Deploy package: ${DEPLOY_DIR}/"
echo "  📚 Documentation: ${DEPLOY_DIR}/README.md"
echo ""
echo "Next steps:"
echo "  1. cd ${DEPLOY_DIR}"
echo "  2. ./deploy_to_device.sh"
echo "  3. adb push /path/to/model /sdcard/causallm_models/"
echo "  4. adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/your_model'"
echo ""
echo "For help: cat ${DEPLOY_DIR}/README.md"