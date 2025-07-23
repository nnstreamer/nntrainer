#!/bin/bash

set -e

# CausalLM Android Build Script using Meson
# This script leverages Meson build system to build CausalLM for Android

SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
CAUSALLM_ROOT="${SCRIPT_DIR}/.."
NNTRAINER_ROOT="${SCRIPT_DIR}/../../.."

echo "========================================"
echo "Building CausalLM for Android (Meson)"
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

# Parse command line arguments
MESON_ARGS=""
CLEAN_BUILD=false
INSTALL_APPS=true
ENABLE_CAUSALLM=true
BUILD_TYPE="release"

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
        -D*)
            MESON_ARGS="$MESON_ARGS $1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--debug] [--no-install] [--disable-causallm] [-Doption=value]"
            exit 1
            ;;
    esac
done

# Go to nntrainer root
cd "$NNTRAINER_ROOT"

# Clean previous build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning previous build..."
    if [ -d "builddir" ]; then
        rm -rf builddir
    fi
fi

# Step 1: Configure Meson for Android with CausalLM
echo ""
echo "Step 1: Configuring Meson build for Android..."

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

# Add user-provided meson arguments
if [ -n "$MESON_ARGS" ]; then
    echo "Additional Meson arguments: $MESON_ARGS"
    MESON_CONFIG+=($MESON_ARGS)
fi

# Configure or reconfigure build
if [ ! -d builddir ]; then
    echo "Creating new build directory..."
    meson setup builddir "${MESON_CONFIG[@]}"
else
    echo "Reconfiguring existing build directory..."
    meson configure builddir "${MESON_CONFIG[@]}"
    # Wipe build if configuration changed significantly
    meson setup --wipe builddir
fi

if [ $? -ne 0 ]; then
    echo "❌ Meson configuration failed"
    exit 1
fi

echo "✅ Meson configuration completed"

# Step 2: Build with Meson
echo ""
echo "Step 2: Building with Meson..."

cd builddir

# Build everything
echo "Running meson compile..."
meson compile

if [ $? -ne 0 ]; then
    echo "❌ Meson build failed"
    exit 1
fi

echo "✅ Meson build completed"

# Step 3: Install to staging area
echo ""
echo "Step 3: Installing to staging area..."

# Install to android_build_result
echo "Running meson install..."
meson install

if [ $? -ne 0 ]; then
    echo "❌ Meson install failed"
    exit 1
fi

echo "✅ Meson install completed"

# Step 4: Create deployment package
echo ""
echo "Step 4: Creating deployment package..."

cd "$NNTRAINER_ROOT"

# Create tarball (similar to package_android.sh)
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

# Step 5: Extract and prepare CausalLM deployment
echo ""
echo "Step 5: Preparing CausalLM deployment..."

DEPLOY_DIR="${SCRIPT_DIR}/deploy_meson"
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
if [ -f "bin/nntr_causallm" ]; then
    CAUSALLM_EXE="bin/nntr_causallm"
elif [ -f "Applications/nntr_causallm" ]; then
    CAUSALLM_EXE="Applications/nntr_causallm"
else
    # Search for the executable
    CAUSALLM_EXE=$(find . -name "nntr_causallm" -type f | head -1)
fi

if [ -z "$CAUSALLM_EXE" ] || [ ! -f "$CAUSALLM_EXE" ]; then
    echo "❌ CausalLM executable not found in build artifacts"
    echo "Available files:"
    find . -name "*causallm*" -type f
    exit 1
fi

echo "✅ Found CausalLM executable: $CAUSALLM_EXE"

# Copy executable to deployment root
cp "$CAUSALLM_EXE" "./nntr_causallm_android"
chmod +x "./nntr_causallm_android"

# Copy all shared libraries
echo "Copying shared libraries..."
if [ -d "lib/arm64-v8a" ]; then
    cp lib/arm64-v8a/*.so . 2>/dev/null || true
elif [ -d "lib" ]; then
    find lib -name "*.so" -exec cp {} . \; 2>/dev/null || true
fi

# List copied libraries
echo "Copied libraries:"
ls -la *.so 2>/dev/null || echo "No .so files found"

# Step 6: Create enhanced deployment scripts
echo ""
echo "Step 6: Creating deployment scripts..."

# Create deployment script
cat > "deploy_to_device.sh" << 'EOF'
#!/bin/bash

# Meson-built CausalLM Android Deployment Script
DEVICE_PATH=${1:-/data/local/tmp/causallm}

echo "========================================"
echo "Deploying Meson-built CausalLM to Android"
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

echo "  Pushing CausalLM executable..."
if [ -f "nntr_causallm_android" ]; then
    adb push nntr_causallm_android "$DEVICE_PATH/"
    adb shell "chmod +x $DEVICE_PATH/nntr_causallm_android"
    echo "    ✅ Executable deployed"
else
    echo "    ❌ Executable not found"
    exit 1
fi

echo "  Pushing shared libraries..."
lib_count=0
for lib in *.so; do
    if [ -f "$lib" ]; then
        adb push "$lib" "$DEVICE_PATH/"
        echo "    Pushed: $lib"
        ((lib_count++))
    fi
done

if [ $lib_count -eq 0 ]; then
    echo "    ⚠️  No shared libraries found"
else
    echo "    ✅ $lib_count libraries deployed"
fi

# Create enhanced run script
echo "  Creating run script..."
adb shell "cat > $DEVICE_PATH/run_causallm.sh << 'EOFSCRIPT'
#!/system/bin/sh

# Meson-built CausalLM Runner Script
CAUSALLM_DIR=\$(dirname \"\$0\")
cd \"\$CAUSALLM_DIR\"

# Set library path
export LD_LIBRARY_PATH=\".:\$LD_LIBRARY_PATH\"

# Check if model path is provided
if [ \$# -eq 0 ]; then
    echo \"Usage: \$0 <model_directory> [options]\"
    echo \"Example: \$0 /sdcard/causallm_models/qwen3-4b\"
    echo \"\"
    echo \"Available options:\"
    echo \"  --help    Show this help\"
    echo \"  --version Show version info\"
    exit 1
fi

MODEL_PATH=\"\$1\"
shift

# Handle special options
if [ \"\$MODEL_PATH\" = \"--help\" ]; then
    echo \"CausalLM Android Runner (Meson-built)\"
    echo \"Usage: \$0 <model_directory> [additional_args]\"
    echo \"\"
    echo \"Model directory should contain:\"
    echo \"  - config.json\"
    echo \"  - generation_config.json\"
    echo \"  - nntr_config.json\"
    echo \"  - tokenizer.json\"
    echo \"  - *.bin (weight file)\"
    exit 0
fi

if [ \"\$MODEL_PATH\" = \"--version\" ]; then
    echo \"CausalLM Android (Meson-built)\"
    ./nntr_causallm_android --version 2>/dev/null || echo \"Version info not available\"
    exit 0
fi

# Check if model directory exists
if [ ! -d \"\$MODEL_PATH\" ]; then
    echo \"❌ Error: Model directory not found: \$MODEL_PATH\"
    echo \"\"
    echo \"Available models:\"
    ls -la /sdcard/causallm_models/ 2>/dev/null || echo \"No models found in /sdcard/causallm_models/\"
    exit 1
fi

echo \"🚀 Running CausalLM with model: \$MODEL_PATH\"
echo \"📁 Working directory: \$(pwd)\"
echo \"📚 Library path: \$LD_LIBRARY_PATH\"
echo \"\"

# Run CausalLM with all remaining arguments
./nntr_causallm_android \"\$MODEL_PATH\" \$@
EOFSCRIPT"

adb shell "chmod +x $DEVICE_PATH/run_causallm.sh"

# Create model management script
adb shell "cat > $DEVICE_PATH/manage_models.sh << 'EOFSCRIPT'
#!/system/bin/sh

# CausalLM Model Management Script

case \"\$1\" in
    list)
        echo \"Available models:\"
        ls -la /sdcard/causallm_models/ 2>/dev/null || echo \"No models found\"
        ;;
    info)
        if [ -z \"\$2\" ]; then
            echo \"Usage: \$0 info <model_name>\"
            exit 1
        fi
        MODEL_DIR=\"/sdcard/causallm_models/\$2\"
        if [ -d \"\$MODEL_DIR\" ]; then
            echo \"Model: \$2\"
            echo \"Location: \$MODEL_DIR\"
            echo \"Files:\"
            ls -la \"\$MODEL_DIR\"
            echo \"\"
            if [ -f \"\$MODEL_DIR/config.json\" ]; then
                echo \"Config preview:\"
                head -20 \"\$MODEL_DIR/config.json\"
            fi
        else
            echo \"Model not found: \$2\"
        fi
        ;;
    test)
        if [ -z \"\$2\" ]; then
            echo \"Usage: \$0 test <model_name>\"
            exit 1
        fi
        MODEL_DIR=\"/sdcard/causallm_models/\$2\"
        if [ -d \"\$MODEL_DIR\" ]; then
            echo \"Testing model: \$2\"
            ./run_causallm.sh \"\$MODEL_DIR\" \"Hello, world!\"
        else
            echo \"Model not found: \$2\"
        fi
        ;;
    *)
        echo \"CausalLM Model Management\"
        echo \"Usage: \$0 {list|info|test} [model_name]\"
        echo \"\"
        echo \"Commands:\"
        echo \"  list           List available models\"
        echo \"  info <model>   Show model information\"
        echo \"  test <model>   Test model with sample input\"
        ;;
esac
EOFSCRIPT"

adb shell "chmod +x $DEVICE_PATH/manage_models.sh"

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📱 Usage on device:"
echo "  adb shell"
echo "  cd $DEVICE_PATH"
echo "  ./run_causallm.sh /path/to/model"
echo ""
echo "🔧 Management commands:"
echo "  ./manage_models.sh list                    # List models"
echo "  ./manage_models.sh info qwen3-4b          # Model info"
echo "  ./manage_models.sh test qwen3-4b          # Test model"
echo ""
echo "🚀 Quick test:"
echo "  adb shell '$DEVICE_PATH/run_causallm.sh /sdcard/causallm_models/your_model'"
EOF

chmod +x "deploy_to_device.sh"

# Create comprehensive README
cat > "README.md" << 'EOF'
# CausalLM Android Deployment (Meson-built)

This package contains CausalLM built using the Meson build system for Android.

## Contents

- `nntr_causallm_android` - Main executable (ARM64)
- `*.so` - Required shared libraries
- `deploy_to_device.sh` - Device deployment script
- `README.md` - This documentation

## Quick Start

1. **Deploy to device:**
   ```bash
   ./deploy_to_device.sh
   ```

2. **Copy model files:**
   ```bash
   adb push /path/to/your/model /sdcard/causallm_models/model_name
   ```

3. **Run CausalLM:**
   ```bash
   adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/model_name'
   ```

## Features

### Enhanced Runner Script
- Automatic library path setup
- Model validation
- Help and version commands
- Detailed error messages

### Model Management
- List available models
- Show model information
- Test models quickly

### Commands

```bash
# Basic usage
./run_causallm.sh /sdcard/causallm_models/qwen3-4b

# Show help
./run_causallm.sh --help

# Show version
./run_causallm.sh --version

# Model management
./manage_models.sh list
./manage_models.sh info qwen3-4b
./manage_models.sh test qwen3-4b
```

## Build Information

This CausalLM was built using:
- **Build System**: Meson
- **Target**: Android ARM64 (aarch64-linux-android)
- **NDK**: Android NDK (version depends on build environment)
- **Optimization**: Release build with optimizations

## Model Requirements

Your model directory should contain:
- `config.json` - Model configuration
- `generation_config.json` - Generation parameters
- `nntr_config.json` - NNTrainer configuration
- `tokenizer.json` - Tokenizer data
- `*.bin` - Model weights

## Troubleshooting

### Common Issues

1. **Executable not found**
   - Ensure deployment completed successfully
   - Check file permissions: `adb shell 'ls -la /data/local/tmp/causallm/'`

2. **Library loading errors**
   - Verify all .so files are present
   - Check LD_LIBRARY_PATH in run script

3. **Model loading failures**
   - Validate model directory structure
   - Check available device memory
   - Verify model file integrity

### Debug Commands

```bash
# Check deployment
adb shell 'ls -la /data/local/tmp/causallm/'

# Test library loading
adb shell 'cd /data/local/tmp/causallm && LD_LIBRARY_PATH=. ./nntr_causallm_android --help'

# Monitor logs
adb logcat | grep -i causallm
```

## Performance Notes

- Built with Meson optimizations enabled
- OpenMP support for multi-threading
- Optimized for ARM64 architecture
- Consider model size vs. device memory capacity

For more information, see the main CausalLM documentation.
EOF

echo ""
echo "========================================"
echo "✅ Meson-powered build completed!"
echo "========================================"
echo ""
echo "📊 Build Summary:"
echo "  🔧 Build System: Meson"
echo "  📱 Target: Android ARM64"
echo "  📦 Executable: $DEPLOY_DIR/nntr_causallm_android"
echo "  📚 Libraries: $(ls -1 $DEPLOY_DIR/*.so 2>/dev/null | wc -l) shared libraries"
echo "  📁 Deploy Package: $DEPLOY_DIR/"
echo ""
echo "🚀 Next Steps:"
echo "  1. cd $DEPLOY_DIR"
echo "  2. ./deploy_to_device.sh"
echo "  3. adb push /path/to/model /sdcard/causallm_models/"
echo "  4. adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/your_model'"
echo ""
echo "📖 For detailed usage: cat $DEPLOY_DIR/README.md"