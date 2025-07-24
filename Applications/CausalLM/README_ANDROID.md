# Building CausalLM for Android

This guide explains how to build and deploy the CausalLM application on Android devices following the standard nntrainer Android build process.

## Prerequisites

1. **Android NDK**: Download and install Android NDK (r21d recommended)
   ```bash
   # Download NDK
   wget https://dl.google.com/android/repository/android-ndk-r21d-linux-x86_64.zip
   unzip android-ndk-r21d-linux-x86_64.zip
   
   # Set environment variables
   export ANDROID_NDK=$(pwd)/android-ndk-r21d
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANDROID_NDK
   export PATH=$PATH:$ANDROID_NDK
   ```

2. **Android SDK Tools**: Install ADB (Android Debug Bridge)
   ```bash
   sudo apt-get install android-tools-adb  # On Ubuntu/Debian
   ```

3. **Build Tools**: 
   - Meson build system
   - Ninja build system
   - Python 3.6+

4. **Tokenizer Library**: The pre-built `libtokenizers_c.a` for Android arm64-v8a
   - Place it in `Applications/CausalLM/lib/libtokenizers_c.a`
   - You can build it using the provided script (requires Rust)

## Building

### Step 1: Build the tokenizer library (optional)

If you don't have the pre-built tokenizer library:

```bash
cd Applications/CausalLM
chmod +x build_tokenizer_android.sh
./build_tokenizer_android.sh
```

### Step 2: Build the application

```bash
cd Applications/CausalLM
chmod +x build_android.sh
./build_android.sh
```

This script will:
1. Build nntrainer core library for Android using `tools/package_android.sh` (if not already built)
2. Copy the built libraries to the expected location
3. Build CausalLM application using ndk-build

### Step 3: Install on Android device

```bash
chmod +x install_android.sh
./install_android.sh
```

## Running on Android

### Prepare model files

First, push your model files to the device:

```bash
# Example for Qwen3-4B model
adb push res/qwen3-4b /data/local/tmp/nntrainer/causallm/models/qwen3-4b/
```

The model directory should contain:
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `nntr_config.json`
- Model weight file (e.g., `nntr_qwen3_4b_fp32.bin`)

### Run the application

```bash
# Run with model path
adb shell /data/local/tmp/nntrainer/causallm/run_causallm.sh /data/local/tmp/nntrainer/causallm/models/qwen3-4b
```

Or connect to device shell:

```bash
adb shell
cd /data/local/tmp/nntrainer/causallm
./run_causallm.sh models/qwen3-4b
```

## Troubleshooting

### Missing tokenizer library
If you get an error about missing `libtokenizers_c.a`, you need to build the tokenizer library for Android:

1. Clone tokenizers repository
2. Install Rust and Android targets
3. Build using cargo with Android target

### Permission denied
If you get permission errors, make sure the files have execute permissions:

```bash
adb shell chmod 755 /data/local/tmp/nntrainer/causallm/nntr_causallm
adb shell chmod 755 /data/local/tmp/nntrainer/causallm/run_causallm.sh
```

### Library loading errors
If you get library loading errors, check that all required libraries are present:

```bash
adb shell ls -la /data/local/tmp/nntrainer/causallm/
```

Required libraries:
- `libnntrainer.so`
- `libccapi-nntrainer.so`
- `libcausallm.so`
- `libc++_shared.so`

## Performance Notes

- The application is optimized for ARM64 (aarch64) architecture
- OpenMP is enabled for multi-threading support
- NEON optimizations are enabled
- For best performance, use devices with sufficient RAM (8GB+ recommended for larger models)

## Limitations

- Currently only supports arm64-v8a ABI
- Requires Android API level 29 (Android 10) or higher
- Model size is limited by device memory