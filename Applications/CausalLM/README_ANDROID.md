# Building CausalLM for Android

This guide explains how to build and deploy the CausalLM application on Android devices.

## Prerequisites

1. **Android NDK**: Download and install Android NDK (r21 or later recommended)
   ```bash
   export ANDROID_NDK_ROOT=/path/to/your/android-ndk
   ```

2. **Android SDK Tools**: Install ADB (Android Debug Bridge)
   ```bash
   sudo apt-get install android-tools-adb  # On Ubuntu/Debian
   ```

3. **Build Tools**: 
   - CMake 3.16 or later
   - Ninja build system
   - Python 3.6+
   - Meson build system

4. **Tokenizer Library**: The pre-built `libtokenizers_c.a` for Android arm64-v8a
   - Place it in `Applications/CausalLM/lib/libtokenizers_c.a`
   - You can build it from [huggingface/tokenizers](https://github.com/huggingface/tokenizers) with Rust cross-compilation

## Building

### Step 1: Apply the Android build patch

```bash
cd Applications/CausalLM
patch -p2 < android_build_patch.diff
```

### Step 2: Build the application

```bash
chmod +x build_android.sh
./build_android.sh
```

This script will:
1. Build nntrainer core library for Android using Meson cross-compilation
2. Build CausalLM application using Android NDK

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