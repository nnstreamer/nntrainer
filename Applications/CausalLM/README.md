# ☄️ CausalLM Inference with NNTrainer

This application provides examples to run causal language model (LLM) inference using nntrainer with **Android build support**.

## Features

- **Cross-platform support**: Works on Linux, Android, and other platforms
- **Multiple model architectures**: Supports Llama, Qwen3, Qwen3MoE models
- **Simple executable**: Single executable for both native and Android platforms
- **Efficient inference**: Optimized for mobile and embedded devices

## Supported Models

- Llama
- Qwen3 (1.7b/4b/7b/14b)
- Qwen3MoE (30b-A3b)
- Custom models with custom layers

## Android Build Support

### Prerequisites for Android Builds

1. **Android NDK**: Version 21 or higher
2. **Meson build system**: Version 0.55 or higher
3. **Android tokenizer library**: Place in `lib/android/` directory (optional)

### Building for Android

1. **Set environment variables**:
   ```bash
   export ANDROID_NDK_ROOT=/path/to/android-ndk
   export ANDROID_ABI=arm64-v8a  # or armeabi-v7a, x86, x86_64
   export ANDROID_API_LEVEL=21
   ```

2. **Run the build script**:
   ```bash
   cd Applications/CausalLM
   ./build_android.sh
   ```

3. **Deploy to Android device**:
   ```bash
   adb push build_android/package/bin/nntr_causallm_android /data/local/tmp/
   adb shell chmod +x /data/local/tmp/nntr_causallm_android
   adb shell /data/local/tmp/nntr_causallm_android
   ```

### Android Build Script Options

```bash
# Show help
./build_android.sh --help

# Clean build directory
./build_android.sh clean

# Build with specific ABI
ANDROID_ABI=armeabi-v7a ./build_android.sh
```

## Native Build (Linux/Desktop)

### How to Run (Native)

1. Download and copy model files from HuggingFace to `res/{model}` directory
2. The folder should contain:
   - `config.json`
   - `generation_config.json`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `vocab.json`
   - `nntr_config.json`
   - nntrainer weight binfile

3. Compile the application:
   ```bash
   # From the main nntrainer directory
   meson setup builddir -Dplatform=none
   meson compile -C builddir
   ```

4. Run the model:
   ```bash
   cd builddir/Applications/CausalLM
   ./nntr_causallm /path/to/model/config/folder/
   ```

## Usage Examples

### Native Linux
```bash
./nntr_causallm /home/user/models/qwen3-4b/
```

### Android
```bash
# Copy model to device first
adb push /home/user/models/qwen3-4b/ /data/local/tmp/qwen3-4b/

# Run on device
adb shell /data/local/tmp/nntr_causallm_android /data/local/tmp/qwen3-4b/
```

## Directory Structure

```
Applications/CausalLM/
├── README.md                 # This file
├── main.cpp                  # Main executable source
├── meson.build              # Native build configuration
├── meson_android.build      # Android build configuration
├── build_android.sh         # Android build script
├── layers/                  # Custom layer implementations
│   └── README.md           # Layer documentation
├── lib/                    # External libraries
│   └── android/            # Android-specific libraries
│       └── README.md       # Android library documentation
└── res/                    # Model resources (when available)
```

## Build Configuration

The build system automatically detects the target platform:

- **Android builds**: Uses `meson_android.build` with Android-specific settings
- **Native builds**: Uses main `meson.build` with full feature set
- **Cross-compilation**: Supports various Android architectures (arm64-v8a, armeabi-v7a, x86, x86_64)

## Dependencies

### Core Dependencies
- nntrainer library (optional for Android builds)
- Standard C++ library

### Android-Specific Dependencies
- Android NDK
- Android logging library

### Optional Dependencies
- Tokenizer library (libtokenizers_c.a)
- Custom layer implementations

## Troubleshooting

### Android Build Issues

1. **NDK not found**: Set `ANDROID_NDK_ROOT` environment variable
2. **Build script permission**: Run `chmod +x build_android.sh`
3. **Cross-compilation errors**: Ensure NDK version is 21 or higher

### Runtime Issues

1. **Permission denied**: Run `adb shell chmod +x /data/local/tmp/nntr_causallm_android`
2. **Model loading**: Ensure model files are accessible on device
3. **Library not found**: Check if required .so files are in the same directory

## Contributing

Feel free to contribute improvements, especially for:
- Additional model architectures
- Performance optimizations
- iOS support
- Enhanced Android integration

## License

This project follows the same license as the parent nntrainer project.