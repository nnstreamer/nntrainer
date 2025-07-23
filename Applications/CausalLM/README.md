# ☄️ CausalLM Inference with NNTrainer

This application provides examples to run causal language model (LLM) inference using nntrainer with **Android build support**.

## Features

- **Cross-platform support**: Works on Linux, Android, and other platforms
- **Multiple model architectures**: Supports Llama, Qwen3, Qwen3MoE models
- **Android JNI support**: Native Android integration through JNI
- **Efficient inference**: Optimized for mobile and embedded devices

## Supported Models

- Llama
- Qwen3 (1.7b/4b/7b/14b)
- Qwen3MoE (30b-A3b)
- Custom models with custom layers

## Android Build Support

### Prerequisites

1. **Android NDK**: Version 21 or higher
2. **Meson build system**: Version 0.55 or higher
3. **nntrainer**: Built and configured for Android

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
   # The script will provide deployment instructions
   adb push build_android_causallm/package/bin/nntrainer_causallm /data/local/tmp/
   adb push build_android_causallm/package/lib/*.so /data/local/tmp/
   adb shell chmod +x /data/local/tmp/nntrainer_causallm
   adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=. ./nntrainer_causallm'
   ```

### Android Build Options

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

## Directory Structure

```
Applications/CausalLM/
├── README.md                 # This file
├── meson.build              # Main build configuration
├── build_android.sh         # Android build script
├── jni/                     # Android JNI wrapper
│   ├── meson.build         # JNI build configuration
│   └── main.cpp            # JNI main entry point
├── layers/                  # Custom layer implementations
├── lib/                    # External libraries
│   └── libtokenizers_c.a   # Tokenizer library
└── res/                    # Model resources
```

## Build Configuration

The build system automatically detects the target platform:

- **Android builds**: Uses JNI wrapper in `jni/` directory
- **Native builds**: Direct executable build with full feature set
- **Platform detection**: Automatic based on `-Dplatform=android` option

## Dependencies

### Core Dependencies
- nntrainer library
- nntrainer C++ API
- OpenMP (for parallel processing)

### Android-Specific Dependencies
- Android NDK
- Android system libraries

### Optional Dependencies
- Tokenizer library (libtokenizers_c.a)
- Custom layer implementations

## Troubleshooting

### Android Build Issues

1. **NDK not found**: Set `ANDROID_NDK_ROOT` environment variable
2. **Build script permission**: Run `chmod +x build_android.sh`
3. **Cross-compilation errors**: Ensure NDK version is 21 or higher
4. **Missing dependencies**: Make sure nntrainer is built for Android first

### Runtime Issues

1. **Permission denied**: Run `adb shell chmod +x /data/local/tmp/nntrainer_causallm`
2. **Library not found**: Ensure all .so files are in the same directory and LD_LIBRARY_PATH is set
3. **Model loading**: Ensure model files are accessible on device with proper permissions

## Architecture

### Android Build Flow

1. **Main Build**: Uses nntrainer's main build system with `-Dplatform=android`
2. **JNI Integration**: CausalLM includes JNI subdirectory for Android-specific code
3. **Cross-compilation**: Automatic NDK toolchain setup and cross-compilation
4. **Packaging**: Automatic executable and library packaging for deployment

### Reference Implementation

This implementation follows the reference patch style from commit `ae24db6e9c018a819841f5884defb2c9c1fc3a14`, providing:

- Clean separation between native and Android builds
- Minimal JNI wrapper for Android compatibility  
- Integration with existing nntrainer build system
- Automated build and deployment process

## Contributing

Feel free to contribute improvements, especially for:
- Additional model architectures
- Performance optimizations
- Enhanced Android integration
- iOS support

## License

This project follows the same license as the parent nntrainer project.