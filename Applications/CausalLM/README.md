# ☄️ CausalLM Inference with NNTrainer

This application provides examples to run causal language model (LLM) inference using nntrainer with **Android build support**.

## Features

- **Cross-platform support**: Works on Linux, Android, and other platforms
- **Multiple model architectures**: Supports Llama, Qwen3, Qwen3MoE models
- **Android JNI integration**: Full Android NDK support with JNI wrapper
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
3. **Android tokenizer library**: Place in `lib/android/` directory

### Building for Android

1. **Configure for Android platform**:
   ```bash
   meson setup builddir --cross-file android-cross-file.txt -Dplatform=android
   ```

2. **Build the application**:
   ```bash
   meson compile -C builddir
   ```

3. **Install to device**:
   ```bash
   meson install -C builddir
   ```

### Android-Specific Features

- **JNI Interface**: Native Java interface for Android apps
- **Android logging**: Uses Android log system for debugging
- **Resource management**: Automatic resource copying for Android
- **Memory optimization**: Optimized for mobile memory constraints

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
   meson setup builddir -Dplatform=none
   meson compile -C builddir
   ```

4. Run the model:
   ```bash
   cd builddir/Applications/CausalLM
   ./nntr_causallm /path/to/model/config/folder/
   ```

## JNI API (Android)

The Android build provides the following JNI methods:

```java
public class CausalLM {
    // Initialize the CausalLM model
    public native boolean initialize();
    
    // Load model weights from file
    public native boolean loadModel(String modelPath);
    
    // Run inference with input text
    public native String runInference(String input);
    
    // Cleanup resources
    public native void cleanup();
}
```

## Directory Structure

```
Applications/CausalLM/
├── README.md                 # This file
├── meson.build              # Main build configuration
├── jni/                     # Android JNI wrapper
│   ├── meson.build         # JNI build configuration
│   ├── main.cpp            # JNI main entry point
│   └── causallm_jni.cpp    # JNI interface implementation
├── layers/                  # Custom layer implementations
│   └── README.md           # Layer documentation
├── lib/                    # External libraries
│   └── android/            # Android-specific libraries
│       └── README.md       # Android library documentation
└── res/                    # Model resources (when available)
```

## Build Configuration

The build system automatically detects the target platform and configures accordingly:

- **Android builds**: Uses JNI wrapper, Android-specific libraries, and optimized settings
- **Native builds**: Direct executable build with full feature set
- **Cross-compilation**: Supports various target architectures

## Dependencies

### Core Dependencies
- nntrainer library
- nntrainer C++ API
- OpenMP (for parallel processing)

### Android-Specific Dependencies
- Android NDK
- Android logging library
- JNI headers

### Optional Dependencies
- Tokenizer library (libtokenizers_c.a)
- Custom layer implementations

## Troubleshooting

### Android Build Issues

1. **Missing tokenizer library**: Place `libtokenizers_c.a` in `lib/android/`
2. **NDK version**: Ensure Android NDK 21+ is installed
3. **Architecture mismatch**: Build tokenizer for target Android architecture

### Runtime Issues

1. **Model loading**: Ensure model files are accessible on device
2. **Memory constraints**: Use appropriate model size for device capabilities
3. **Permissions**: Ensure app has necessary file access permissions

## Contributing

Feel free to contribute improvements, especially for:
- Additional model architectures
- Performance optimizations
- Android UI integration
- iOS support

## License

This project follows the same license as the parent nntrainer project.