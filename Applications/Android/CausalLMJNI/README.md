# CausalLM Android Application

This is an Android application for running CausalLM (Causal Language Model) inference using NNTrainer framework.

## Features

- Support for Qwen3 and Qwen3-MoE models
- HuggingFace tokenizer integration
- Real-time inference on Android devices
- User-friendly interface with sampling options

## Prerequisites

### Development Environment
- Android Studio Arctic Fox or later
- Android NDK 25.2.9519653 or later
- Android SDK API level 24 or higher
- Linux development environment (for cross-compilation)

### Build Tools
- Meson build system
- Ninja build system
- CMake (optional)

### Device Requirements
- Android 7.0 (API level 24) or higher
- ARM64 (aarch64) processor
- Minimum 4GB RAM (8GB+ recommended for larger models)
- Sufficient storage space for model files

## Building the Application

### 1. Setup Environment Variables

```bash
export ANDROID_NDK=$HOME/Android/Sdk/ndk/25.2.9519653
export ANDROID_SDK=$HOME/Android/Sdk
```

### 2. Build NNTrainer and CausalLM

From the project root directory:

```bash
# Clean previous builds (optional)
./build_causallm_android.sh clean

# Build the application
./build_causallm_android.sh build
```

### 3. Install on Device

Connect your Android device and enable USB debugging, then:

```bash
./build_causallm_android.sh install
```

## Manual Build Process

If you prefer to build manually:

### 1. Build NNTrainer for Android

```bash
mkdir -p build/android
cd build/android

meson setup \
    --cross-file=../../cross_file_android_arm64.txt \
    --buildtype=release \
    --prefix=$(pwd)/install \
    -Dplatform=android \
    -Denable-test=false \
    -Denable-app=false \
    -Denable-capi=true \
    -Denable-ccapi=true \
    ../..

ninja
ninja install
```

### 2. Build Android App

```bash
cd Applications/Android/CausalLMJNI
./gradlew assembleDebug
```

## Model Setup

### 1. Prepare Model Files

Create a model directory on your device (e.g., `/sdcard/nntrainer/causallm/qwen3-4b/`) with the following files:

- `config.json` - HuggingFace model configuration
- `generation_config.json` - Generation parameters
- `nntr_config.json` - NNTrainer specific configuration
- `tokenizer.json` - HuggingFace tokenizer
- `*.bin` - NNTrainer weight file

### 2. Example nntr_config.json

```json
{
    "model_tensor_type": "FP32-FP32",
    "model_file_name": "nntr_qwen3_4b_fp32.bin",
    "fc_layer_dtype": "FP32",
    "embedding_dtype": "FP32",
    "bad_word_ids": [],
    "fsu": false,
    "num_to_generate": 512,
    "init_seq_len": 1024,
    "max_seq_len": 2048,
    "batch_size": 1,
    "tokenizer_file": "/sdcard/nntrainer/causallm/qwen3-4b/tokenizer.json",
    "sample_input": "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"
}
```

### 3. Convert HuggingFace Model

Use the provided weight converter scripts in `Applications/CausalLM/res/` to convert HuggingFace models to NNTrainer format.

## Usage

1. Launch the CausalLM app on your Android device
2. Tap "Initialize Model" to load the model (this may take several minutes)
3. Enter your prompt in the input text field
4. Optionally enable sampling for more creative outputs
5. Tap "Run Inference" to generate text

## Supported Models

- Qwen3 (1.7B, 4B, 7B, 14B)
- Qwen3-MoE (30B-A3B)
- LLaMA (with minor modifications)
- Custom models following the same architecture

## Performance Tips

1. **Model Size**: Smaller models (1.7B-4B) work better on mobile devices
2. **Memory**: Close other apps to free up memory before running inference
3. **Storage**: Use fast internal storage for model files
4. **Cooling**: Allow device to cool between long inference sessions

## Troubleshooting

### Build Issues

1. **NDK not found**: Ensure ANDROID_NDK environment variable is set correctly
2. **Meson errors**: Install latest meson and ninja build tools
3. **Cross-compilation errors**: Check that the cross-file paths are correct

### Runtime Issues

1. **Model initialization failed**: Check model file paths and permissions
2. **Out of memory**: Try a smaller model or close other apps
3. **Slow inference**: This is normal for large models on mobile devices

### Common Error Messages

- "Model not initialized": Tap "Initialize Model" first
- "Failed to load model weights": Check file paths in nntr_config.json
- "Tokenizer error": Ensure tokenizer.json is in the correct location

## File Structure

```
Applications/Android/CausalLMJNI/
├── app/
│   ├── build.gradle
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/ai/nnstreamer/nntrainer/causallm/
│       │   └── CausalLMActivity.java
│       ├── jni/
│       │   ├── Android.mk
│       │   ├── causallm_jni.cpp
│       │   └── prepare_tokenizer.sh
│       └── res/
│           ├── layout/activity_causallm.xml
│           └── values/strings.xml
├── build.gradle
├── gradle.properties
└── settings.gradle
```

## Contributing

When contributing to this project:

1. Follow the existing code style
2. Test on multiple Android devices if possible
3. Update documentation for any new features
4. Consider performance impact on mobile devices

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.