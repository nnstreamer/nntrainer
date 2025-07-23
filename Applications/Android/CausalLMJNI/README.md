# CausalLM Android Application

This directory contains the Android application for running CausalLM models (Llama, Qwen3, Qwen3-MoE) using NNTrainer on Android devices.

## Overview

This Android application provides a JNI interface to run CausalLM models from PR #3344 on Android devices. It supports various transformer-based causal language models including:

- Llama
- Qwen3 (1.7b/4b/7b/14b)
- Qwen3-MoE (30b-A3b)

## Prerequisites

1. **Android NDK**: Make sure you have Android NDK installed and `ANDROID_NDK` environment variable set
2. **Android SDK**: Android SDK with API level 30 or higher
3. **NNTrainer**: The main NNTrainer library must be built for Android first
4. **Model Files**: CausalLM model configuration and weight files

## Directory Structure

```
CausalLMJNI/
├── app/
│   ├── build.gradle                 # App-level build configuration
│   └── src/main/
│       ├── AndroidManifest.xml      # Android manifest
│       ├── java/com/applications/causallmjni/
│       │   └── MainActivity.java    # Main activity with JNI calls
│       ├── jni/                     # JNI native code
│       │   ├── Android.mk           # NDK build configuration
│       │   ├── Application.mk       # NDK application configuration
│       │   ├── causallm_jni.h       # JNI header file
│       │   ├── causallm_jni.cpp     # JNI implementation
│       │   └── prepare_android_deps.sh # Dependency preparation script
│       └── res/                     # Android resources
├── build.gradle                     # Project-level build configuration
├── settings.gradle                  # Gradle settings
├── gradle.properties               # Gradle properties
├── build_causallm_android.sh       # Main build script
└── README.md                       # This file
```

## Build Instructions

### Step 1: Set Environment Variables

```bash
export ANDROID_NDK=/path/to/your/android-ndk
export ANDROID_SDK_ROOT=/path/to/your/android-sdk
```

### Step 2: Build the Application

Run the main build script:

```bash
cd Applications/Android/CausalLMJNI
./build_causallm_android.sh
```

This script will:
1. Run `package_android.sh` to build NNTrainer for Android
2. Prepare Android dependencies
3. Build the JNI library
4. Build the Android APK (optional)

### Step 3: Manual Build (Alternative)

If you prefer to build manually:

```bash
# 1. Build NNTrainer for Android
cd /path/to/nntrainer/root
./tools/package_android.sh

# 2. Prepare dependencies
cd Applications/Android/CausalLMJNI/app/src/main/jni
./prepare_android_deps.sh

# 3. Build JNI
$ANDROID_NDK/ndk-build

# 4. Build APK
cd ../../../..
./gradlew assembleDebug
```

## Model Setup

### Preparing Model Files

1. Download or convert your CausalLM model to NNTrainer format
2. Ensure you have the following files:
   - `config.json` - Model configuration
   - `generation_config.json` - Generation parameters
   - `nntr_config.json` - NNTrainer specific configuration
   - `tokenizer.json` - Tokenizer configuration
   - `*.bin` - Model weight file

### Model Directory Structure

```
/sdcard/causallm_models/qwen3-4b/
├── config.json
├── generation_config.json
├── nntr_config.json
├── tokenizer.json
└── nntr_qwen3_4b_fp32.bin
```

### Updating Model Path

Edit the model path in `MainActivity.java`:

```java
String configPath = "/sdcard/causallm_models/qwen3-4b";
String weightPath = configPath + "/nntr_qwen3_4b_fp32.bin";
```

## Usage

1. Install the APK on your Android device
2. Copy model files to the device storage
3. Launch the application
4. Tap "Initialize Model" to load the model
5. Enter your prompt in the text field
6. Tap "Run Inference" to generate text

## JNI Interface

The JNI interface provides the following methods:

- `createCausalLMModel(String configPath)` - Create model instance
- `initializeModel(long modelPointer)` - Initialize the model
- `loadWeights(long modelPointer, String weightPath)` - Load model weights
- `runInference(long modelPointer, String inputText, boolean doSample)` - Run inference
- `destroyModel(long modelPointer)` - Clean up model resources

## Troubleshooting

### Common Issues

1. **NDK Build Fails**
   - Ensure `ANDROID_NDK` is correctly set
   - Check that all dependencies are properly extracted

2. **Model Loading Fails**
   - Verify model files are in the correct location
   - Check file permissions on the device
   - Ensure sufficient storage space

3. **JNI Library Not Found**
   - Make sure the JNI build completed successfully
   - Check that the library is in the correct architecture folder

### Debug Logs

Use `adb logcat` to view debug logs:

```bash
adb logcat -s CausalLMJNI
```

## Performance Considerations

- **Memory**: CausalLM models require significant memory. Ensure your device has sufficient RAM
- **Storage**: Model files can be large (several GB). Ensure adequate storage space
- **CPU**: Inference may be slow on mobile devices. Consider using smaller models for better performance

## Customization

### Adding New Models

To add support for new CausalLM models:

1. Update the factory registration in `causallm_jni.cpp`
2. Ensure the model follows the CausalLM interface
3. Update model configuration files accordingly

### Modifying UI

The UI is defined in `activity_main.xml` and can be customized as needed. The main activity handles the model lifecycle and user interactions.

## License

This project follows the same license as NNTrainer (Apache 2.0).

## Contributing

When contributing to this Android application:

1. Follow the existing code style
2. Test on multiple Android devices if possible
3. Update documentation as needed
4. Ensure compatibility with the main CausalLM implementation