# Android Build for arm64-v8a Only

This document explains the changes made to build NNTrainer for Android targeting only the arm64-v8a architecture.

## Changes Made

### 1. Application.mk
- Updated `APP_ABI` to only include `arm64-v8a`

### 2. prepare_protobuf.sh
- Modified to only build protobuf for arm64-v8a architecture
- Removed build steps for armeabi-v7a

### 3. Android.mk.in
- Updated library paths to only use arm64-v8a libraries:
  - ml-api-inference
  - OpenCL
  - protobuf
  - protobuf-lite
- Removed conditional logic for armeabi-v7a in tensorflow-lite library path

### 4. package_android.sh
- Updated to only copy arm64-v8a protobuf libraries

## Rationale

Building for only arm64-v8a simplifies the build process and reduces the size of the resulting APK. Most modern Android devices support arm64-v8a, making this a reasonable optimization.

## Building

To build NNTrainer for Android with arm64-v8a only:

```bash
./tools/package_android.sh
```

This will create an `nntrainer_for_android.tar.gz` file in the project root directory containing the built libraries and headers for arm64-v8a.
