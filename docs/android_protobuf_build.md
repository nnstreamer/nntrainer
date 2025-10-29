# Android Protobuf Build Fix

## Problem

When building nntrainer for Android with ONNX interpreter enabled, the build process was failing with errors like:

```
ld.lld: error: /home/code/niket/sumon_nn/nntrainer/builddir/jni/protobuf-25.2/lib/libprotobuf.a(absl_base_internal_cycleclock.cc.o) is incompatible with aarch64linux
```

This happened because the Android build was using host-built protobuf libraries (built for x86_64) instead of Android-compatible libraries (built for aarch64).

## Solution

We've implemented a solution that builds protobuf specifically for Android architectures using the Android NDK. The changes include:

### 1. Updated `prepare_protobuf.sh` script

The script now accepts an optional third parameter for the Android NDK path and builds protobuf for both arm64-v8a and armeabi-v7a architectures when the NDK path is provided.

### 2. Modified `jni/meson.build`

Updated to pass the Android NDK path to the prepare_protobuf.sh script. The NDK path is determined by:
1. The `android-ndk-path` meson option
2. Common locations like `/opt/android-ndk` or `/usr/local/android-ndk`

### 3. Updated `meson_options.txt`

Added the `android-ndk-path` option to allow specifying the Android NDK path directly.

### 4. Modified `jni/Android.mk.in`

Updated to use architecture-specific protobuf libraries:
- For arm64-v8a: `lib/arm64-v8a/libprotobuf.a`
- For armeabi-v7a: `lib/armeabi-v7a/libprotobuf.a`

### 5. Created `jni/Application.mk`

Added proper configurations for Android builds:
```
APP_ABI := arm64-v8a armeabi-v7a
APP_PLATFORM := android-21
APP_STL := c++_static
APP_CPPFLAGS := -fexceptions -frtti
```

### 6. Updated `tools/package_android.sh`

Modified to use our new prepare_protobuf.sh script when the Android NDK is available, falling back to the old method when it's not.

## Usage

### Prerequisites

1. Install Android NDK (recommended version r26d)
2. Set the `ANDROID_NDK` or `ANDROID_NDK_HOME` environment variable

### Building

You can build using either method:

#### Using shell script (recommended):
```bash
./tools/package_android.sh
```

#### Using meson:
```bash
meson setup build -Dplatform=android -Denable-onnx-interpreter=true -Dandroid-ndk-path=/path/to/your/ndk
meson compile -C build
```

## How it works

1. When building for Android with ONNX interpreter enabled, the build system detects if an Android NDK is available
2. If available, it uses the NDK to build protobuf specifically for the target Android architectures
3. The Android NDK build process creates architecture-specific libraries in the correct directory structure
4. The Android.mk file is configured to use the correct architecture-specific libraries
5. This ensures that all libraries are compatible with the target Android architecture

## Fallback

If the Android NDK is not found, the build process falls back to the previous method of using host-built libraries, but with a warning that this may cause compatibility issues.
