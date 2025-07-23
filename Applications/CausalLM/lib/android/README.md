# Android Tokenizer Library

This directory should contain the Android-specific tokenizer library for CausalLM builds.

## Required Files

For Android builds, place the following files in this directory:

- `libtokenizers_c.a` - Static tokenizer library compiled for Android target architecture
- `libtokenizers_c.so` - Shared tokenizer library (if needed)

## Building the Tokenizer Library

The tokenizer library should be built using the Android NDK with the appropriate target architecture (arm64-v8a, armeabi-v7a, x86, x86_64).

## Usage

The meson build system will automatically detect and link against the tokenizer library when building CausalLM for Android platform.

If the library is not found, a warning will be displayed during the build process, but the build will continue without tokenizer support.