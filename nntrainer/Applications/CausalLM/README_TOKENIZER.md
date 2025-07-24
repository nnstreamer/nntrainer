# Tokenizer Build for Android

This document describes how to build the tokenizer library for Android using mlc-ai/tokenizers-cpp.

## Prerequisites

1. **Android NDK**: Make sure `ANDROID_NDK` environment variable is set
   ```bash
   export ANDROID_NDK=/path/to/android-ndk
   ```

2. **Rust**: Install Rust from https://rustup.rs/
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **CMake**: Install CMake (version 3.10 or higher)
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install cmake
   
   # On macOS
   brew install cmake
   ```

## Building the Tokenizer Library

The `build_tokenizer_android.sh` script automates the process of building the tokenizer library for Android.

### Basic Usage

Build for the default target (arm64-v8a):
```bash
./build_tokenizer_android.sh
```

### Building for Different Architectures

Build for a specific Android ABI:
```bash
# For ARM 64-bit (default)
./build_tokenizer_android.sh arm64-v8a

# For ARM 32-bit
./build_tokenizer_android.sh armeabi-v7a

# For x86 (emulator)
./build_tokenizer_android.sh x86

# For x86_64 (emulator)
./build_tokenizer_android.sh x86_64
```

### Output

The script will:
1. Clone mlc-ai/tokenizers-cpp repository
2. Install the required Rust target for Android
3. Build the tokenizer libraries using CMake and Cargo
4. Combine all static libraries into a single `libtokenizers_android_c.a`
5. Place the output in:
   - `lib/<ABI>/libtokenizers_android_c.a` for the specific ABI
   - `lib/libtokenizers_android_c.a` for arm64-v8a (backward compatibility)

## Integration with build_android.sh

The main Android build script (`build_android.sh`) will automatically:
1. Check if `lib/libtokenizers_android_c.a` exists
2. If not found, run `build_tokenizer_android.sh` to build it
3. Link the tokenizer library when building the CausalLM application

## Troubleshooting

### Rust Target Installation Issues
If you encounter issues with Rust target installation, you can manually install them:
```bash
rustup target add aarch64-linux-android    # for arm64-v8a
rustup target add armv7-linux-androideabi  # for armeabi-v7a
rustup target add i686-linux-android       # for x86
rustup target add x86_64-linux-android     # for x86_64
```

### Build Failures
If the build fails:
1. Check that all prerequisites are installed correctly
2. Ensure `ANDROID_NDK` points to a valid NDK installation
3. Try cleaning the build directory:
   ```bash
   rm -rf tokenizers-cpp-build/
   ```
4. Check the build logs for specific error messages

### Library Not Found
If the combined library is not created:
1. Check that the individual libraries are built successfully
2. Verify that `llvm-ar` from the NDK is accessible
3. Look for `.a` files in the build directory:
   ```bash
   find tokenizers-cpp-build -name "*.a" -type f
   ```

## Notes

- The script combines three libraries: `libtokenizers_cpp.a`, `libtokenizers_c.a`, and `libsentencepiece.a`
- All libraries are statically linked to avoid runtime dependencies
- The combined library includes all necessary tokenizer functionality for the CausalLM application