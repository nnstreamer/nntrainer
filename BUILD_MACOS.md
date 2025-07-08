# Building nntrainer on macOS x86

This guide explains how to build nntrainer on macOS with x86 CPU after applying the macOS compatibility patches.

## Prerequisites

### Required Tools

1. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

2. **Homebrew** (recommended package manager):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Meson Build System**:
   ```bash
   brew install meson ninja
   ```

4. **CMake** (for subprojects):
   ```bash
   brew install cmake
   ```

### Optional Dependencies

5. **OpenBLAS** (for BLAS acceleration - recommended):
   ```bash
   brew install openblas
   ```

6. **pkg-config**:
   ```bash
   brew install pkg-config
   ```

7. **iniparser**:
   ```bash
   brew install iniparser
   ```

## Build Process

### 1. Clone the Repository
```bash
git clone https://github.com/nnstreamer/nntrainer.git
cd nntrainer
```

### 2. Apply macOS Compatibility Patches
The following changes have been made to support macOS:

- Fixed hardcoded `.so` extensions to use `.dylib` on macOS
- Updated OpenBLAS library path detection for macOS
- Added macOS-specific OpenCL library loading
- Enhanced platform detection in build system

### 3. Configure Build
```bash
meson setup builddir --buildtype=release
```

### 4. Build Options

You can customize the build with various options:

#### Basic Build (minimal dependencies):
```bash
meson setup builddir \
    --buildtype=release \
    --platform=none \
    -Denable-blas=false \
    -Denable-test=false \
    -Denable-app=false
```

#### Full Build (with BLAS acceleration):
```bash
meson setup builddir \
    --buildtype=release \
    --platform=none \
    -Denable-blas=true \
    -Denable-test=true \
    -Denable-app=true \
    -Denable-logging=true
```

#### Performance Build (optimized):
```bash
meson setup builddir \
    --buildtype=release \
    --platform=none \
    -Denable-blas=true \
    -Denable-openmp=true \
    -Denable-fp16=false \
    -Dnntr-num-threads=4 \
    -Domp-num-threads=4
```

### 5. Compile
```bash
cd builddir
ninja
```

### 6. Install (optional)
```bash
sudo ninja install
```

## Platform-Specific Notes

### macOS x86 Optimizations
The build system automatically detects x86_64 architecture and enables:
- `-march=native` for CPU-specific optimizations
- `-mavx2` and `-mfma` for SIMD acceleration

### Library Extensions
On macOS, the build system now correctly uses:
- `.dylib` for shared libraries (instead of `.so`)
- Proper framework paths for system libraries like OpenCL

## Troubleshooting

### Common Issues

1. **OpenBLAS not found**:
   ```bash
   export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
   ```

2. **Compiler warnings**:
   ```bash
   meson setup builddir -Dwerror=false
   ```

3. **Missing dependencies**:
   Install missing packages via Homebrew:
   ```bash
   brew install <package-name>
   ```

### Build Configuration
To see all available build options:
```bash
meson configure
```

To reconfigure an existing build:
```bash
meson configure builddir -Doption=value
```

## Testing

Run the test suite (if built with `-Denable-test=true`):
```bash
cd builddir
ninja test
```

## Example Applications

If you built with `-Denable-app=true`, example applications will be available in:
```
builddir/Applications/
```

## Performance Notes

For optimal performance on macOS x86:

1. **Enable BLAS**: `-Denable-blas=true`
2. **Use OpenMP**: `-Denable-openmp=true`
3. **Set thread count**: `-Dnntr-num-threads=<num_cores>`
4. **Release build**: `--buildtype=release`

## Apple Silicon (M1/M2) Support

While this patch specifically targets x86 macOS, the codebase also supports ARM64 (Apple Silicon). For M1/M2 Macs, you may want to:

1. Use `arch=aarch64` instead of `arch=x86_64`
2. Consider enabling FP16 optimizations: `-Denable-fp16=true`
3. The same `.dylib` fixes apply to Apple Silicon as well

## Next Steps

After successful compilation, you can:

1. Use the C API: Include `#include <nntrainer.h>`
2. Use the C++ API: Include relevant headers from `nntrainer/`
3. Run example applications from `Applications/` directory
4. Integrate with your own projects by linking against `libnntrainer.dylib`

## Contributing

If you encounter any macOS-specific issues, please:

1. Check this documentation first
2. Search existing issues on GitHub
3. Report new issues with detailed system information:
   ```bash
   system_profiler SPSoftwareDataType SPHardwareDataType
   ```