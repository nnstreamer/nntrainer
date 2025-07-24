---
title: Getting Started (Windows)
...

# Getting Started (Windows 10/11)

## Prerequisites

The following dependencies are needed to compile/build/run.

1. CMake (3.31.x)
2. Python (3.10 and above)
3. Meson (1.61 and above)
4. Visual Studio Build Tools 22 (17.13.x) 
- Install Desktop development with C++
- Additionally install: c++ Clang tools for Windows (needed for clang build only)

#### Clone the needed repositories

1. Clone repository:
```
git clone https://github.com/nnstreamer/nntrainer
```

2. Clone sub-repositories:
```
cd nntrainer
git submodule update --init --recursive
```

## How to build (using msvc)

1. Setup:
```
meson setup --native-file windows-native.ini builddir
```

2. Build:
```
meson compile -C builddir
```

## How to build (using clang)

Prerequisites:
Make sure that path to  llvm tools are added to "Path", example path: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin

1. Setup:
```
meson setup --native-file windows-native-clang.ini builddir
```

2. Build:
```
meson compile -C builddir
```

### How to run tests:

```
meson test -C builddir
```

### Known issues:

Windows build does not work with meson 1.8.0 due to some regression (/WX parameters is added to cmake submodules by default)
