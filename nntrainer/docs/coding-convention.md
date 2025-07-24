---
title: Coding Convention
...

# Coding Convention

## C headers (.h)

You may indent differently from what clang-format does. You may also break the 80-column rule with header files.

Except the two, you are required to follow the general coding styles mandated by clang-format.

## C/C++ files (.cpp, .c)

Use .h for headers and .cpp / .c for source.
You have to use clang-format with the given [.clang-format](https://github.com/nnstreamer/nntrainer/blob/main/.clang-format) file


## Other files

- [Java] TBD
- [Python] TBD
- [Bash] TBD


# File Locations

## Directory structure of nntrainer.git

- **api**: API definitions and implementations
    - **capi**: C-APIs (Tizen and others)
    - **ccapi**: C++-APIs
- **Applications**: Examples for NNtrainer
- **debian**: Debian/Ubuntu packaging files
- **docs**: Documentations
- **jni**: Android/Java build scripts.
- **nnstreamer**: NNStreamer sub-filter codes for NNTrainer
- **nntrainer**: All core NNTrainer codes are located here
- **packaging**: Tizen RPM build scripts. OpenSUSE/Redhat Linux may reuse this.
- **test**: Unit test cases. We have GTEST test cases. There are subdirectories, which are groups of unit test cases.
- **tools**: Various developmental tools and scripts of NNTrainer.

## Related git repositories

- [NNStreamer](https://github.com/nnstreamer/nnstreamer)
- [TAOS-CI, CI Service for On-Device AI Systems](https://github.com/nnstreamer/TAOS-CI)
