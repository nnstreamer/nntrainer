---
title: How to run examples - Android
...

# How to run examples - Android

## Install NDK-Build

Prepare NDK tool chain to build.
Download the ndk tool chain from official site of android which is https://developer.android.com/ndk/downloads

Choose the ndk package which is proper to your platform. In this guide, we use Linux 64-bit(x86). We choose latest version (currently r21d).
Once the download is finished, decompress the package and set the library path properly. You can also set the LD_LIBRARAY_PATH in bashrc.

```bash
$ ls
android-ndk-r21d-linux-x86_64.zip
$ unzip android-ndk-r21d-linux-86_64.zip
$ ls
android-ndk-r21d
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/android-ndk-r21d/
$ export PATH=$PATH:${PWD}/android-ndk-r21d/
$ export ANDROID_NDK=${PWD}/android-ndk-r21d/
```

## Build NNTrainer with NDK-Build
Once you install NDK package, you can build the nntrainer.so for android as below.
Currently, the APP_ABI is set arm64-v8a. If you want to use armeabi-v7a, you have to change in Application.mk file.
There is 2 way to build nntrainer for android

### Build using shell script

```bash
$ ls
api                 CONTRIBUTING.md  index.md  MAINTAINERS.md     nnstreamer        nntrainer.pc.in  RELEASE.md
Applications        debian           jni       meson.build        nntrainer         packaging        test
CODE_OF_CONDUCT.md  docs             LICENSE   meson_options.txt  nntrainer.ini.in  README.md        tools
$
$ ./tools/package_android.sh
$ ls builddir/android_build_result
Android.mk  conf  examples  include  lib
$ ls builddir/android_build_result/libs/arm64-v8a
libcapi-nntrainer.so  libccapi-nntrainer.so  libc++_shared.so  libnnstreamer-native.so  libnntrainer.so
```

### Build using meson

```bash
$ ls
api                 CONTRIBUTING.md  index.md  MAINTAINERS.md     nnstreamer        nntrainer.pc.in  RELEASE.md
Applications        debian           jni       meson.build        nntrainer         packaging        test
CODE_OF_CONDUCT.md  docs             LICENSE   meson_options.txt  nntrainer.ini.in  README.md        tools
$ meson build -Dplatform=android
The Meson build system
Version: 0.53.2
Source dir: /home/hs89lee/workspace/git/nntrainer
Build dir: /home/hs89lee/workspace/git/nntrainer/build
Build type: native build
Project name: nntrainer
Project version: 0.3.0
C compiler for the host machine: cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0")
C linker for the host machine: cc ld.bfd 2.34
C++ compiler for the host machine: c++ (gcc 9.4.0 "c++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0")
...
$ ninja -C build
ninja: Entering directory 'build'
[2/2] Generating ndk-build with a custom command.
[arm64-v8a] Prebuilt       : libc++_shared.so <= <NDK>/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/
[arm64-v8a] Prebuilt       : libnnstreamer-native.so <= /home/hs89lee/workspace/git/nntrainer/build/ml-api-inference/lib/arm64-v8a/
[arm64-v8a] Compile++      : nntrainer <= nntrainer_logger.cpp
[arm64-v8a] Compile++      : nntrainer <= remap_realizer.cpp
...
$ pushd build/jni
$ ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)
[arm64-v8a] Prebuilt       : libc++_shared.so <= <NDK>/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/
[arm64-v8a] Prebuilt       : libnnstreamer-native.so <= /home/hs89lee/workspace/git/nntrainer/build/ml-api-inference/lib/arm64-v8a/
[arm64-v8a] Compile++      : nntrainer <= nntrainer_logger.cpp
[arm64-v8a] Compile++      : nntrainer <= remap_realizer.cpp
[arm64-v8a] Compile++      : nntrainer <= previous_input_realizer.cpp
...
$ popd
$ ls build/jni/libs/arm64-v8a
libcapi-nntrainer.so  libccapi-nntrainer.so  libc++_shared.so  libnnstreamer-native.so  libnntrainer.so
```

## Build NNTrainer Applications with NDK-Build
Now you are ready to build nntrainer application in ${NNTRAINER_ROOT}/Applications
If you want to build Application/LogisticRegression,

```bash
$ cd ${NNTRAINER_ROOT}
$ cd Applications/LogisticRegression/jni
$ ls
Android.mk  Application.mk  CMakeLists.txt  main.cpp  meson.build
$ ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)
[arm64-v8a] Prebuilt       : libnntrainer.so <= /libs/arm64-v8a/
[arm64-v8a] Prebuilt       : libc++_shared.so <= <NDK>/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/
[arm64-v8a] Install        : libnntrainer.so => libs/arm64-v8a/libnntrainer.so
[arm64-v8a] Install        : libc++_shared.so => libs/arm64-v8a/libc++_shared.so
[arm64-v8a] Compile++      : nntrainer_logistic <= main.cpp
In file included from ./main.cpp:36:
...
$ ls
Android.mk  Application.mk  CMakeLists.txt  libs  main.cpp  meson.build  obj
$ ls libs/arm64-v8a
libc++_shared.so  libnntrainer.so  nntrainer_logistic
```

Now you can find the execution binary of nntrainer_logistic.
You can copy execution binary and nntrainer.so libraries in a proper place of your android device using ADB.

Then you can run it using ADB shell.


## Troubleshooting

1. Check version of your ndk before executing the above commands.
2. `ANDROID_NDK` not defined while bulding the particular application such as Logistic regression.

```bash
$ ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)
Android.mk:7: *** ANDROID_NDK is not defined!.  Stop.
```
Fix: Check if the environment variable is loaded with `echo $ANDROID_NDK`. If not, then define it using `export` .
