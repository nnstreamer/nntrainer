---
title: How to run examples - Android
...

# How to run examples - Android

### Install NDK-Build

Prepare NDK tool chain to build.
Download the ndk tool chain from official site of android which is https://developer.android.com/ndk/downloads

Choose the ndk package which is proper to you platform. In this guide, we use Linux 64-bit(x86). We choose latest version (currently r21d).
Once the download is finished, decompress the package and set the library path properly. You can also set the LD_LIBRARAY_PATH in bashrc.

```bash
$ ls
android-ndk-r21d-linux-x86_64.zip
$ unzip android-ndk-r21d-linux-86_64.zip
$ ls
android-ndk-r21d
$ export LD_LIBRARY_PATH = $LD_LIBRARY_PATH:${PWD}/android-ndk-r21d/
```

### Build NNTrainer with NDK-Build
Once you install NDK package, you can build the nntrainer.so for android as below.

```bash
$ ls
api                 CONTRIBUTING.md  index.md  MAINTAINERS.md     nnstreamer       packaging
Applications        debian           jni       meson.build        nntrainer        README.md
CODE_OF_CONDUCT.md  docs             LICENSE   meson_options.txt  nntrainer.pc.in  test
$
$ export NNTRAINER_ROOT=${PWD}
$
$ cd ./jni
Android.mk      prepare_iniparser.sh  prepare_tflite.sh
Application.mk  prepare_openblas.sh
$ ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)
Android.mk:19: INIPARSER_ROOT is not defined!
Android.mk:20: INIPARSER SRC is going to be downloaded!
Cloning into 'iniparser'...
~/nntrainer_new/jni ~/nntrainer_new/jni PREPARING ini parser at . ~/nntrainer_new/jni
Android.mk:35: BUILDING TFLITE BACKBONE !
Android.mk:40: TENSORFLOW_ROOT is not defined!
Android.mk:41: TENSORFLOW SRC is going to be downloaded!
PREPARING TENSORFLOW 1.13.1 at . ~/nntrainer_new/jni ~/nntrainer_new/jni [TENSORFLOW-LITE] Download tensorflow-1.13.1 [TENSORFLOW-LITE] Finish downloading tensorflow-1.13.1 [TENSORFLOW-LITE] untar tensorflow-1.13.1 ~/nntrainer_new/jni
[arm64-v8a] Compile++      : nntrainer <= tflite_layer.cpp
In file included from ././../nntrainer/layers/tflite_layer.cpp:15:
In file included from ./../nntrainer/layers/tflite_layer.h:18:
In file included from ./../nntrainer/layers/layer_internal.h:31:
...
$ ls
Android.mk      iniparser  obj                   prepare_openblas.sh  tensorflow-1.13.1
Application.mk  libs       prepare_iniparser.sh  prepare_tflite.sh
$ ls libs/arm64-v8a
libcapi-nntrainer.so  libccapi-nntrainer.so  libc++_shared.so  libnntrainer.so
$ mv libs ../
```

### Build NNTrainer Applications with NDK-Build
Now you are ready to build nntrainer application in ${NNTRAINER_ROOT}/Applications
If you want to build Application/LogisticRegration,

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
