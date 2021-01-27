---
title: How to run examples
...

# How to run examples

## Preparing NNTrainer for execution

### Use PPA

If you don't want to build build binaries, you can directly download from PPA with daily releases.

```bash
sudo add-apt-repository ppa:nnstreamer/ppa
sudo apt-get update
sudo apt-get install nntrainer
```

Note that this may install Tensorflow-Lite packaged by us.

## Build examples (Ubuntu 18.04)

Refer <https://github.com/nnstreamer/nntrainer/blob/master/docs/getting-started.md> for more info.

Install related packages before building nntrainer and examples.

1. gcc/g++ >=4.9 ( std=c++14 is used )
2. meson >= 0.50
3. libopenblas-dev and base
4. tensorflow-lite >=1.14.0
5. libiniparser
6. libjsoncpp >=0.6.0 ( if you wand to use open AI )
7. libcurl3 >=7.47 ( if you wand to use open AI )
8. libgtest ( for testing )

Important build options (meson)

1. enable-tizen : default false, add option for tizen build (-Denable-tizen=false)
2. enable-blas : default true, add option to enable blas (-Denable-blas=true)
3. enable-app : default true, add option to enable Applications (-Denable-app=true)
4. install-app : default true, add option to install Applications (-Dinstall-app=true)
5. use_gym : default false, add option to use openAI gym (-Duse_gym=false)
6. enable-capi : default true, add option to install C-API (-Denable-capi=true)
7. enable-test : default true, add option to test (-Denable-test=true)
8. enable-logging : default true, add option to do logging (-Denable-logging=true)
9. enable-tizen-feature-check : default true, add option to enable tizen feature check (-Denable-tizen-feature-check=true)

For example, to build and install NNTrainer and C-API,

```bash
meson --prefix=${NNTRAINER_ROOT} --sysconfdir=${NNTRAINER_ROOT} --libdir=lib --bindir=bin --includedir=include -Denable-capi=true build
```

Build source code

```bash
# Set your own path to install libraries and header files
$ sudo vi ~/.bashrc

export NNTRAINER_ROOT=$HOME/nntrainer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NNTRAINER_ROOT/lib
# Include NNStreamer headers and libraries
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$NNTRAINER_ROOT/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NNTRAINER_ROOT/include
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$NNTRAINER_ROOT/lib/pkgconfig

$ source ~/.bashrc

# Download source, then compile it.
# Build and install nntrainer
$ git clone https://github.com/nnstreamer/nntrainer.git nntrainer.git
$ meson --prefix=${NNTRAINER_ROOT} --sysconfdir=${NNTRAINER_ROOT} --libdir=lib --bindir=bin --includedir=include build
$ ninja -C build install
$ cd ..
```
