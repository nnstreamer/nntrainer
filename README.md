# NNtrainer

[![Code Coverage](http://nnsuite.mooo.com/nntrainer/ci/badge/codecoverage.svg)](http://nnsuite.mooo.com/nntrainer/ci/gcov_html/index.html)
![GitHub repo size](https://img.shields.io/github/repo-size/nnstreamer/nntrainer)
![GitHub issues](https://img.shields.io/github/issues/nnstreamer/nntrainer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/nnstreamer/nntrainer)

NNtrainer is Software Framework for Training Neural Network Models on Devices.

## Overview

NNtrainer is an Open Source Project. The aim of the NNtrainer is to develop Software Framework to train neural network model on embedded devices which has relatively limited resources. Rather than training the whole layers, NNtrainer trains only one or a few layers added after the feature extractor.

Even though it trains part of the neural network models, NNtrainer requires quite a lot of functionalities to train from common neural network frameworks. By implementing them, it is good enough to run several examples which can help to understand how it works. There are k-NN, Neural Network, Logistic Regression and Reinforcement Learning with CartPole in Applications directory and some of them use Mobilenet V2 with tensorflow-lite as a feature extractor. All of them tested on Galaxy S8 with Android and PC (Ubuntu 16.04).

## Maintainer
* [Jijoong Moon](https://github.com/jijoongmoon)
* [MyungJoo Ham](https://github.com/myungjoo)
* [Geunsik Lim](https://github.com/leemgs)

## Reviewers
* [Sangjung Woo](https://github.com/again4you)
* [Wook Song](https://github.com/wooksong)
* [Jaeyun Jung](https://github.com/jaeyun-jung)
* [Hyoungjoo Ahn](https://github.com/helloahn)
* [Parichay Kapoor](https://github.com/kparichay)
* [Dongju Chae](https://github.com/dongju-chae)
* [Gichan Jang](https://github.com/gichan-jang)
* [Yongjoo Ahn](https://github.com/anyj0527)
* [Jihoon Lee](https://github.com/zhoonit)

## Components

### NeuralNetwork

This is the component which controls neural network layers. Read the configuration file ([Iniparser](https://github.com/ndevilla/iniparser) is used to parse the configuration file.) and constructs Layers including Input and Output Layer, according to configured information by the user.
The most important role of this component is to activate forward / backward propagation. It activates inferencing and training of each layer while handling the data properly among them. There are properties to describe network model as below:

- **_Type:_** Network Type - Regression, k-NN, NeuralNetwork
- **_Layers:_** Name of Layers of Network Model
- **_Learning\_rate:_** Learning rate which is used for all Layers
- **_Decay\_rate:_** Rate for Exponentially Decayed Learning Rate
- **_Epoch:_** Max Number of Training Iteration.
- **_Optimizer:_** Optimizer for the Network Model - sgd, adam
- **_Activation:_** Activation Function - sigmoid , tanh
- **_Cost:_** Cost Function -
      mse(mean squared error), cross (cross entropy)
- **_Model:_** Name of Model. Weight Data is saved in the name of this.
- **_minibach:_** mini batch size
- **_beta1,beta2,epsilon:_** hyper parameters for the adam optimizer


### Layers

This component defines Layers which consist of Neural Network Model. Every neural network model must have one Input & Output Layer and other layers such as Fully Connected or Convolution can be added between them. (For now, supports Input & Output Layer, Fully Connected Layer.)

- **_Type:_** InputLayer, OutputLayer, FullyConnectedLayer
- **_Id:_** Index of Layer
- **_Height:_** Height of Weight Data (Input Dimension)
- **_Width:_** Width of Weight Data ( Hidden Layer Dimension)
- **_Bias\_zero:_** Boolean for Enable/Disable Bias


### Tensor

Tensor is responsible for the calculation of Layer. It executes the addition, division, multiplication, dot production, averaging of Data and so on. In order to accelerate the calculation speed, CBLAS (C-Basic Linear Algebra: CPU) and CUBLAS (CUDA: Basic Linear Algebra) for PC (Especially NVIDIA GPU)  for some of the operation. Later, these calculations will be optimized.

## Getting Started

### Prerequisites

The following dependencies are needed to compile / build / run.

*   gcc/g++ (>= 4.9, std=c++14 is used)
*   meson (>= 0.50.0)
*   blas library (CBLAS) (for CPU Acceleration, libopenblas is used for now)
*   cuda, cudart, cublas (should match the version) (GPU Acceleration on PC)
*   tensorflow-lite (>= 1.4.0)
*   libjsoncpp ( >= 0.6.0) (openAI Environment on PC)
*   libcurl3 (>= 7.47) (openAI Environment on PC)
*   libiniparser
*   libgtest (for testing)


### Give It a Go Build with Docker

You can use [docker image](https://hub.docker.com/r/lunapocket/nntrainer-build-env) to easily set up and try building.

To run the docker

```bash
$ docker pull lunapocket/nntrainer-build-env:ubuntu-18.04
$ docker run --rm -it  lunapocket/nntrainer-build-env:ubuntu-18.04
```

Inside docker...

```bash
$ cd /root/nntrainer
$ git pull # If you want to build with latest sources.
```

You can try build from now on without concerning about Prerequisites.

### How to Build

Download the source file by cloning the github repository.

```bash
$ git clone https://github.com/nnstreamer/nntrainer
```

After completing download the sources, you can find the several directories and files as below.

``` bash
$ cd nntrainer

$ ls -1
api
Applications
debian
doc
jni
LICENSE
meson.build
meson_options.txt
nntrainer
nntrainer.pc.in
packaging
README.md
test

$ git log --oneline
f1a3a05 (HEAD -> master, origin/master, origin/HEAD) Add more badges
37032a1 Add Unit Test Cases for Neural Network Initialization
181a003 lower case for layer type.
1eb399b Update clang-format
87f1de7 Add Unit Test for Neural Network
cd5c36e Add Coverage Test badge for nntrainer
...
```

You can find the source code of the core library in nntrainer/src. In order to build them, use [meson](https://mesonbuild.com/)
```bash
$ meson build
The Meson build system
Version: 0.50.1
Source dir: /home/wook/Work/NNS/nntrainer
Build dir: /home/wook/Work/NNS/nntrainer/build
Build type: native build
Project name: nntrainer
Project version: 0.0.1
Native C compiler: cc (gcc 7.5.0 "cc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0")
Native C++ compiler: c++ (gcc 7.5.0 "c++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0")
Build machine cpu family: x86_64
Build machine cpu: x86_64
...
Build targets in project: 11
Found ninja-1.8.2 at /usr/bin/ninja

$ ninja -C build
ninja: Entering directory `build'
[41/41] Linking target test/unittest/unittest_nntrainer_internal.
```

After completion of the build, the shared library, 'libnntrainer.so' and the static library, 'libnntrainer.a' will be placed in build/nntrainer.
```bash
$ ls build/nntrainer -1
d48ed23@@nntrainer@sha
d48ed23@@nntrainer@sta
libnntrainer.a
libnntrainer.so
```

In order to install them with related header files to your system, use the 'install' sub-command.
```bash
$ ninja -C build install
```
Then, you will find the libnntrainer.so and related .h files in /usr/local/lib and /usr/local/include directories.

By default, the command ```ninja -C build`` generates the five example application binaries (Classification, k-NN, LogisticRegression, ReinforcementLearning, and Training) you could try in build/Applications. For 'Training' as an example case,
```bash
$ ls build/Applications/Training/jni/ -1
e189c96@@nntrainer_training@exe
nntrainer_training
```

In order to run such example binaries, Tensorflow-lite is a prerequisite. If you are trying to run on the Android, it will automatically download tensorflow (1.9.0) and compile as static library. Otherwise, you need to install it by yourself.

### Running Examples


1. [Training](https://github.com/nnstreamer/nntrainer/blob/master/Applications/Training/README.md)

After build, run with following arguments
Make sure to put last '/' for the resources directory.
```bash
$./path/to/example ./path/to/settings.ini ./path/to/resource/directory/
```

To run the 'Training', for example, do as follows.

```bash
$ pwd
./nntrainer
$ LD_LIBRARY_PATH=./build/nntrainer ./build/Applications/Training/jni/nntrainer_training ./Applications/Training/res/Training.ini ./Applications/Training/res/
../../res/happy/happy1.bmp
../../res/happy/happy2.bmp
../../res/happy/happy3.bmp
../../res/happy/happy4.bmp
../../res/happy/happy5.bmp
../../res/sad/sad1.bmp
../../res/sad/sad2.bmp

...

```

## Open Source License

The nntrainer is an open source project released under the terms of the Apache License version 2.0.
