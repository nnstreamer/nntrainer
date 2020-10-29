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

### Supported Layers

This component defines Layers which consist of Neural Network Model. Layers has own properties to be set.

 | Keyword | Layer Name | Description |
 |:-------:|:---:|:---|
 |  conv2d | Convolution 2D |Convolution 2-Dimentional Layer |
 |  pooling2d | Pooling 2D |Pooling 2-Dimentional Layer. Support average / max / global average / global max pooing |
 | flatten | Flatten | Flatten Layer |
 | fully_connected | Fully Connected | Fully Connected Layer |
 | input | Input | Input Layer.  This is not always requied. |
 | batch_normalization | Batch Normalization Layer | Batch Normalization Layer. |
 | loss layer | loss layer | hidden from users |
 | activation | activaiton layer | set by layer property |

### Supported Optimizers

NNTrainer Provides

 | Keyward | Optimizer Name | Description |
 |:-------:|:---:|:---:|
 | sgd | Stochastic Gradient Decent | - |
 | adam | Adaptive Moment Estimation | - |

### Supported Loss

NNTrainer provides

 | Keyward | Loss Name | Description |
 |:-------:|:---:|:---:|
 | mse | Mean squared Error | - |
 | cross | Cross Entropy - sigmoid | if activation last layer is sigmoid |
 | cross | Cross Entropy - softmax | if activation last layer is softmax |

### Supported Activations

NNTrainer provides

 | Keyward | Loss Name | Description |
 |:-------:|:---:|:---|
 | tanh | tanh function | set as layer property |
 | sigmoid | sigmoid function | set as layer property |
 | relu | relu function | set as layer propery |
 | softmax | softmax function | set as layer propery |
 | weight_initializer | Weight Initialization | Xavier(Normal/Uniform), LeCun(Normal/Uniform),  HE(Normal/Unifor) |
 | weight_regularizer | weight decay ( L2Norm only ) | needs set weight_regularizer_param & type |
 | learnig_rate_decay | learning rate decay | need to set step |

### Tensor

Tensor is responsible for the calculation of Layer. It executes the addition, division, multiplication, dot production, averaging of Data and so on. In order to accelerate the calculation speed, CBLAS (C-Basic Linear Algebra: CPU) and CUBLAS (CUDA: Basic Linear Algebra) for PC (Especially NVIDIA GPU)  for some of the operation. Later, these calculations will be optimized.
Currently we supports lazy calculation mode to reduce copy of tensors during calcuation.

 | Keyward | Description |
 |:-------:|:---:|
 | 4D Tensor | B, C, H, W|
 | Add/sub/mul/div | - |
 | sum, average, argmax | - |
 | Dot, Transpose | - |
 | normalization, standardization | - |
 | save, read | - |

### Others

NNTrainer provides

 | Keyward | Loss Name | Description |
 |:-------:|:---:|:---|
 | weight_initializer | Weight Initialization | Xavier(Normal/Uniform), LeCun(Normal/Uniform),  HE(Normal/Unifor) |
 | weight_regularizer | weight decay ( L2Norm only ) | needs set weight_regularizer_constant & type |
 | learnig_rate_decay | learning rate decay | need to set step |

### APIs
Currently we privde [C APIs](https://github.com/nnstreamer/nntrainer/blob/master/api/capi/include/nntrainer.h) for Tizen. C++ API also provides soon.


### Examples for NNTrainer

#### [Custom Shortcut Application](https://github.com/nnstreamer/nntrainer/tree/master/Applications/Tizen_native/CustomShortcut)


This is demo application which enable user defined custom shortcut on galaxy watch.

#### [MNIST Example](https://github.com/nnstreamer/nntrainer/tree/master/Applications/MNIST)

This is example to train mnist dataset. It consists two convolution 2d layer, 2 pooling 2d layer, flatten layer and fully connected layer.

#### [Reinforcement Learning Example](https://github.com/nnstreamer/nntrainer/tree/master/Applications/ReinforcementLearning/DeepQ)

This is reinforcement learning example with cartpole game. It is using deepq alogrightm.

#### [Classification for cifar 10](https://github.com/nnstreamer/nntrainer/tree/master/Applications/TransferLearning/CIFAR_Classification)

This is Transfer learning example with cifar 10 data set. TFlite is used for feature extractor and modify last layer (fully connected layer) of network.

#### ~Tizen CAPI Example~

~This is for demonstrate c api for tizen. It is same transfer learing but written with tizen c api.~
Deleted instead moved to a [test](https://github.com/nnstreamer/nntrainer/blob/master/test/tizen_capi/unittest_tizen_capi.cpp)

#### [KNN Example](https://github.com/nnstreamer/nntrainer/tree/master/Applications/KNN)

This is Transfer learning example with cifar 10 data set. TFlite is used for feature extractor and compared with KNN

#### [Logistic Regression Example](https://github.com/nnstreamer/nntrainer/tree/master/Applications/LogisticRegression)

This is simple logistic regression example using nntrainer.

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

### How to Test

1. Unittest
Meson build `enable-test` set to true

```bash
$ echo $(pwd)
(project root)

$ meson build -Denable-test=true
The Meson build system
Version: 0.54.3
...
Configuring capi-nntrainer.pc using configuration
Run-time dependency GTest found: YES (building self)
Build targets in project: 17

Found ninja-1.10.0.git.kitware.jobserver-1 at /home/jlee/.local/bin/ninja

$ ninja -C build test
[79/79] Running all tests.
 1/12 unittest_tizen_capi            OK             8.86s
 2/12 unittest_tizen_capi_layer      OK             0.05s
 3/12 unittest_tizen_capi_optimizer  OK             0.01s
 4/12 unittest_tizen_capi_dataset    OK             0.03s
 5/12 unittest_nntrainer_activations OK             0.03s
 6/12 unittest_nntrainer_internal    OK             0.23s
 7/12 unittest_nntrainer_layers      OK             0.22s
 8/12 unittest_nntrainer_lazy_tensor OK             0.04s
 9/12 unittest_nntrainer_tensor      OK             0.04s
10/12 unittest_util_func             OK             0.05s
11/12 unittest_databuffer_file       OK             0.12s
12/12 unittest_nntrainer_modelfile   OK             2.22s

Ok:                 12
Expected Fail:      0
Fail:               0
Unexpected Pass:    0
Skipped:            0
Timeout:            0
```

if you want to run particular test

```bash
$ meson -C build test $(test name)
```

2. Sample app test
NNTrainer provides extensive sample app running test.

Meson build with `enable-app` set to true
```bash
$ echo $(pwd)
(project root)

$ meson build -Denable-app=true
The Meson build system
Version: 0.54.3
...
Configuring capi-nntrainer.pc using configuration
Run-time dependency GTest found: YES (building self)
Build targets in project: 17

Found ninja-1.10.0.git.kitware.jobserver-1 at /home/jlee/.local/bin/ninja

$ ninja -C build test
...
 1/21 app_classification             OK             3.59s
 2/21 app_classification_func        OK             42.77s
 3/21 app_knn                        OK             4.81s
 4/21 app_logistic                   OK             14.11s
 5/21 app_DeepQ                      OK             30.30s
 6/21 app_training                   OK             38.36s
 7/21 app_classification_capi_ini    OK             32.65s
 8/21 app_classification_capi_file   OK             32.04s
 9/21 app_classification_capi_func   OK             29.13s
 ...
```

if you want to run particular example only

```bash
$ meson -C build test $(test name) #app_classification_capi_func
```

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

## Contributing

Contributions are welcome! Please see our [Contributing](https://github.com/nnstreamer/nntrainer/blob/main/docs/contributing.md) Guide for more details.

[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/0)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/0)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/1)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/1)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/2)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/2)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/3)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/3)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/4)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/4)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/5)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/5)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/6)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/6)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/7)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/7)
