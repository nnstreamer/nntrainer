# NNtrainer

[![Code Coverage](http://ec2-54-180-96-14.ap-northeast-2.compute.amazonaws.com/nntrainer/ci/badge/codecoverage.svg)](http://ec2-54-180-96-14.ap-northeast-2.compute.amazonaws.com/nntrainer/ci/gcov_html/index.html)

NNtrainer is Software Framework for Training Neural Network Models on Devices.

## Overview

NNtrainer is an Open Source Project. The aim of the NNtrainer is to develop Software Framework to train neural network model on embedded devices which has relatively limited resources. Rather than training the whole layers, NNtrainer trains only one or a few layers added after the feature extractor.

Even though it trains part of the neural network models, NNtrainer requires quite a lot of functionalities to train from common neural network frameworks. By implementing them, it is good enough to run several examples which can help to understand how it works. There are KNN, Neural Network, Logistic Regression and Reinforcement Learning with CartPole in Applications directory and some of them use Mobilenet V2 with tensorflow-lite as a feature extractor. All of them tested on Galaxy S8 with Android and PC (Ubuntu 16.04).

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

## Components

### NeuralNetwork

This is the component which controls neural network layers. Read the configuration file ([Iniparser](https://github.com/ndevilla/iniparser) is used to parse the configuration file.) and constructs Layers including Input and Output Layer, according to configured information by the user.
The most important role of this component is to activate forward / backward propagation. It activates inferencing and training of each layer while handling the data properly among them. There are properties to describe network model as below:

- **_Type:_** Network Type - Regression, KNN, NeuralNetwork
- **_Layers:_** Name of Layers of Network Model
- **_Learning\_rate:_** Learning rate which is used for all Layers
- **_Decay\_rate:_** Rate for Exponentially Decayed Learning Rate
- **_Epoch:_** Max Number of Training Iteration.
- **_Optimizer:_** Optimizer for the Network Model - sgd, adam
- **_Activation:_** Activation Function - sigmoid , tanh
- **_Cost:_** Cost Function -
      msr(mean square root error), categorical (for logistic regression), cross (cross entropy)
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

*	gcc/g++
*	CMake ( >= 2.8.3)
*	blas library ( CBLAS ) (for CPU Acceleration)
*	cuda, cudart, cublas (should match the version) (GPU Acceleration on PC)
*	tensorflow-lite (>=1.4.0)
*	jsoncpp ( >=0.6.0) (openAI Environment on PC) 
*	libcurl3 (>= 7.47 ) (openAI Environment on PC)

### How to Build

Download the source file by clone the github repository.

```bash
$ git clone https://github.com/nnsuite/nntrainer
```

After completing download the sources, you can find the several directories as below.

``` bash
$ ls
Applications  CMakeLists.txt  external  include  jni  LICENSE  package.pc.in
```

There are four applications tested on the Android and Ubuntu (16.04). All of them include the code in NeuralNet directory and has their own CMake file to compile. This is the draft version of the code and need more tailoring.
Just for the example, let\'s compile Training application. Once it is compiled and installed you will find the libnntrainer.so and related .h in /usr/local/lib and /usr/local/include directories.

``` bash
$ mkdir build
$ cd build
$ cmake  ..
$ make
$ sudo make install
```

And you could test the nntrainer with Examples in Applications. There are several examples you could try and you can compile with cmake(Ubuntu) and ndk-build (Android). Tensorflow-lite is pre-requite for this example, so you have to install it by your self. If you are trying to run on the Android, it will automatically download tensorflow (1.9.0) and compile as static library.
For the Training Examples, you can do like this:

```bash
$ cd Application/Training/jni
$ mkdir build
$ cd build
$ cmake  ..
$ make
$ ls 
CMakeCache.txt  CMakeFiles  cmake_install.cmake  Makefile  TransferLearning
$ ./Transfer_learning ../../res/Training.ini ../../res/
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
