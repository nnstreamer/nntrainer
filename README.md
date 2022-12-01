# NNtrainer

[![Code Coverage](http://ci.nnstreamer.ai/nntrainer/ci/badge/codecoverage.svg)](http://ci.nnstreamer.ai/nntrainer/ci/gcov_html/index.html)
![GitHub repo size](https://img.shields.io/github/repo-size/nnstreamer/nntrainer)
![GitHub issues](https://img.shields.io/github/issues/nnstreamer/nntrainer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/nnstreamer/nntrainer)
<a href="https://scan.coverity.com/projects/nnstreamer-nntrainer">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/22512/badge.svg"/>
</a>
[![DailyBuild](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/daily_build_test_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)

NNtrainer is a Software Framework for training Neural Network models on devices.

## Overview

NNtrainer is an Open Source Project. The aim of the NNtrainer is to develop a Software Framework to train neural network models on embedded devices which have relatively limited resources. Rather than training whole layers of a network from the scratch, NNtrainer finetunes the neural network model on device with user data for the personalization.

Even if NNtariner runs on device, it provides full functionalities to train models and also utilizes limited device resources efficiently. NNTrainer is able to train various machine learning algorithms such as k-Nearest Neighbor (k-NN), Neural Networks, Logistic Regression, Reinforcement Learning algorithms, Recurrent network and more. We also provide examples for various tasks such as Few-shot learning, ResNet, VGG, Product Rating and more will be added. All of these were tested on Samsung Galaxy smart phone with Android and PC (Ubuntu 18.04/20.04).

[ NNTrainer: Light-Weight On-Device Training Framework ](https://arxiv.org/pdf/2206.04688.pdf), arXiv, 2022 <br />
[ NNTrainer: Towards the on-device learning for personalization ](https://www.youtube.com/watch?v=HWiV7WbIM3E), Samsung Software Developer Conference 2021 (Korean) <br />
[ NNTrainer: Personalize neural networks on devices! ](https://www.youtube.com/watch?v=HKKowY78P1A), Samsung Developer Conference 2021 <br />
[ NNTrainer: "On-device learning" ](https://www.youtube.com/embed/Jy_auavraKg?start=4035&end=4080), Samsung AI Forum 2021

## Official Releases

|     | [Tizen](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [Ubuntu](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) | Android/NDK Build |
| :-- | :--: | :--: | :--: |
|     | 6.0M2 and later | 18.04 | 9/P |
| arm | [![armv7l badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/tizen.armv7l_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/) | Available  | Ready |
| arm64 |  [![aarch64 badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/tizen.aarch64_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/) | Available  | [![android badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/arm64_v8a_android_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/) |
| x64 | [![x64 badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/tizen.x86_64_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)  | [![ubuntu badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/ubuntu_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)  | Ready  |
| x86 | [![x86 badge](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/badge/tizen.i586_result_badge.svg)](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)  | N/A  | N/A  |
| Publish | [Tizen Repo](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [PPA](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) |   |
| API | C (Official) | C/C++ | C/C++  |

- Ready: CI system ensures build-ability and unit-testing. Users may easily build and execute. However, we do not have automated release & deployment system for this instance.
- Available: binary packages are released and deployed automatically and periodically along with CI tests.
- [Daily Release](http://ci.nnstreamer.ai/nntrainer/ci/daily-build/build_result/)
- SDK Support: Tizen Studio (6.0 M2+)

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
* [Hyeonseok Lee](https://github.com/lhs8928)
* [Mete Ozay](https://github.com/meteozay)
* [Hyunil Park](https://github.com/songgot)
* [Jiho Chu](https://github.com/jihochu)
* [Yelin Jeong](https://github.com/niley7464)
* [Donghak Park](https://github.com/DonghakPark)


## Components

### Supported Layers

This component defines layers which consist of a neural network model. Layers have their own properties to be set.

 | Keyword | Layer Class Name | Description |
 |:-------:|:---:|:---|
 | conv1d | Conv1DLayer | Convolution 1-Dimentional Layer |
 | conv2d | Conv2DLayer |Convolution 2-Dimentional Layer |
 | pooling2d | Pooling2DLayer |Pooling 2-Dimentional Layer. Support average / max / global average / global max pooling |
 | flatten | FlattenLayer | Flatten layer |
 | fully_connected | FullyConnectedLayer | Fully connected layer |
 | input | InputLayer | Input Layer.  This is not always required. |
 | batch_normalization | BatchNormalizationLayer | Batch normalization layer |
 | layer_normalization | LayerNormalizationLayer | Layer normalization layer |
 | activation | ActivationLayer | Set by layer property |
 | addition | AdditionLayer | Add input input layers |
 | attention | AttentionLayer | Attenstion layer |
 | centroid_knn | CentroidKNN | Centroid K-nearest neighbor layer |
 | concat | ConcatLayer | Concatenate input layers |
 | multiout | MultiOutLayer | Multi-Output Layer |
 | backbone_nnstreamer | NNStreamerLayer | Encapsulate NNStreamer layer |
 | backbone_tflite | TfLiteLayer | Encapsulate tflite as a layer |
 | permute | PermuteLayer | Permute layer for transpose |
 | preprocess_flip | PreprocessFlipLayer | Preprocess random flip layer |
 | preprocess_l2norm | PreprocessL2NormLayer | Preprocess simple l2norm layer to normalize |
 | preprocess_translate | PreprocessTranslateLayer | Preprocess translate layer |
 | reshape | ReshapeLayer | Reshape tensor dimension layer |
 | split | SplitLayer | Split layer |
 | dropout | DropOutLayer | Dropout Layer |
 | embedding | EmbeddingLayer | Embedding Layer |
 | positional_encoding | PositionalEncodingLayer | Positional Encoding Layer |
 | rnn | RNNLayer | Recurrent Layer |
 | rnncell | RNNCellLayer | Recurrent Cell Layer |
 | gru | GRULayer | Gated Recurrent Unit Layer |
 | grucell | GRUCellLayer | Gated Recurrent Unit Cell Layer |
 | lstm | LSTMLayer | Long Short-Term Memory Layer |
 | lstmcell | LSTMCellLayer | Long Short-Term Memory Cell Layer |
 | zoneoutlstmcell | ZoneoutLSTMCellLayer | Zoneout Long Short-Term Memory Cell Layer |
 | time_dist | TimeDistLayer | Time distributed Layer |
 | multi_head_attention | MultiHeadAttentionLayer | Multi Head Attention Layer |


### Supported Optimizers

NNTrainer Provides

 | Keyword | Optimizer Name | Description |
 |:-------:|:---:|:---:|
 | sgd | Stochastic Gradient Decent | - |
 | adam | Adaptive Moment Estimation | - |

 | Keyword | Leanring Rate | Description |
 |:-------:|:---:|:---:|
 | exponential | exponential learning rate decay | - |
 | constant | constant learning rate | - |
 | step | step learning rate | - |

### Supported Loss Functions

NNTrainer provides

 | Keyword | Class Name | Description |
 |:-------:|:---:|:---:|
 | cross_sigmoid | CrossEntropySigmoidLossLayer | Cross entropy sigmoid loss layer |
 | cross_softmax | CrossEntropySoftmaxLossLayer | Cross entropy softmax loss layer |
 | constant_derivative | ConstantDerivativeLossLayer | Constant derivative loss layer |
 | mse | MSELossLayer | Mean square error loss layer |
 | kld | KLDLossLayer | Kullback-Leibler Divergence loss layer |

### Supported Activation Functions

NNTrainer provides

 | Keyword | Loss Name | Description |
 |:-------:|:---:|:---|
 | tanh | tanh function | set as layer property |
 | sigmoid | sigmoid function | set as layer property |
 | relu | relu function | set as layer property |
 | softmax | softmax function | set as layer property |

### Tensor

Tensor is responsible for calculation of a layer. It executes several operations such as addition, division, multiplication, dot production, data averaging and so on. In order to accelerate  calculation speed, CBLAS (C-Basic Linear Algebra: CPU) and CUBLAS (CUDA: Basic Linear Algebra) for PC (Especially NVIDIA GPU) are implemented for some of the operations. Later, these calculations will be optimized.
Currently, we support lazy calculation mode to reduce complexity for copying tensors during calculations.

 | Keyword | Description |
 |:-------:|:---:|
 | 4D Tensor | B, C, H, W|
 | Add/sub/mul/div | - |
 | sum, average, argmax | - |
 | Dot, Transpose | - |
 | normalization, standardization | - |
 | save, read | - |

### Others

NNTrainer provides

 | Keyword | Loss Name | Description |
 |:-------:|:---:|:---|
 | weight_initializer | Weight Initialization | Xavier(Normal/Uniform), LeCun(Normal/Uniform),  HE(Normal/Unifor) |
 | weight_regularizer | weight decay ( L2Norm only ) | needs set weight_regularizer_param & type |

### APIs
Currently, we provide [C APIs](https://github.com/nnstreamer/nntrainer/blob/master/api/capi/include/nntrainer.h) for Tizen. [C++ APIs](https://github.com/nnstreamer/nntrainer/blob/master/api/ccapi/include) are also provided for other platform. Java & C# APIs will be provided soon.


## [Getting Started](https://github.com/nnstreamer/nntrainer/blob/main/docs/getting-started.md)

Instructions for installing NNTrainer.

### [Running Examples](https://github.com/nnstreamer/nntrainer/blob/main/docs/how-to-run-examples.md)

Instructions for preparing NNTrainer for execution

### [Examples for NNTrainer](https://github.com/nnstreamer/nntrainer/tree/main/Applications)

NNTrainer example for a variety of networks

## Open Source License

The NNtrainer is an open source project released under the terms of the Apache License version 2.0.

## Contributing

Contributions are welcome! Please see our [Contributing](https://github.com/nnstreamer/nntrainer/blob/main/docs/contributing.md) Guide for more details.

[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/0)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/0)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/1)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/1)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/2)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/2)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/3)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/3)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/4)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/4)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/5)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/5)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/6)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/6)[![](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/7)](https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/7)
