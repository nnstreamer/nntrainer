# Classification

## Introduction

This example demonstrates the transfer learning with MobileNetV2 which pre-trained for ImageNet. "mobilenetv2.tflite" is used for feature extractor which is removed last fully connected layer manually.

After feature extraction, it is used for training fully connected layer. To take the feature as an input, we add input layer as the first layer.

Total number of data for training is 10 (number of class) * 100 (number of image for each class)

Classification.ini file (res directory) is input which has configuration and hyper-parameters of network for NNTrainer.

This example has two options to train.

### 1. Train with training set files ( main.cpp )

- This option is activated if [ DataSet ] section is configured in configuration file.
- This assumes that training set saved as an file is available.
   For the data file, the ith image feature data (62720 x sizeof(float)) + label (10 x sizeof(float)) must be at i x ((62720 x sizeof(float) + label(10 x sizeof(float)) byte position.
- The input image (bmp format : 32x32x3) is stored as in ExtractFeatures() at "data_path" directory defined as argv #1.

```c++
   string total_label[10] = {"airplane", "automobile", "bird", "cat", "deer",
                           "dog", "frog", "horse", "ship", "truck"};
```

- Training dataset files are generated using 'ExtractFeatures()' function in main.cpp.
- Once the dataset files are generated, NNTrainer start to read and feed the buffer to train.

### 2. Train with training data generator ( main_func.cpp )

- This option is activated if there is no [ DataSet ] section defined in the configuration file.
- Expect to get the generator function which defined by user.
- The sample implementation just loads the data from the training data set file (from option 1).
- training data set generator is getBatch_train(),  getBatch_val() for validation. test data set is not used.

## How to run examples

### Preparing NNTrainer

<https://github.com/nnstreamer/nntrainer/blob/master/docs/getting-started.md>

### Write Configuration file

you can find detail explanation about each keyword in
<https://github.com/nnstreamer/nntrainer/blob/master/docs/configuration-ini.md>

```ini
[Model]
Type = NeuralNetwork
Learning_rate = 0.0001
Decay_rate = 0.96
Decay_steps = 1000
Epochs = 30000
Optimizer = adam
Loss = cross
Save_Path = "model.bin"
batch_size = 32
beta1 = 0.9
beta2 = 0.9999
epsilon = 1e-7

[DataSet]
BufferSize=100
TrainData="trainingSet.dat"
ValidData="trainingSet.dat"
LabelData="label.dat"

[inputlayer]
Type = input
Input_Shape = 1:1:62720
bias_initializer = zeros
Normalization = true

[outputlayer]
Type = fully_connected
Unit = 10
bias_initializer = zeros
Activation = softmax
Weight_Decay = l2norm
weight_Decay_Lambda = 0.005
```

If you want to use generator (option #2), then remove [DataSet] section, and provide dataset generator callbacks.

## How to execute

With `-Denable-app=true` and `-Dinstall-app=true` set in meson_options, execution files are installed in build/Application/Classification/jni or $(NNTRAINER_ROOT}/bin directory.

The training data set images are stored in Application/Classification/res/${Class Name}

```bash
$ mkdir Application/Classification/jni/test
$ cp build/Application/Classification/jni/nntrainer_classification* Application/Classification/jni/test/
$ cd Application/Classification/jni/test/
$ ./nntrainer_classification ../../res/Classification.ini ../../res/
```
If there is no trainingSet.dat, then it will start generate it with data files in ${data_path}.
