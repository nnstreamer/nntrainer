---
title: Configuration ini
...

# Writing Configuration File

NNTrainer requires network configuration file which includes network layers and hyper-parameters. The format of configuration file is iniparser format which is commonly used. Keywords are not case sensitive and the line start with '#' will be ignored.

If you want more about iniparser, please visit
<https://github.com/ndevilla/iniparser>

## Sections

Configuration file consists of Two Sections, Network and Layer

### Network Section

Network section includes the hyper-parameters about Network such as batch size, name of model file to save trained weight, epochs and etc.

Start with "[Model]"

1. ```type (mandatory) = <string>```

   Type of Network
     * regression : network for linear regression
     * knn : K-nearest neighbor
     * neuralnetwork : Deep Neural Network

2. ```epochs = <unsigned int>```

   Number of epochs to train

   Create a new section for this

3. ```loss = <string>```

   Loss function
     * mse : mean squared error
     * cross : cross entropy
        Only allowed with sigmoid and softmax activation function
     * none : no loss for the model (this model will only support inference)

4. ```save_path = <string>```

   Model file path to save updated weights

5. ```batch_size = <unsigned int>```

   Mini batch size

Below is sample Network section.

```ini
# Network Section : Network
[Model]
Type = NeuralNetwork
Epochs = 1500
Loss = cross
Save_Path = "model.bin"
batch_size = 32
```

### Optimizer Section

Define the optimizer to be used for training. This is an optional section needed only for training, and can be skipped for inference.

Start with "[ Optimizer ]"

1. ```type = <string>```

   Optimizer type to apply the gradients to weights.
     * adam : Adaptive Moment Estimation
     * sgd : stochastic gradient decent

2. ```learning_rate = <float>```

   Initial learning rate to decay

3. ```beta1 = <float>```

   beta1 parameter for adam optimizer. Only valid for adam.   0.9 is default.

4. ```beta2 = <float>```

   beta2 parameter for adam optimizer. Only valid for adam. 0.999 is default.

5. ```epsilon = <float>```

     Epsilon parameter for adam optimizer. Only valid for adam. 1.0e-7 is default.

Below is a sample Optimizer section.

```ini
# Optimizer Section
[Optimizer]
Type = adam
Learning_rate = 1e-4
batch_size = 32
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7
```

### DataSet Section

Define data set. training, validation, test data set.

Start with "[ DataSet ]"

1. ```buffersize = <unsigned int>```

    Define Buffer size. usually it is greater than batch size.

    Data buffer thread keeps reading the data from the file and stores the data into the data buffer.
    Meanwhile main thread gets the training data from this data buffer and feeds it to the model.
    This keyword defines the size of Data Buffer.

2. ```traindata = <string>```

    training data file path.   The data must be saved as following

    ```feature data[i], label data[i], feature data[i+1], label data[i+1], ...```

3. ```validdata = <string>```

    validation data file path.   The data must be saved as following

    ```feature data[i], label data[i], feature data[i+1], label data[i+1], ...```

4. ```testdata = <string>```

    test data file path.   The data must be saved as following

    ```feature data[i], label data[i], feature data[i+1], label data[i+1], ...```

5. ```labeldata = <string>```

    label data file path. The data must be saved as following

    ```class Name [i], class name [i+1],...```

### Layer Section

Describe hyper-parameters for layer. Order of layers in the model follows the order of definition of layers here from top to bottom.

Start with "[ ${layer name} ]". This layer name must be unique throughout network model.

1. ```type = <string>```

   Type of Layer
     * input : input layer
     * conv2d : 2D convolution layer
     * pooling2d : 2D pooling layer
     * flatten : flatten layer
     * fully_connected : fully connected layer
     * batch_normalization : batch normalization layer
     * activation : activation layer

2. ```kernel_size = <unsigned int>,<unsigned int>```

   Kernel size for convolution layer

3. ```bias_init_zero = <bool>```

   token to initialize bias with zeros. Setting to False would initialize bias randomly.

4. ```normalization = <bool>```

    normalization on the input of this layer.

5. ```standardization = <bool>```

    standardization on the input of this layer.

6. ```input_shape = <unsigned int>:<unsigned int>:<unsigned int>```

   shape of input (shouldn't be zero).

7. ```activation = <string>```

   set activation layer
     * tanh : tanh function
     * sigmoid : sigmoid function
     * relu : ReLU function
     * softmax : softmax function

8. ```weight_regularizer = <string>```

   set weight decay
     * l2norm : L2 normalization

9. ```weight_regularizer_constant = <float>```

   coefficient for weight decay

10. ```unit = <unsigned int>```

     set the output layer for fully connected layer

11. ```weight_initializer = <string>```

     set weight initialization method
       * zeros : Zero initialization
       * lecun_normal : LeCun normal initialization
       * lecun_uniform : LeCun uniform initialization
       * xavier_normal : xavier normal initialization
       * xavier_uniform : xavier uniform initialization
       * he_normal : He normal initialization
       * he_uniform : He uniform initialization

12. ```filters = <unsigned int>```

     set filters size for convolution layer

13. ```stride = <unsigned int>,<unsigned int>```

     set stride for convolution and pooling layer

14. ```padding = <unsigned int>,<unsigned int>```

     set padding for convolution and pooling layer

15. ```pool_size = <unsigned int>,<unsigned int>```

     set pooling size for pooling layer

16. ```pooling = <string>```

     define type of pooling
       * max : max pooling
       * average : average pooling
       * global_max : global max pooling
       * global_average : global average pooling

17. ```flatten = <bool>```

    flattens the output of this layer.

    Enabling this option is equivalent to attaching a flatten layer after the current layer.

18. ```epsilon = <float>```

    Epsilon parameter for batch normalization layer. Default is 0.001.

### Properties for layer

Each layer requires different properties.

 | Layer | Properties |
 |:-------:|:---|
 | conv2d |<ul><li>filters</li><li>kernel_size</li><li>stride</li><li>padding</li><li>normalization</li><li>standardization</li><li>input_shape</li><li>bias_init_zero</li><li>activation</li><li>flatten</li><li>weight_regularizer</li><li>weight_regularizer_constant</li><li>weight_initializer</li></ul>|
 | pooling2d | <ul><li>pooling</li><li>pool_size</li><li>stride</li><li>padding</li></ul> |
 | flatten | - |
 | fully_connected | <lu><li>unit</li><li>normalization</li><li>standardization</li><li>input_shape</li><li>bias_initializer</li><li>activation</li><li>flatten</li><li>weight_regularizer</li><li>weight_regularizer_constant</li><li>weight_initializer</li></lu>|
 | input | <lu><li>normalization </li><li>standardization</li><li>input_shape</li><li>flatten</li></lu>|
 | batch_normalization | <lu><li>epsilon</li><li>flatten</li></lu> |

Below is sample for layers to define a model.

```ini
[conv2d_c2_layer]
Type = conv2d
kernel_size = 5,5
bias_initializer=zeros
Activation=sigmoid
weight_initializer = xavier_uniform
filters = 12
stride = 1,1
padding = 0,0

[outputlayer]
Type = fully_connected
Unit = 10
weight_initializer = xavier_uniform
bias_initializer = zeros
Activation = softmax
```

### Backbone section

This allows to describe another model, termed as backbone, to be used in the model described by the current ini file.
The backbone to be used can be described with another ini configuration file path, or with model file for external frameworks.
Support for backbones of external framework for Tensorflow-Lite is provided natively with Tensorflow-Lite framework.
Support for backbones of other external frameworks is done using nnstreamer and its plugin.
When using nnstreamer for external framework, ensure to add the corresponding baseline ML framework and its corresponding nnstreamer plugin as a dependency or install manually.
For example, when using PyTorch based model as a backbone, both the packages *PyTorch* and *nnstreamer-pytorch* must be installed.

Backbones made of nntrainer models, described using ini, support training the backbone also.
However, this is not supported with external frameworks.
It is possible to describe a backbone inside a backbone ini configuration file, as well as listing down multiple backbones to build a single model.
For backbone ini configuration file, Model and Dataset sections are ignored.

Describing a backbone is very similar to describing a layer.
Start with a "[ ${layer name} ]" which must be unique throughtout the model. In case of backbone, the name of the backbone is prepended to the name of all the layers inside the backbone.

1. ```backbone = <string>```

   Path of the backbone file. Supported model files:
    * .ini - NNTrainer models
    * .tflite - Tensorflow-Lite models
    * .pb / .pt / .py / .circle etc via NNStreamer (corresponding nnstreamer plugin required)

2. ```trainable = <bool>```

   If this backbone must be trained (defaults to false). Only supported for ini backbones (nntrainer models).

3. ```Preload = <bool>```

   Load pretrained weights from the saved modelfile of backbone (defaults to false). Only supported for ini backbone (nntrainer models).

4. ```ScaleSize = <float>```

   Scale the size of the layers from backbone (defaults to 1.0). This applies for fully connected and convolution layer for now, where the units and the output channels are scaled respectively. Only supported for ini backbone (nntrainer models). If the model is being scaled, it cannot be preloaded from the saved modelfile. Only of the two options, ScaleSize and Preload, must be set at once.

5. ```InputShape = <string>```

   Set the shape of the input layer for the backbone model. Only supported for ini backbones (nntrainer models).

6. ```InputLayer = <string>```

   Choose the start layer for the backbone. This allows taking a subgraph starting with the specified layer name as a backbone. Only supported for ini backbones (nntrainer models).

7. ```OutputLayer = <string>```

   Choose the end layer for the backbone. This allows taking a subgraph ending with the specified layer name as a backbone. Only supported for ini backbones (nntrainer models).
``
Below is sample backbone section.

```ini
# Model Section
[Model]
...

# block1
[block1]
backbone = resnet_block.ini
trainable = false

# block2
[block2]
backbone = resnet_block.ini
trainable = true

[outputlayer]
type = fully_connected
unit = 10
activation = softmax
```

### Configuration file example

This has one input layer, two convolution layers, two pooling layers, one flatten layer and one fully connected layer to classify MNIST example.

It takes 1 x 28 x 28 gray data (0~255) as an input. Adam optimizer is used to apply gradient and learning rate is 1.0e-4.

```ini
# Model Section
[Model]
type = NeuralNetwork
learning_rate = 1e-4
epochs = 1500
optimizer = adam
loss = cross
Save_Path = "model.bin"
batch_size = 32
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

# Layer Section
[inputlayer]
type = input
input_shape = 1:28:28

[conv2d_c1_layer]
type = conv2d
kernel_size = 5,5
bias_initializer=zeros
activation=sigmoid
weight_initializer = xavier_uniform
filters = 6
stride = 1,1
padding = 0,0

[pooling2d_p1]
type=pooling2d
pool_size = 2,2
stride =2,2
padding = 0,0
pooling = average

[conv2d_c2_layer]
type = conv2d
kernel_size = 5,5
bias_initializer=zeros
activation=sigmoid
weight_initializer = xavier_uniform
filters = 12
stride = 1,1
padding = 0,0

[pooling2d_p2]
type=pooling2d
pool_size = 2,2
stride =2,2
padding = 0,0
pooling = average

[flatten]
type=flatten

[outputlayer]
type = fully_connected
unit = 10
weight_initializer = xavier_uniform
bias_initializer = zeros
activation = softmax
```
