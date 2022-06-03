---
title: Configuration ini
...

# Writing Configuration File

NNTrainer requires network configuration file which includes network layers and hyper-parameters. The format of configuration file is iniparser format which is commonly used. Keywords are not case sensitive and the line start with '#' will be ignored.

If you want more about iniparser, please visit
<https://github.com/ndevilla/iniparser>

## Sections

Configuration file consists of Two Sections, Network and Layer

### Model Section

Model section includes the hyper-parameters for the Network such type, epochs, loss, save path and batch size.

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
     * skip this property if no loss is desired for the model (this model will only support inference)

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

   Optimizer type to apply the gradients to weights. The default value is adam if the type is not used.
     * adam : Adaptive Moment Estimation
     * sgd : stochastic gradient decent

2. ```beta1 = <float>```

   beta1 parameter for adam optimizer. Only valid for adam. The default value is 0.9.


3. ```beta2 = <float>```

   beta2 parameter for adam optimizer. Only valid for adam. The default value is 0.999.

4. ```epsilon = <float>```

   Epsilon parameter for adam optimizer. Only valid for adam. The default value is 1.0e-7 is.

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

### Learning Rate Scheduler Section

Define the learing rate, decay steps and decay rate.

1. ```learning_rate = <float>```

   Initial learning rate to decay

2. ```decay_steps = <float> ```

   Decay steps

3. ```decay_rate = <float> ```

   Decay rate

Below is a sample Learning Rate scheduler Section.

```ini
# Learning Rate scheduler Section
[LearningRateScheduler]
type=constant
Learning_rate = 1e-4 	# Learning Rate
```

### Train Set Section

Define the type and path of the traing data file.
Start with "[train_set]"

1. ``` type = <string> ```

   Currently only file is supported.

2. ``` path = <string> ```

   Data path for training, The path is mandatory.

Below is a sample TrainSet section.

```ini
# TrainSet Section
[train_set]
Type = file
path = trainDataset.dat
```

### Validation Set Section

Define the type and path of the validation data file.
Start with "[valid_set]"

1. ``` type = <string> ```

   Currently only file is supported.

2. ``` path = <string> ```

   Data path for validation.

Below is a sample TrainSet section.

```ini
# TrainSet Section
[train_set]
Type = file
path = valDataset.dat
```

### Test Set Section

Define the type and path of the test data file.
Start with "[valid_set]"

1. ``` type = <string> ```

   Currently only file is supported.

2. ``` path = <string> ```

   Data path for test.

Below is a sample TrainSet section.

```ini
# TrainSet Section
[train_set]
Type = file
path = testDataset.dat
```
### Layer Section

Describe hyper-parameters for layer. Order of layers in the model follows the order of definition of layers here from top to bottom.

Start with "[ ${layer name} ]". This layer name must be unique throughout network model.

1. ```type = <string>```

   Type of Layer
     * input : input layer
     * fully_connected : fully connected layer
     * batch_normalization : batch normalization layer
     * conv2d : convolution 2D layer
     * pooling2d : pooling 2D layer
     * flatten : flatten layer
     * activation : activation layer
     * addition : addition layer
     * concat : concat layer
     * multiout : multiout layer
     * embedding : embedding layer
     * rnn : RNN layer
     * lstm : LSTM layer
     * split : split layer
     * gru : GRU layer
     * permute : permute layer
     * dropout : dropout layer
     * backbone_nnstreamer : backbone layer using nnstreamer
     * centroid_knn : centroid KNN layer
     * conv1d : convolution 1D layer
     * lstmcell : LSTM Cell layer
     * grucell : GRU Cell layer
     * rnncell : RNN Cell layer
     * zoneout_lstmcell : Zoneout LSTM Cell layer
     * preprocess_flip : preprocess flip layer
     * preprocess_translate : preprocess translate layer
     * preprocess_l2norm : preprocess l2norm layer
     * mse : MSE loss layer
     * cross_sigmoid : cross entropy with sigmoid loss layer
     * cross_softmax : Cross entropy with softmax loss layer

2. ```key = value```

   The table below shows the available keys and values for each layer type.
   There are two types of layers. One type includes commonly trainable weights and the other type does not include. The following are the available properties for each layer type which include commonly trainable weights:

Type | Key | Value | Default value | Description
---------- | --- | ----- | ----------- | -----------
(Universal properties)                                       |                             |                             |                         | Universal properties that applies to every layer
&#xfeff;                                                     | name                        | (string)                    |                         | An identifier for each layer
&#xfeff;                                                     | trainable                   | (boolean)                   | true                    | Allow weights to be trained if true
&#xfeff;                                                     | input_layers                | (string)                    |                         | Comma-seperated names of layers to be inputs of the current layer
&#xfeff;                                                     | input_shape                 | (string)                    |                         | Comma-seperated Formatted string as "channel:height:width". If there is no channel then it must be 1. First layer of the model must have input_shape. Other can be omitted as it is calculated at compile phase.
&#xfeff;                                                     | flatten                     | (boolean)                   |                         | Flatten shape from `c:h:w` to `1:1:c*h*w`
&#xfeff;                                                     | activation                  | (categorical)               |                         | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | loss                        | (float)                     | 0                       | Loss
&#xfeff;                                                     | weight_initializer          | (categorical)               | xavier_uniform          | Weight initializer
&#xfeff;                                                     |                             | zeros                       |                         | Zero initialization
&#xfeff;                                                     |                             | lecun_normal                |                         | LeCun normal initialization
&#xfeff;                                                     |                             | lecun_uniform               |                         | LeCun uniform initialization
&#xfeff;                                                     |                             | xavier_normal               |                         | Xavier normal initialization
&#xfeff;                                                     |                             | xavier_uniform              |                         | Xavier uniform initialization
&#xfeff;                                                     |                             | he_normal                   |                         | He normal initialization
&#xfeff;                                                     |                             | he_uniform                  |                         | He uniform initialization
&#xfeff;                                                     | bias_initializer            | (categorical)               | zeros                   | Bias initializer
&#xfeff;                                                     |                             | zeros                       |                         | Zero initialization
&#xfeff;                                                     |                             | lecun_normal                |                         | LeCun normal initialization
&#xfeff;                                                     |                             | lecun_uniform               |                         | LeCun uniform initialization
&#xfeff;                                                     |                             | xavier_normal               |                         | Xavier normal initialization
&#xfeff;                                                     |                             | xavier_uniform              |                         | Xavier uniform initialization
&#xfeff;                                                     |                             | he_normal                   |                         | He normal initialization
&#xfeff;                                                     |                             | he_uniform                  |                         | He uniform initialization
&#xfeff;                                                     | weight_regularizer          | (categorical)               |                         | Weight regularizer. Currently, only l2norm is supported
&#xfeff;                                                     |                             | l2norm                      |                         | L2 weight regularizer
&#xfeff;                                                     | weight_regularizer_constant | (float)                     | 1                       | Weight regularizer constant
`fully_connected`                                            |                             |                             |                         | Fully connected layer
&#xfeff;                                                     | unit                        | (unsigned integer)          |                         | Number of outputs
`conv1d`                                                     |                             |                             |                         | 1D Convolution layer
&#xfeff;                                                     | filters                     | (unsigned integer)          |                         | Number of filters
&#xfeff;                                                     | kernel_size                 | (unsigned integer)          |                         | Kernel size
&#xfeff;                                                     | stride                      | (unsigned integer)          | 1                       | Strides
&#xfeff;                                                     | padding                     | (categorical)               | valid                   | Padding type
&#xfeff;                                                     |                             | valid                       |                         | No padding
&#xfeff;                                                     |                             | same                        |                         | Preserve dimension
&#xfeff;                                                     |                             | (unsigned integer)          |                         | Size of padding applied uniformly to all side
&#xfeff;                                                     |                             | (array of unsigned integer of size 2) |                         | Padding for left, right
`conv2d`                                                     |                             |                             |                         | 2D Convolution layer
&#xfeff;                                                     | filters                     | (unsigned integer)          |                         | Number of filters
&#xfeff;                                                     | kernel_size                 | (array of unsigned integer) |                         | Comma-seperated unsigned integers for kernel size, `height, width`  respectively
&#xfeff;                                                     | stride                      | (array of unsigned integer) | 1, 1                    | Comma-seperated unsigned integers for strides, `height, width`  respectively
&#xfeff;                                                     | padding                     | (categorical)               | valid                   | Padding type
&#xfeff;                                                     |                             | valid                       |                         | No padding
&#xfeff;                                                     |                             | same                        |                         | Preserve height/width dimension
&#xfeff;                                                     |                             | (unsigned integer)          |                         | Size of padding applied uniformly to all side
&#xfeff;                                                     |                             | (array of unsigned integer of size 2) |                         | Padding for height, width
&#xfeff;                                                     |                             | (array of unsigned integer of size 4) |                         | Padding for top, bottom, left, right
`embedding`                                                  |                             |                             |                         | Embedding layer
&#xfeff;                                                     | in_dim                      | (unsigned integer)          |                         | Vocabulary size
&#xfeff;                                                     | out_dim                     | (unsigned integer)          |                         | Word embeddeing size
`rnn`                                                        |                             |                             |                         | RNN layer
&#xfeff;                                                     | unit                        | (unsigned integer)          |                         | Number of output neurons
&#xfeff;                                                     | hidden_state_activation     | (categorical)               | tanh                    | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | return_sequences            | (boolean)                   | false                   | Return only the last output if true, else return full output
&#xfeff;                                                     | dropout                     | (float)                     | 0                       | Dropout rate
&#xfeff;                                                     | integrate_bias              | (boolean)                   | false                   | Integrate bias_ih, bias_hh to bias_h
`rnncell`                                                    |                             |                             |                         | RNNCELL layer
&#xfeff;                                                     | unit                        | (unsigned integer)          |                         | Number of output neurons
&#xfeff;                                                     | hidden_state_activation     | (categorical)               | tanh                    | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | dropout                     | (float)                     | 0                       | Dropout rate
&#xfeff;                                                     | integrate_bias              | (boolean)                   | false                   | Integrate bias_ih, bias_hh to bias_h
`lstm`                                                       |                             |                             |                         | LSTM layer
&#xfeff;                                                     | unit                        | (unsigned integer)          |                         | Number of output neurons
&#xfeff;                                                     | hidden_state_activation     | (categorical)               | tanh                    | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | recurrent_activation        | (categorical)               | sigmoid                 | Activation type for recurrent step
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | return_sequences            | (boolean)                   | false                   | Return only the last output if true, else return full output
&#xfeff;                                                     | dropout                     | (float)                     | 0                       | Dropout rate
&#xfeff;                                                     | integrate_bias              | (boolean)                   | false                   | Integrate bias_ih, bias_hh to bias_h
&#xfeff;                                                     | max_timestep                | (unsigned integer)          |                         | Maximum timestep
`lstmcell`                                                   |                             |                             |                         | LSTMCELL layer
&#xfeff;                                                     | unit                        | (unsigned integer)          |                         | Number of output neurons
&#xfeff;                                                     | hidden_state_activation     | (categorical)               | tanh                    | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | recurrent_activation        | (categorical)               | sigmoid                 | Activation type for recurrent step
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | dropout                     | (float)                     | 0                       | Dropout rate
&#xfeff;                                                     | integrate_bias              | (boolean)                   | false                   | Integrate bias_ih, bias_hh to bias_h
`gru`                                                        |                             |                             |                         | GRU layer
&#xfeff;                                                     | unit                        | (unsigned integer)          |                         | Number of output neurons
&#xfeff;                                                     | hidden_state_activation     | (categorical)               | tanh                    | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | recurrent_activation        | (categorical)               | sigmoid                 | Activation type for recurrent step
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | return_sequences            | (boolean)                   | false                   | Return only the last output if true, else return full output
&#xfeff;                                                     | dropout                     | (float)                     | 0                       | Dropout rate
&#xfeff;                                                     | integrate_bias              | (boolean)                   | false                   | Integrate bias_ih, bias_hh to bias_h
&#xfeff;                                                     | reset_after                 | (boolean)                   | true                    | Apply reset gate before/after the matrix
`grucell`                                                    |                             |                             |                         | GRUCELL layer
&#xfeff;                                                     | unit                        | (unsigned integer)          |                         | Number of output neurons
&#xfeff;                                                     | reset_after                 | (boolean)                   | true                    | Apply reset gate before/after the matrix multiplication
&#xfeff;                                                     | hidden_state_activation     | (categorical)               | tanh                    | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | recurrent_activation        | (categorical)               | sigmoid                 | Activation type for recurrent step
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | dropout                     | (float)                     | 0                       | Dropout rate
&#xfeff;                                                     | integrate_bias              | (boolean)                   | false                   | Integrate bias_ih, bias_hh to bias_h
`zoneout_lstmcell`                                           |                             |                             |                         | ZONEOUTLSTMCELL layer
&#xfeff;                                                     | unit                        | (unsigned integer)          |                         | Number of output neurons
&#xfeff;                                                     | hidden_state_activation     | (categorical)               | tanh                    | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | recurrent_activation        | (categorical)               | sigmoid                 | Activation type for recurrent step
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | cell_state_zoneout_rate     | (float)                     | 0                       | Zoneout rate for cell state
&#xfeff;                                                     | hidden_state_zoneout_rate   | (float)                     | 0                       | zoneout rate for hidden state
&#xfeff;                                                     | integrate_bias              | (boolean)                   | false                   | Integrate bias_ih, bias_hh to bias_h
&#xfeff;

The following are the available properties for each layer type which does not include (`weight_initializer`, `bias_initializer`, `weight_regularizer`, `weight_regularizer_constant`) properties.


Type | Key | Value | Default value | Description
---------- | --- | ----- | ----------- | -----------
(Universal properties)                                       |                             |                             |                         | Universal properties that applies to every layer
&#xfeff;                                                     | name                        | (string)                    |                         | An identifier for each layer
&#xfeff;                                                     | trainable                   | (boolean)                   | true                    | Allow weights to be trained if true
&#xfeff;                                                     | input_layers                | (string)                    |                         | Comma-seperated names of layers to be inputs of the current layer
&#xfeff;                                                     | input_shape                 | (string)                    |                         | Comma-seperated Formatted string as "channel:height:width". If there is no channel then it must be 1. First layer of the model must have input_shape. Other can be omitted as it is calculated at compile phase.
&#xfeff;                                                     | flatten                     | (boolean)                   |                         | Flatten shape from `c:h:w` to `1:1:c*h*w`
&#xfeff;                                                     | activation                  | (categorical)               |                         | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
&#xfeff;                                                     | loss                        | (float)                     | 0                       | Loss
`input`                                                      |                             |                             |                         | Input layer
&#xfeff;                                                     | normalization               | (boolean)                   | false                   | Normalize input if true
&#xfeff;                                                     | standardization             | (boolean)                   | false                   | Standardize input if true
`batch_normalization`                                        |                             |                             |                         | Batch normalization layer
&#xfeff;                                                     | epsilon                     | (float)                     | 0.001                   | Small value to avoid divide by zero
&#xfeff;                                                     | moving_mean_initializer     | (categorical)               | zeros                   | Moving mean initializer
&#xfeff;                                                     |                             | zeros                       |                         | Zero initialization
&#xfeff;                                                     |                             | lecun_normal                |                         | LeCun normal initialization
&#xfeff;                                                     |                             | lecun_uniform               |                         | LeCun uniform initialization
&#xfeff;                                                     |                             | xavier_normal               |                         | Xavier normal initialization
&#xfeff;                                                     |                             | xavier_uniform              |                         | Xavier uniform initialization
&#xfeff;                                                     |                             | he_normal                   |                         | He normal initialization
&#xfeff;                                                     |                             | he_uniform                  |                         | He uniform initialization
&#xfeff;                                                     | moving_variance_initializer | (categorical)               | ones                    | Moving variance initializer
&#xfeff;                                                     |                             | zeros                       |                         | Zero initialization
&#xfeff;                                                     |                             | lecun_normal                |                         | LeCun normal initialization
&#xfeff;                                                     |                             | lecun_uniform               |                         | LeCun uniform initialization
&#xfeff;                                                     |                             | xavier_normal               |                         | Xavier normal initialization
&#xfeff;                                                     |                             | xavier_uniform              |                         | Xavier uniform initialization
&#xfeff;                                                     |                             | he_normal                   |                         | He normal initialization
&#xfeff;                                                     |                             | he_uniform                  |                         | He uniform initialization
&#xfeff;                                                     | gamma_initializer           | (categorical)               | ones                    | Gamma initializer
&#xfeff;                                                     |                             | zeros                       |                         | Zero initialization
&#xfeff;                                                     |                             | lecun_normal                |                         | LeCun normal initialization
&#xfeff;                                                     |                             | lecun_uniform               |                         | LeCun uniform initialization
&#xfeff;                                                     |                             | xavier_normal               |                         | Xavier normal initialization
&#xfeff;                                                     |                             | xavier_uniform              |                         | Xavier uniform initialization
&#xfeff;                                                     |                             | he_normal                   |                         | He normal initialization
&#xfeff;                                                     |                             | he_uniform                  |                         | He uniform initialization
&#xfeff;                                                     | beta_initializer            | (categorical)               | zeros                   | Beta initializer
&#xfeff;                                                     |                             | zeros                       |                         | Zero initialization
&#xfeff;                                                     |                             | lecun_normal                |                         | LeCun normal initialization
&#xfeff;                                                     |                             | lecun_uniform               |                         | LeCun uniform initialization
&#xfeff;                                                     |                             | xavier_normal               |                         | Xavier normal initialization
&#xfeff;                                                     |                             | xavier_uniform              |                         | Xavier uniform initialization
&#xfeff;                                                     |                             | he_normal                   |                         | He normal initialization
&#xfeff;                                                     |                             | he_uniform                  |                         | He uniform initialization
&#xfeff;                                                     | momentum                    | (float)                     | 0.99                    | Momentum for moving average in batch normalization
`pooling2d`                                                  |                             |                             |                         | Pooling layer
&#xfeff;                                                     | pooling                     | (categorical)               |                         | Pooling type
&#xfeff;                                                     |                             | max                         |                         | Max pooling
&#xfeff;                                                     |                             | average                     |                         | Average pooling
&#xfeff;                                                     |                             | global_max                  |                         | Global max pooling
&#xfeff;                                                     |                             | global_average              |                         | Global average pooling
&#xfeff;                                                     | pool_size                   | (array of unsigned integer) |                         | Comma-seperated unsigned intergers for pooling size, `height, width`  respectively
&#xfeff;                                                     | stride                      | (array of unsigned integer) | 1, 1                    | Comma-seperated unsigned intergers for stride, `height, width`  respectively
&#xfeff;                                                     | padding                     | (categorical)               | valid                   | Padding type
&#xfeff;                                                     |                             | valid                       |                         | No padding
&#xfeff;                                                     |                             | same                        |                         | Preserve height/width dimension
&#xfeff;                                                     |                             | (unsigned integer)          |                         | Size of padding applied uniformly to all side
&#xfeff;                                                     |                             | (array of unsigned integer of size 2) |                         | Padding for height, width
&#xfeff;                                                     |                             | (array of unsigned integer of size 4) |                         | Padding for top, bottom, left, right
`flatten`                                                    |                             |                             |                         | Flatten layer
`activation`                                                 |                             |                             |                         | Activation layer
&#xfeff;                                                     | activation                  | (categorical)               |                         | Activation type
&#xfeff;                                                     |                             | tanh                        |                         | Hyperbolic tangent
&#xfeff;                                                     |                             | sigmoid                     |                         | Sigmoid function
&#xfeff;                                                     |                             | relu                        |                         | Relu function
&#xfeff;                                                     |                             | softmax                     |                         | Softmax function
`addition`                                                   |                             |                             |                         | Addition layer
`concat`                                                     |                             |                             |                         | Concat layer
`multiout`                                                   |                             |                             |                         | Multiout layer
`split`                                                      |                             |                             |                         | Split layer
&#xfeff;                                                     | split_dimension             | (unsigned integer)          |                         | Which dimension to split. Split batch dimension is not allowed
`permute`                                                    |                             |                             |                         | Permute layer
`dropout`                                                    |                             |                             |                         | Dropout layer
&#xfeff;                                                     | dropout                     | (float)                     | 0                       | Dropout rate
`backbone_nnstreamer`                                        |                             |                             |                         | NNStreamer layer
&#xfeff;                                                     | model_path                  | (string)                    |                         | NNStreamer model path
`centroid_knn`                                               |                             |                             |                         | Centroid KNN layer
&#xfeff;                                                     | num_class                   | (unsigned integer)          |                         | Number of class
`preprocess_flip`                                            |                             |                             |                         | Preprocess flip layer
&#xfeff;                                                     | flip_direction              | (categorical)               |                         | Flip direction
&#xfeff;                                                     |                             | horizontal                  |                         | Horizontal direction
&#xfeff;                                                     |                             | vertical                    |                         | Vertiacl direction
&#xfeff;                                                     |                             | horizontal_and_vertical     | horizontal_and_vertical | Horizontal_and vertical direction
`preprocess_translate`                                       |                             |                             |                         | Preprocess translate layer
&#xfeff;                                                     | random_translate            | (float)                     |                         | Translate factor value
`preprocess_l2norm`                                          |                             |                             |                         | Preprocess l2norm layer
`mse`                                                        |                             |                             |                         | MSE loss layer
`cross_sigmoid`                                              |                             |                             |                         | Cross entropy with sigmoid loss layer
`cross_softmax`                                              |                             |                             |                         | Cross entropy with softmax loss layer


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

4. ```InputShape = <string>```

   Set the shape of the input layer for the backbone model. Only supported for ini backbones (nntrainer models).

5. ```InputLayer = <string>```

   Choose the start layer for the backbone. This allows taking a subgraph starting with the specified layer name as a backbone. Only supported for ini backbones (nntrainer models).

6. ```OutputLayer = <string>```

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
Oonly INI formatted files *.ini is supported to construct a model from a file.
Special sections [Model], [Optimizers], [train_set], [valid_set], [test_set] are respectively referring to model, optimizer and data provider objects. Rest of INI sections map to a layer. Keys and values from each section set properties of the layer. All keys and values are treated as case-insensitive.

Below is sample backbone section.
It takes 1 x 28 x 28 gray data (0~255) as an input. Adam optimizer is used to apply gradient and learning rate is 1.0e-4.

```ini
# Network Section : Network
[Model]
Type = NeuralNetwork # Network Type : Regression, KNN, NeuralNetwork
Epochs = 1500		   # Epochs
Loss = cross  		   # Loss function : mse (mean squared error)
                     #                 cross ( for cross entropy )
# Save_Path = "mnist_model.bin"  	  # model path to save / read
batch_size = 32		# batch size

[Optimizer]
Type = adam
beta1 = 0.9       # beta 1 for adam
beta2 = 0.999	   # beta 2 for adam
epsilon = 1e-7    # epsilon for adam

[LearningRateScheduler]
type=constant
Learning_rate = 1e-4 # Learning Rate

# Layer Section : Name
[inputlayer]
Type = input
Input_Shape = 1:28:28

# Layer Section : Name
[conv2d_c1_layer]
Type = conv2d
input_layers = inputlayer
kernel_size = 5,5
bias_initializer=zeros
Activation=sigmoid
weight_initializer = xavier_uniform
filters = 6
stride = 1,1
padding = 0,0

[pooling2d_p1]
Type=pooling2d
input_layers = conv2d_c1_layer
pool_size = 2,2
stride =2,2
padding = 0,0
pooling = average

[conv2d_c2_layer]
Type = conv2d
input_layers = pooling2d_p1
kernel_size = 5,5
bias_initializer=zeros
Activation=sigmoid
weight_initializer = xavier_uniform
filters = 12
stride = 1,1
padding = 0,0

[pooling2d_p2]
Type=pooling2d
input_layers = conv2d_c2_layer
pool_size = 2,2
stride =2,2
padding = 0,0
pooling = average

[flatten]
Type=flatten
input_layers = pooling2d_p2

[outputlayer]
Type = fully_connected
input_layers=flatten
Unit = 10		# Output Layer Dimension ( = Weight Width )
weight_initializer = xavier_uniform
bias_initializer = zeros
Activation = softmax 	# activation : sigmoid, softmax
```
The following restrictions must be adhered to:

 - Model file must have a `[Model]` section.
 - Model file must have at least one layer.
 - Valid keys must have valid properties. The invalid keys in each section result in an error.
 - All paths inside the INI file are relative to the INI file path unless the absolute path is stated.