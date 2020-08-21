# Writing Configuraion File

NNtrainer requires network configuration file which includes network layers and hyper-parameters. The format of configuration file is iniparser format which is commonly used. Keywards are not case sensitive and the line start with '#' will be ignored.

If you want more about iniparser, please visit
* https://github.com/ndevilla/iniparser

## Sections
Configuration file consists of Two Sections, Network and Layer

### Network Section
Network section includes the hyper-parameters about Network such as mini batch size, name of model file to save trained weight, epoch and etc.

Start with "[Model]"

1. ```type = <string>```

   Type of Network
     - regression : network for linear regression
     - knn : K-nearest neighbor
     - neuralnetwork : Deep Neural Network


2. ```learning_rate = <float>```

   Initial learning rate to decay


3. ```epoch = <unsigned int>```

   Number of epochs to train


4. ```optimizer = <string>```

   Optimizer to apply the gradients to weights.
     - adam : Adaptive Moment Estimation
     - sgd : stochastic gradient decent


5. ```cost = <string>```

   Cost (Loss) function
     - mse : mean squared error
     - cross : cross entropy
        Only allowed with sigmoid and softmax activation function


6. ```model = <string>```

   Model file path to save updated weights


7. ```minibatch = <unsigned int>```

   Mini batch size


8. ```beta1 = <float>```

   beta1 parameter for adam optimizer. Only valid for adam.   0.9 is default.


9. ```beta2 = <float>```

   beta2 parameter for adam optimizer. Only valid for adam.   0.999 is default.


10. ```epsilon = <float>```

     Epsilon parameter for adam optimizer. Only valid for adam.   1.0e-7 is defalut.



**example**


```bash
# Network Section : Network
[Model]
Type = NeuralNetwork
Learning_rate = 1e-4
Epoch = 1500
Optimizer = adam
Cost = cross
Model = "model.bin"
minibatch = 32
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7
```


### DataSet Section
Define data set. training, validation, test data set.

Start with "[ DataSet ]"

1. ```buffersize = <unsigned int>```

    Define Buffer size. usually it is greater than mini batch size.

	Data buffer thread keep read the data from file and store into Data Buffer, and meanwhile
	main thread get the training data from this Data Buffer and use them to train.
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

	```Class Name [i], Class name [i+1],...```


### Layer Section
Describe hyper-parameters for layer. Order of layers in the model follows the order of definition of layers here from top to bottom.

Start with "[ ${layer name} ]". This layer name must be unique throughout network model.

1. ```type = <string>```

   Type of Layer
     - input : input layer
     - conv2d : 2D convolution layer
     - pooling2d : 2D pooling layer
     - flatten : flatten layer
     - fully_connected : fully connected layer
     - batch_normalization : batch normalization layer
     - activation : activation layer


2. ```kernel_size = <unsigned int>,<unsigned int>```

   Kernel size for convolution layer


3. ```bias_init_zero = <bool>```

   token to initialize bias with zeros. Setting to False would initialize bias randomly.


4. ```normalization = <bool>```

	normalization on the input of this layer.


5. ```standardization = <bool>```

   	standardization on the input of this layer.


6. ```input_shape = <unsigned int>:<unsigned int>:<unsigned int>```

   shape of input
   it shouldn't be zero.


7. ```activation = <string>```

   set activation layer <string>
     - tanh : tanh function
     - sigmoid : sigmoid function
     - relu : relu function
     - softmax : softmax function


8. ```weight_decay = <string>```

   set weight decay
     - l2norm : L2Norm


9. ```weight_decay_lambda = <float>```

   coefficient for weight decay


10. ```unit = <unsigned int>```

     set the output layer for fully connected layer


11. ```weight_ini = <string>```

     set weight initialization method
       - lecun_normal : LeCun normal initialization
       - lecun_uniform : LeCun uniform initialization
       - xavier_normal : xavier normal iniitalization
       - xavier_uniform : xavier uniform initialization
       - he_normal : He normal initialization
       - he_uniform : He uniform initialization


12. ```filter = <unsigned int>```

     set filter size for convolution layer


13. ```stride = <unsigned int>,<unsigned int>```

     set stride for convolution and pooling layer


14. ```padding = <unsigned int>,<unsigned int>```

     set padding for convolution and pooling layer


15. ```pooling_size = <unsigned int>,<unsigned int>```

     set pooling size for pooling layer


16. ```pooling = <string>```

     define type of pooling
       - max : max pooling
       - average : average pooling
       - global_max : global max pooling
       - global_average : global average pooling


17. ```flatten = <bool>```

	flattens the output of this layer.
	Enabling this option is equivalent to attaching a flatten layer after the current layer.

18. ```epsilon = <float>```

    Epsilon parameter for batch normalization layer. Default is 0.001.


### Properties for layer

Each layer requires different properties.

 | Layer | Properties |
 |:-------:|:---|
 | conv2d |<ul><li>filter</li><li>kernel_size</li><li>stride</li><li>padding</li><li>normalization</li><li>standardization</li><li>input_shape</li><li>bias_init_zero</li><li>activation</li><li>flatten</li><li>weight_decay</li><li>weight_decay_lambda</li><li>weight_ini</li></ul>|
 | pooling2d | <ul><li>pooling</li><li>pooling_size</li><li>stride</li><li>padding</li></ul> |
 | flatten | - |
 | fully_connected | <lu><li>unit</li><li>normalization</li><li>standardization</li><li>input_shape</li><li>bias_init_zero</li><li>activation</li><li>flatten</li><li>weight_decay</li><li>weight_decay_lambda</li><li>weight_ini</li></lu>|
 | input | <lu><li>normalization </li><li>standardization</li><li>input_shape</li><li>flatten</li></lu>|
 | batch_normalization | <lu><li>epsilon</li><li>flatten</li></lu> |


**example**


```bash
[conv2d_c2_layer]
Type = conv2d
kernel_size = 5,5
bias_init_zero=true
Activation=sigmoid
weight_ini = xavier_uniform
filter = 12
stride = 1,1
padding = 0,0

[outputlayer]
Type = fully_connected
Unit = 10
weight_ini = xavier_uniform
bias_init_zero = true
Activation = softmax
```

### Configuraion file example
This has one input layer, two convolution layers, two pooling layers, one flatten layer and one fully connected layer to classify mnist example.

It takes 1 x 28 x 28 gray data (0~255) as an input. Adam optimizer is used to apply gradient and learning rate is 1.0e-4.

```batch
# Network Section : Network
[Model]
type = NeuralNetwork
learning_rate = 1e-4
epoch = 1500
optimizer = adam
cost = cross
model = "model.bin"
minibatch = 32
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

# Layer Section : Name
[inputlayer]
type = input
input_shape = 1:28:28

# Layer Section : Name
[conv2d_c1_layer]
type = conv2d
kernel_size = 5,5
bias_init_zero=true
activation=sigmoid
weight_ini = xavier_uniform
filter = 6
stride = 1,1
padding = 0,0

[pooling2d_p1]
type=pooling2d
pooling_size = 2,2
stride =2,2
padding = 0,0
pooling = average

[conv2d_c2_layer]
type = conv2d
kernel_size = 5,5
bias_init_zero=true
activation=sigmoid
weight_ini = xavier_uniform
filter = 12
stride = 1,1
padding = 0,0

[pooling2d_p2]
type=pooling2d
pooling_size = 2,2
stride =2,2
padding = 0,0
pooling = average

[flatten]
type=flatten

[outputlayer]
type = fully_connected
unit = 10
weight_ini = xavier_uniform
bias_init_zero = true
activation = softmax
```


