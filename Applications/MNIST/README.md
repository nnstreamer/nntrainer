# MNIST

In this example, we demonstrate training full neural network model with mnist dataset.

### Neural Network Model Configuration
Network model consists of two convolution layer, two pooling layer, one flatten layer and one fully connected layer.

![MNIST Model](/docs/images/mnist_model.png?raw=true)

### MNIST Data Set and DataBuffer
In order to make training short, we use subset of full mnist dataset, 100 images per class and save it into file named ```mnist_trainingSet.dat```. The input image size is 784 (28x28) and the label is an one-hot vector for classifying 10 digits. The i<sup>th</sup> image feature and label data must be at ```i x ((784 x sizeof(float) + label(10 x sizeof(float))``` byte position.

``` bash
...
0 0 0 223 225 0 0 0 0 0 0... 0  --> ith feature data (784xsizeof(float))
1 0 0 0 0 0 0 0 0 0         --> ith label data as an one-hot vector
0 0 0 0 0 0 0 0 224 123 0... 0  --> i+1th feature data (784xsizeof(float))
0 1 0 0 0 0 0 0 0 0         --> i+1th label data as an one-hot vector
...

```

In order to feed the training data to the neural network model, ```DataBufferFromCallback``` option is used and ```getData_train``` and ```getData_val``` functions are defined which reads the feature and label with batch_size and fill the databuffer. If there is no remaining data for the current epoch, then ```last``` will be true.

``` c++
/**
 * @brief      get data which size is batch for train
 * @param[out] inVec input Container to hold all the input data. Should not be freed by the user
 * @param[out] inLabel label Container to hold corresponding label data. Should not be freed by the user.
 * @param[out] last Container to notify if data is finished. Set true if no more
 * data to provide, else set false. Should not be freed by the user.
 * @param[in] user_data  User application's private data.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int getBatch_train(float **inVec, float **inLabel, bool *last,
                   void *user_data);
```

### MNIST Configuration file
The model configuration is in ```res/mnist.ini```. ADAM optimizer is used and the trained model is going to be saved in ```./model.bin```.

``` bash
# Model Section : Model
[Model]
Type = NeuralNetwork	    # Network Type NeuralNetwork
Learning_rate = 1e-4 	    # Learning Rate
Epochs = 1500		        # Epochs
Optimizer = adam 	        # Optimizer : adam (Adamtive Moment Estimation)
Loss = cross  		        # Loss function : cross ( for cross entropy )
Save_Path = "model.bin"  	# model path to save / read
batch_size = 32		        # batch size
beta1 = 0.9 		        # beta 1 for adam
beta2 = 0.999	            # beta 2 for adam
epsilon = 1e-7	            # epsilon for adam

# Layer Section : Name
[inputlayer]
Type = input
Input_Shape = 1:28:28

# Layer Section : Name
[conv2d_c1_layer]
Type = conv2d
kernel_size = 5,5
bias_initializer=zeros
Activation=sigmoid
weight_initializer = xavier_uniform
filters = 6
stride = 1,1
padding = 0,0

[pooling2d_p1]
Type=pooling2d
pool_size = 2,2
stride =2,2
padding = 0,0
pooling = average

[conv2d_c2_layer]
Type = conv2d
kernel_size = 5,5
bias_initializer=zeros
Activation=sigmoid
weight_initializer = xavier_uniform
filters = 12
stride = 1,1
padding = 0,0

[pooling2d_p2]
Type=pooling2d
pool_size = 2,2
stride =2,2
padding = 0,0
pooling = average

[flatten]
Type=flatten

[outputlayer]
Type = fully_connected
Unit = 10
weight_initializer = xavier_uniform
bias_initializer = zeros
Activation = softmax

```

### How to run
Once you compile the codes, you can run with

``` bash

$ cd ${build_dir}
$ export MNIST_RES=${build_dir}/res/app/MNIST
$ ./Applications/MNIST/jni/nntrainer_mnist .${MNIST_RES}/mnist.ini ${MNIST_RES}/mnist_trainingSet.dat
```

For the comparison, we provide Tensorflow code for same model and dataset in ```Tensorflow/Training_Keras.py```.

```dataset.py``` is for the data generation from the input data file, ```mnist_trainingSet.dat```.


You can run with

``` bash
$ python3 Training_Keras.py
```

```mnist_trainingSet.data``` must be in the same directory with ```Training_Keras.py```.


### Comparison with Tensorflow
This is the comparison with tensorflow-1.14.0 for the two cases. One is with zero initialization of weight and bias and the other is random weight initialization data using the default intializers for each layer from tensorflow. As can be seen with the result below, the results are same within the margin of error.

![image](/docs/images/image2020-9-1_8-23-40.png?raw=true)


![image](/docs/images/image2020-9-1_8-23-40.png?raw=true)
