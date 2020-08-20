# Learning with Feature Extractor (Fully Connected Layer)

Unlike using kNN for the classifier at #55 , we could add new layer at the end of feature extractor.
In this toy example, fully connected layer is added to classify as below. The size of the feature is also same with previous example and the training set and test set are also same. ( 3 classes, 5 training set and 2 test set of each )
Only fully connected layers are updatable and Mobilenet ssd v2 is used for feature extractor like previous example and all the testing and training is done on the Galaxy S8.

The activation is softmax. Mean Squared Error loss function and Stochastic Gradient Decent is used for the loss function and optimizer. Two fully connected layers are added for the hidden layer and iterate 300 times to make is more simple.

### Neural Network Configuration
The configuration of the example is below,
<p align = "center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/docs/images/02a7ee80-f0ce-11e9-97b8-bcc19b7eb222.png" width="400" height="250" > </p>

### Resource Data

This is in Applications/Training/res/Training.ini

```bash
$ pwd
  ./Applications/Training/res
$ ls
happy  sad  soso  ssd_mobilenet_v2_coco_feature.tflite  testset  Training.ini
```

```happy, sad, soso``` is the images to train and ```ssd_mobile_v2_coco_feature.tflite``` is for feature extractor. The last layer of tflite file is removed and replaced layer which is in Trainging.ini. ```testset``` images are used to evaluate accuracy.


Training set and test set are below
<p align = "center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/docs/images/7944ec00-f0ce-11e9-87af-aea730bcd0f5.png" >
</p>

**Configuration File**

Training.ini coniguration file is:

```bash
# Network Section : Network ignored line started with '#'
[Network]                     # Network Configuration
Type = NeuralNetwork          # Network type
Learning_rate = 0.01          # Learning rate
Epoch = 100                   # Epoch
Optimizer = sgd               # Optimizer to apply gradient
Cost = cross                  # Cost function. last layer uses softmax as an activation.
                              # So, it is Cross-Entropy with softmax
save_path = "model.bin"       # Updated weights are stored in files named 'model.bin'
batch_size = 1                # Mini batch size (used it throughout layers)

# Layer Section : Name        # Layer Configuration
[inputlayer]                  # Layer Name
Type = input                  # Layer type
Input_Shape = 1:1:128         # Input shape (without batch size)

[fc1layer]                    # Layer Name
Type = fully_connected        # Layer Type
Unit = 20                     # output unit
Bias_init_zero = true         # bias initialization 
Activation = sigmoid          # activation

[outputlayer]                 # Layer Name
Type = fully_connected        # Layer Type
Unit = 3                      # output unit
Bias_init_zero = true         # bias initialization
Activation = softmax          # activation
```

### How to run

It takes the input image and ```ssd_mobile_v2_coco_feature.tflite``` is used to extract features.
Feature data is feeded into network described earlier, and it is uesd to train network.

We can run this as below

```bash
$ pwd
  build
$ cd ..
$ build/Applications/Training/jni/nntrainer_training Application/Training/res/Trainig.ini Application/Training/res/

```

### Results

After Iterating 300 times, the change of L2 Norm of the Loss function is below.
<p align = "center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/docs/images/d42b1300-f0cf-11e9-9b6f-6db30def4684.png" width="500" height="300">
</p>

and the test results for the 8 test cases are below. Step function is used to make more clear.
As you can see, the test result is ok.

<p align ="center">
<img src="https://github.com/nnstreamer/nntrainer/blob/master/docs/images/16555400-f0d2-11e9-959b-f61935fefd5a.png" width ="500" height="180">
</p>

