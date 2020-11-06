# Classification of hand-drawn images

Draw Classification example performs classification on the hand-drawn images and maps them to different emotions.
The application uses a Mobilenet-V2 model pre-trained on ms-coco dataset.
Transfer learning is used to take advantage of the pre-training model to learn the new information of the
hand-drawn image dataset efficiently.

## Application Details

Below are the steps to perform transfer learning in this application:

1. The pre-trained model is converted from pb format (Tensorflow trained model file) to TFLite format (Tensorflow-Lite frozen model). The last layer of the model is removed in this conversion to allow transfer learning to be done.
2. The training data is passed through the tflite model and the outputs are cached. As the transfer learning only trains the last few layers, the outputs from the non-trainable layers are cached to save computation. This is achieved using the single-shot C-API of NNStreamer for TIZEN. In case of non-tizen OS, Tensorflow-Lite framework c++ API are directly used.
3. The cached outputs from the step above act as the inputs for the NNTrainer model. The NNTrainer model consists of 2 fully connected layers with Softmax along with training loss and optimizer. The training for this added layer is performed over the cached data for multiple epochs.

Once the training has been performed, the TFLite model (with the last layer removed) and the NNTrainer model (with the newly trained last layer) combined forms the new trained model for the classification of the hand-drawn images.

### NNTrainer Training Details

Two fully connected layers are added to classify the features obtained from pre-trained Mobilenet-V2 model. The number of classes are reduced to 3 while the number of input features remains same with Mobilenet-V2. Training and test set contains 5 training images and 2 test images per class, totalling to 15 training images and 6 test images respectively.

A Softmax activation is added to the fully connected layer. Mean Squared Error loss function and Stochastic Gradient Decent is used as the loss function and optimizer respectively. Training is performed for 1000 epochs with the training configuration described below.

### NNTrainer Model Configuration

The configuration of the example is below,
![image](/docs/images/02a7ee80-f0ce-11e9-97b8-bcc19b7eb222.png?raw=true)

### Resource Data

Training and test data for the application is located in `Applications/TransferLearning/Draw_Classification/res`.

```bash
$ pwd
  ./Applications/TransferLearning/Draw_Classification/res
$ ls
happy  sad  soso  ssd_Mobilenet_v2_coco_feature.tflite  testset  Training.ini
```

```happy, sad, soso``` contains the images to train (5 for each class) and ```ssd_mobile_v2_coco_feature.tflite``` is for feature extractor. ```testset``` images are used to evaluate accuracy (2 images per class).

Training and test dataset are as shown below:
![image](/docs/images/7944ec00-f0ce-11e9-87af-aea730bcd0f5.png?raw=true)

### Configuration File

Configuration for model of the application is described in `Applications/TransferLearning/Draw_Classification/res/Training.ini`.

```bash
# Model Section : Model ignored line started with '#'
[Model]                       # Model Configuration
Type = NeuralNetwork          # Network type
Learning_rate = 0.01          # Learning rate
Epoch = 1000                  # Epoch
Optimizer = sgd               # Optimizer to apply gradient
Loss = cross                  # Cost function. last layer uses softmax as an activation.
                              # So, it is Cross-Entropy with softmax
save_path = "model.bin"       # Updated weights are stored in files named 'model.bin'
batch_size = 1                # Batch size (used it throughout layers)

# Layer Section : Name        # Layer Configuration
[inputlayer]                  # Layer Name
Type = input                  # Layer type
Input_Shape = 1:1:128         # Input shape (without batch size)

[fc1layer]                    # Layer Name
Type = fully_connected        # Layer Type
Unit = 20                     # output unit
Bias_initializer = zeros      # bias initialized to 0
Activation = sigmoid          # activation

[outputlayer]                 # Layer Name
Type = fully_connected        # Layer Type
Unit = 3                      # output unit
Bias_initializer = zeros      # bias initialized to 0
Activation = softmax          # activation
```

### How to run

Once the application has been build with meson, use the instructions below to run:

```bash
$ pwd
  build
$ ./Applications/TransferLearning/Draw_Classification/jni/nntrainer_training ../Applications/TransferLearning/Draw_Classification/res/Training.ini ../Applications/TransferLearning/Draw_Classification/res/

```

### Results

The training reduces the training loss starting from `1.08` to `0.0048`.
The test results for the 8 test cases are below. Top-1 prediction is used to check the results.

![image](/docs/images/16555400-f0d2-11e9-959b-f61935fefd5a.png?raw=true)
